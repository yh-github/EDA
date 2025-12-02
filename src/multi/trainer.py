import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, \
    accuracy_score
import logging
import torch.nn.functional as fnn
import os
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from multi.config import MultiExpConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss with Hard Negative Mining.
    """

    def __init__(self, temperature=0.07, hard_negative_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor):
        device = features.device
        batch_size, seq_len, _ = features.shape

        features = fnn.normalize(features, dim=-1)

        # [B, S, S] - Compute similarity for entire batch at once
        sim_matrix = torch.bmm(features, features.transpose(1, 2)) / self.temperature

        # Mask out self-contrast (diagonal) and padding
        # mask_2d: True if both i and j are valid tokens
        mask_2d = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        # Identity matrix for diagonal masking
        eye = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        valid_mask = mask_2d & (eye == 0)

        # Numerical stability (LogSumExp trick)
        sim_matrix_max, _ = torch.max(sim_matrix, dim=2, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()

        # [B, S, S] -> 1 if labels[i] == labels[j]
        label_mask = torch.eq(labels.unsqueeze(2), labels.unsqueeze(1)).float()

        # Filter noise (-1)
        not_noise_mask = (labels != -1).float().unsqueeze(2)

        # Final Positive Mask: Same Label AND Valid AND Not Diagonal AND Not Noise
        final_mask = label_mask * valid_mask.float() * not_noise_mask

        # Denominator: Sum exp(sim) over all valid j != i
        # Hard Negative Mining: Upweight negatives in the denominator
        exp_sim = torch.exp(sim_matrix) * valid_mask.float()

        if self.hard_negative_weight != 1.0:
            # Negatives are valid items that are NOT positives (label_mask is 1 for pos)
            # Note: label_mask includes self-matches if we didn't mask diagonal, but valid_mask handles diagonal.
            # So valid negatives = valid_mask & (label_mask == 0)
            negative_mask = valid_mask & (label_mask == 0)

            # Apply weight to negatives
            weight_matrix = torch.ones_like(sim_matrix)
            weight_matrix[negative_mask.bool()] = self.hard_negative_weight
            exp_sim = exp_sim * weight_matrix

        log_prob = sim_matrix - torch.log(exp_sim.sum(2, keepdim=True) + 1e-6)

        # Mean log prob of positives
        sum_log_prob_pos = (final_mask * log_prob).sum(2)
        num_pos = final_mask.sum(2)

        mean_log_prob_pos = sum_log_prob_pos / (num_pos + 1e-6)

        # Loss is -mean over anchors that have positives
        has_positives = num_pos > 0
        if has_positives.sum() > 0:
            return -mean_log_prob_pos[has_positives].mean()

        return torch.tensor(0.0, device=device, requires_grad=True)


class MultiTrainer:
    def __init__(self, model, config: MultiExpConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        def get_scheduler() -> LRScheduler:
            if config.scheduler_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=config.scheduler_t0,
                    T_mult=config.scheduler_t_mult
                )
            else:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=config.scheduler_patience
                )

        self.scheduler = get_scheduler()

        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        if config.use_focal_loss:
            self.adj_criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        else:
            weight_tensor = torch.tensor([config.pos_weight]).to(config.device)
            self.adj_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight_tensor)

        self.cycle_criterion = nn.CrossEntropyLoss(reduction='none')

        if config.use_contrastive_loss:
            self.contrastive_criterion = SupervisedContrastiveLoss(
                temperature=config.contrastive_temperature,
                hard_negative_weight=config.hard_negative_weight
            )

        # Signal handling injection point
        self.stop_requested = False

    def request_stop(self):
        """Allows external agents (like signal handlers) to request a graceful stop."""
        logger.info("Trainer received stop request. Finishing current epoch/batch...")
        self.stop_requested = True

    def fit(self, train_loader, val_loader, epochs: int, trial=None, save_path: str = None, stop_callback=None,
            metric_to_track='pr_auc'):
        """
        Unified training loop with Early Stopping, Optuna reporting, and Model Saving.
        """
        # Lazy import to avoid hard dependency if not tuning
        import optuna

        best_score = -1.0
        patience = self.config.early_stopping_patience
        patience_counter = 0

        logger.info(
            f"Starting training for {epochs} epochs. Early Stopping on '{metric_to_track}' (Patience: {patience})")

        for epoch in range(1, epochs + 1):
            # 1. Check external stop signal
            if stop_callback and stop_callback():
                self.request_stop()

            if self.stop_requested:
                logger.info("Stop requested. Exiting training loop.")
                break

            # 2. Train & Evaluate
            train_loss = self.train_epoch(train_loader, epoch)
            metrics = self.evaluate(val_loader)

            # Extract metrics
            current_score = metrics.get(metric_to_track, 0.0)
            val_f1 = metrics['f1']
            val_pr_auc = metrics['pr_auc']
            val_loss = metrics['val_loss']

            # Subtask Metrics
            cycle_f1 = metrics.get('cycle_f1', 0.0)
            cycle_acc = metrics.get('cycle_acc', 0.0)
            cycle_pr_auc = metrics.get('cycle_pr_auc', 0.0)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {train_loss:.3f}/{val_loss:.3f} | "
                f"ADJ F1: {val_f1:.3f} | ADJ PR: {val_pr_auc:.3f} | "
                f"CYC Acc: {cycle_acc:.3f} | CYC F1: {cycle_f1:.3f} | CYC PR: {cycle_pr_auc:.3f}"
            )

            # Log Detailed Class Breakdown occasionally or if requested
            if 'cycle_report' in metrics:
                report = metrics['cycle_report']
                # Create a concise string for logging
                # Assuming classes like 0:None, 1:Monthly, 2:Weekly...
                # We want to see F1 per class to know if we are ignoring rare classes
                breakdown_str = " | ".join([f"C{c}: {stats['f1-score']:.2f} (n={stats['support']})"
                                            for c, stats in report.items() if
                                            isinstance(c, (int, str)) and c != 'accuracy'])
                logger.info(f"  [Class Breakdown] {breakdown_str}")

            # 3. Optuna Reporting (Report the metric we are optimizing for)
            if trial:
                trial.report(current_score, epoch)
                if trial.should_prune():
                    logger.info(f"Trial pruned by Optuna based on {metric_to_track}.")
                    raise optuna.TrialPruned()

            # 4. Early Stopping & Saving
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0

                if save_path:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    checkpoint = {
                        "config": self.config,
                        "state_dict": self.model.state_dict(),
                        "best_score": best_score,
                        "metric_name": metric_to_track,
                        "metrics": metrics,
                        "epoch": epoch
                    }
                    torch.save(checkpoint, save_path)
                    # logger.info(f"  --> New Best Model Saved ({metric_to_track}: {best_score:.4f})")
                else:
                    pass
                    # logger.info(f"  --> New Best {metric_to_track}: {best_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⛔ Early stopping triggered at epoch {epoch}")
                    break

        return best_score

    def _compute_loss_with_pattern_ids(self, batch, adj_logits, cycle_logits, embeddings):
        mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
        eye = torch.eye(adj_logits.shape[1], device=self.config.device).unsqueeze(0)
        mask_2d = mask_2d & (eye == 0)

        adj_loss = self.adj_criterion(adj_logits, batch['adjacency_target'])
        adj_loss = (adj_loss * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)

        cycle_loss = self.cycle_criterion(
            cycle_logits.view(-1, self.config.num_classes),
            batch['cycle_target'].view(-1)
        )
        mask_1d = batch['padding_mask'].view(-1)
        cycle_loss = (cycle_loss * mask_1d.float()).sum() / mask_1d.sum().clamp(min=1)

        con_loss = torch.tensor(0.0, device=adj_loss.device)
        if self.config.use_contrastive_loss and 'pattern_ids' in batch:
            con_loss = self.contrastive_criterion(embeddings, batch['pattern_ids'], batch['padding_mask'])

        total_loss = adj_loss + cycle_loss + (self.config.contrastive_loss_weight * con_loss)
        return total_loss, adj_loss, cycle_loss, con_loss

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        accumulation_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        num_batches_processed = 0
        num_batches_skipped = 0
        batches_since_step = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.stop_requested:
                break

            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            try:
                # --- The Critical Recovery Block ---
                with torch.amp.autocast('cuda'):
                    adj_logits, cycle_logits, embeddings = self.model(batch)
                    loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)
                    loss = loss / accumulation_steps

                # Backward
                self.scaler.scale(loss).backward()
                batches_since_step += 1

                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"⚠️ NaN loss detected at Epoch {epoch_idx}, Batch {batch_idx}. Skipping step.")
                    self.optimizer.zero_grad()
                    batches_since_step = 0  # Reset accumulation
                    num_batches_skipped += 1  # Track NaN batches separately
                    continue

            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                num_batches_skipped += 1
                continue

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    batches_since_step = 0  # Reset accumulation counter
                    num_batches_skipped += 1
                    continue
                raise e

            # Accumulation Step
            if batches_since_step >= accumulation_steps:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                batches_since_step = 0

            total_loss += loss.item() * accumulation_steps
            num_batches_processed += 1

        # Handle remaining gradients
        if batches_since_step > 0 and not self.stop_requested:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return total_loss / max(1, num_batches_processed)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        # Adjacency Collections
        all_pred_edges = []
        all_pred_probs = []
        all_true_edges = []

        # Cycle Collections
        all_cycle_preds = []
        all_cycle_probs = []  # Store full probability distribution for PR-AUC
        all_cycle_targets = []

        total_val_loss = 0.0

        for batch in dataloader:
            if self.stop_requested: break

            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            try:
                with torch.amp.autocast('cuda'):
                    adj_logits, cycle_logits, embeddings = self.model(batch)
                    loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)

                total_val_loss += loss.item()

                # --- Adjacency Metrics ---
                probs = torch.sigmoid(adj_logits)
                preds = (probs > 0.5).float()

                mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
                eye = torch.eye(preds.shape[1], device=self.config.device).unsqueeze(0)
                mask_2d = mask_2d & (eye == 0)

                all_pred_edges.extend(preds[mask_2d].cpu().numpy())
                all_pred_probs.extend(probs[mask_2d].cpu().numpy())
                all_true_edges.extend(batch['adjacency_target'][mask_2d].cpu().numpy())

                # --- Cycle Metrics ---
                # cycle_logits: [Batch, Seq, NumClasses]
                # cycle_target: [Batch, Seq]
                cycle_preds = torch.argmax(cycle_logits, dim=-1)
                cycle_probs = torch.softmax(cycle_logits, dim=-1)  # Full probs [B, S, C]
                mask_1d = batch['padding_mask'].view(-1)

                flat_preds = cycle_preds.view(-1)[mask_1d]
                flat_targets = batch['cycle_target'].view(-1)[mask_1d]

                # Flat Probs: [TotalTokens, NumClasses]
                flat_probs = cycle_probs.view(-1, cycle_probs.shape[-1])[mask_1d]

                all_cycle_preds.extend(flat_preds.cpu().numpy())
                all_cycle_targets.extend(flat_targets.cpu().numpy())
                all_cycle_probs.extend(flat_probs.cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("⚠️ OOM during Evaluation. Skipping batch.")
                continue

        if len(dataloader) > 0:
            avg_val_loss = total_val_loss / len(dataloader)
        else:
            avg_val_loss = 999.0

        # --- Compute Metrics ---
        # 1. Adjacency
        p = precision_score(all_true_edges, all_pred_edges, zero_division=0)
        r = recall_score(all_true_edges, all_pred_edges, zero_division=0)
        f1 = f1_score(all_true_edges, all_pred_edges, zero_division=0)

        try:
            pr_auc = average_precision_score(all_true_edges, all_pred_probs)
            roc_auc = roc_auc_score(all_true_edges, all_pred_probs)
        except Exception as e:
            logger.warning(f"{str(e)}, setting pr_auc=0.0")
            pr_auc = 0.0
            roc_auc = 0.5

        # 2. Cycle (Detailed Metrics)
        cycle_targets_np = np.array(all_cycle_targets)
        cycle_preds_np = np.array(all_cycle_preds)
        cycle_probs_np = np.array(all_cycle_probs)

        # A. Basic Accuracy & Binary Detection F1 (Recurring vs Noise)
        cycle_acc = accuracy_score(cycle_targets_np, cycle_preds_np)

        bin_cycle_pred = (cycle_preds_np > 0).astype(int)
        bin_cycle_true = (cycle_targets_np > 0).astype(int)
        cycle_f1 = f1_score(bin_cycle_true, bin_cycle_pred, zero_division=0)

        # B. Macro PR-AUC (Average over all classes)
        # We need to binarize the targets for One-vs-Rest calculation
        # This handles the multi-class nature of the problem
        try:
            # Check if we have enough classes present to compute AUC
            unique_classes = np.unique(cycle_targets_np)
            if len(unique_classes) > 1:
                # We use 'macro' to weight all classes equally (detecting rare weekly patterns is as important as monthly)
                # Or 'weighted' to account for imbalance. 'macro' is usually better for diagnostics.
                # However, sklearn requires one-hot targets for roc_auc_score/average_precision_score if multi-class
                # The easy way: Convert targets to one-hot manually or rely on 'ovr' logic if simple

                # Manual One-Hot for robustness
                n_classes = cycle_probs_np.shape[1]
                targets_one_hot = np.zeros((len(cycle_targets_np), n_classes))
                targets_one_hot[np.arange(len(cycle_targets_np)), cycle_targets_np] = 1

                cycle_pr_auc = average_precision_score(targets_one_hot, cycle_probs_np, average='macro')
            else:
                cycle_pr_auc = 0.0
        except Exception as e:
            logger.warning(f"Failed to compute Cycle PR-AUC: {str(e)}")
            cycle_pr_auc = 0.0

        # C. Detailed Breakdown (Precision/Recall/F1/Support per Class)
        # We output this as a dict to be logged nicely

        cycle_report_dict = classification_report(
            cycle_targets_np,
            cycle_preds_np,
            zero_division=0,
            output_dict=True
        )

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_val_loss)
        else:
            self.scheduler.step()

        return {
            "val_loss": avg_val_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            # Subtask metrics
            "cycle_acc": cycle_acc,
            "cycle_f1": cycle_f1,
            "cycle_pr_auc": cycle_pr_auc,
            "cycle_report": cycle_report_dict
        }