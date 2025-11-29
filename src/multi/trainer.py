import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
import logging
import torch.nn.functional as fnn
import os
from multi.config import MultiExpConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Pulls embeddings with the same patternId together, pushes others apart.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor):
        device = features.device
        batch_size, seq_len, _ = features.shape

        features = fnn.normalize(features, dim=-1)

        total_loss = 0.0
        n_valid_batches = 0

        for i in range(batch_size):
            valid_mask = padding_mask[i]
            feat = features[i][valid_mask]
            lbl = labels[i][valid_mask]

            if len(lbl) < 2:
                continue

            sim_matrix = torch.matmul(feat, feat.T) / self.temperature
            sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_matrix_max.detach()

            label_mask = torch.eq(lbl.unsqueeze(1), lbl.unsqueeze(0)).float()

            logits_mask = torch.scatter(
                torch.ones_like(label_mask),
                1,
                torch.arange(len(lbl)).view(-1, 1).to(device),
                0
            )

            not_noise = (lbl != -1).float().unsqueeze(1)
            label_mask = label_mask * logits_mask * not_noise

            has_positives = label_mask.sum(1) > 0
            if has_positives.sum() == 0:
                continue

            exp_sim = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos[has_positives].mean()
            total_loss += loss
            n_valid_batches += 1

        if n_valid_batches > 0:
            return total_loss / n_valid_batches
        return torch.tensor(0.0, device=device, requires_grad=True)


class MultiTrainer:
    def __init__(self, model, config: MultiExpConfig, pos_weight: float = None):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=config.scheduler_patience
        )

        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        if pos_weight:
            weight_tensor = torch.tensor([pos_weight]).to(config.device)
            self.adj_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight_tensor)
        else:
            self.adj_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.cycle_criterion = nn.CrossEntropyLoss(reduction='none')

        if config.use_contrastive_loss:
            self.contrastive_criterion = SupervisedContrastiveLoss(temperature=config.contrastive_temperature)

        # Signal handling injection point
        self.stop_requested = False

    def request_stop(self):
        """Allows external agents (like signal handlers) to request a graceful stop."""
        logger.info("Trainer received stop request. Finishing current epoch/batch...")
        self.stop_requested = True

    def fit(self, train_loader, val_loader, epochs: int, trial=None, save_path: str = None, stop_callback=None):
        """
        Unified training loop with Early Stopping, Optuna reporting, and Model Saving.
        Tracks PR-AUC as the primary stability metric if F1 is volatile.
        """
        # Lazy import to avoid hard dependency if not tuning
        import optuna

        best_metric = -1.0
        patience = self.config.early_stopping_patience
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs. Early Stopping Patience: {patience}")

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

            val_f1 = metrics['f1']
            val_pr_auc = metrics['pr_auc']
            val_loss = metrics['val_loss']

            # Log extensive metrics
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"F1: {val_f1:.4f} | PR-AUC: {val_pr_auc:.4f} | "
                f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}"
            )

            # 3. Optuna Reporting (We still report F1 to Optuna as it's the "official" goal usually,
            # but you can switch this to val_pr_auc if F1 is too noisy)
            if trial:
                trial.report(val_f1, epoch - 1)
                if trial.should_prune():
                    logger.info("Trial pruned by Optuna.")
                    raise optuna.TrialPruned()

            # 4. Early Stopping & Saving
            target_metric = val_f1

            if target_metric > best_metric:
                best_metric = target_metric
                patience_counter = 0

                if save_path:
                    # Ensure dir exists
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    checkpoint = {
                        "config": self.config,
                        "state_dict": self.model.state_dict(),
                        "best_metric": best_metric,
                        "metrics": metrics,
                        "epoch": epoch
                    }
                    torch.save(checkpoint, save_path)
                    logger.info(f"  --> New Best Model Saved (Score: {best_metric:.4f})")
                else:
                    logger.info(f"  --> New Best Score: {best_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  ... No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info(f"⛔ Early stopping triggered at epoch {epoch}")
                    break

        return best_metric

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
                with torch.amp.autocast('cuda'):
                    adj_logits, cycle_logits, embeddings = self.model(batch)
                    loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)
                    loss = loss / accumulation_steps

                self.scaler.scale(loss).backward()
                batches_since_step += 1

            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                num_batches_skipped += 1
                continue
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    num_batches_skipped += 1
                    continue
                raise e

            if batches_since_step >= accumulation_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                batches_since_step = 0

            total_loss += loss.item() * accumulation_steps
            num_batches_processed += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch_idx} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * accumulation_steps:.4f}")

        if batches_since_step > 0 and not self.stop_requested:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        if num_batches_skipped > 0:
            logger.info(f"⚠️ Warning: Skipped {num_batches_skipped} batches due to OOM this epoch.")

        return total_loss / max(1, num_batches_processed)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_pred_edges = []
        all_pred_probs = []  # Store raw probs for AUC
        all_true_edges = []
        total_val_loss = 0.0

        for batch in dataloader:
            if self.stop_requested: break

            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            try:
                with torch.amp.autocast('cuda'):
                    adj_logits, cycle_logits, embeddings = self.model(batch)
                    loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)

                total_val_loss += loss.item()

                probs = torch.sigmoid(adj_logits)
                preds = (probs > 0.5).float()

                mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
                eye = torch.eye(preds.shape[1], device=self.config.device).unsqueeze(0)
                mask_2d = mask_2d & (eye == 0)

                # Flatten based on mask
                all_pred_edges.extend(preds[mask_2d].cpu().numpy())
                all_pred_probs.extend(probs[mask_2d].cpu().numpy())
                all_true_edges.extend(batch['adjacency_target'][mask_2d].cpu().numpy())

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        if len(dataloader) > 0:
            avg_val_loss = total_val_loss / len(dataloader)
        else:
            avg_val_loss = 999.0

        # Calculate metrics
        p = precision_score(all_true_edges, all_pred_edges, zero_division=0)
        r = recall_score(all_true_edges, all_pred_edges, zero_division=0)
        f1 = f1_score(all_true_edges, all_pred_edges, zero_division=0)

        try:
            pr_auc = average_precision_score(all_true_edges, all_pred_probs)
            roc_auc = roc_auc_score(all_true_edges, all_pred_probs)
        except Exception:
            pr_auc = 0.0
            roc_auc = 0.5

        self.scheduler.step(avg_val_loss)

        return {
            "val_loss": avg_val_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc
        }