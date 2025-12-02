import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, average_precision_score, \
    roc_auc_score, accuracy_score
import logging
import torch.nn.functional as fnn
import os
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
from multi.config import MultiExpConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
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
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_negative_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor):
        device = features.device
        batch_size, seq_len, _ = features.shape
        features = fnn.normalize(features, dim=-1)
        sim_matrix = torch.bmm(features, features.transpose(1, 2)) / self.temperature

        mask_2d = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        eye = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        valid_mask = mask_2d & (eye == 0)

        sim_matrix_max, _ = torch.max(sim_matrix, dim=2, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()

        label_mask = torch.eq(labels.unsqueeze(2), labels.unsqueeze(1)).float()
        not_noise_mask = (labels != -1).float().unsqueeze(2)
        final_mask = label_mask * valid_mask.float() * not_noise_mask

        exp_sim = torch.exp(sim_matrix) * valid_mask.float()
        if self.hard_negative_weight != 1.0:
            negative_mask = valid_mask & (label_mask == 0)
            weight_matrix = torch.ones_like(sim_matrix)
            weight_matrix[negative_mask.bool()] = self.hard_negative_weight
            exp_sim = exp_sim * weight_matrix

        log_prob = sim_matrix - torch.log(exp_sim.sum(2, keepdim=True) + 1e-6)
        sum_log_prob_pos = (final_mask * log_prob).sum(2)
        num_pos = final_mask.sum(2)
        mean_log_prob_pos = sum_log_prob_pos / (num_pos + 1e-6)
        has_positives = num_pos > 0
        if has_positives.sum() > 0:
            return -mean_log_prob_pos[has_positives].mean()
        return torch.tensor(0.0, device=device, requires_grad=True)


class BaseTrainer:
    """Shared logic for both trainers."""

    def __init__(self, model, config: MultiExpConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        def get_scheduler() -> LRScheduler:
            if config.scheduler_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config.scheduler_t0,
                                                                      T_mult=config.scheduler_t_mult)
            else:
                return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                            patience=config.scheduler_patience)

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

        if config.use_contrastive_loss:
            self.contrastive_criterion = SupervisedContrastiveLoss(
                temperature=config.contrastive_temperature, hard_negative_weight=config.hard_negative_weight
            )
        self.stop_requested = False

    def fit(self, train_loader, val_loader, epochs: int, trial=None, save_path: str = None, stop_callback=None,
            metric_to_track='pr_auc'):
        import optuna
        best_score = -1.0
        patience = self.config.early_stopping_patience
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs. Track: {metric_to_track}")
        logger.info(str(self.config))

        for epoch in range(1, epochs + 1):
            if stop_callback and stop_callback(): self.stop_requested = True
            if self.stop_requested: break

            train_loss = self.train_epoch(train_loader, epoch)
            metrics = self.evaluate(val_loader)
            current_score = metrics.get(metric_to_track, 0.0)

            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {train_loss:.3f}/{metrics['val_loss']:.3f} | {metric_to_track}={current_score:.4f}")
            self.log_metrics(metrics)

            if trial:
                trial.report(current_score, epoch)
                if trial.should_prune(): raise optuna.TrialPruned()

            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({"config": self.config, "state_dict": self.model.state_dict(), "best_score": best_score},
                               save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping.")
                    break
        return best_score

    def log_metrics(self, metrics):
        pass  # Override

    def train_epoch(self, dataloader, epoch_idx):
        raise NotImplementedError

    def evaluate(self, dataloader):
        raise NotImplementedError


class MultiTrainer(BaseTrainer):
    """Multiclass Trainer"""

    def __init__(self, model, config: MultiExpConfig):
        super().__init__(model, config)
        self.cycle_criterion = nn.CrossEntropyLoss(reduction='none')

    def log_metrics(self, metrics):
        # Detailed logging specific to multiclass
        pass

    def _compute_loss(self, batch, adj_logits, cycle_logits, embeddings):
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
        return total_loss

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        count = 0

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                adj_logits, cycle_logits, embeddings = self.model(batch)
                loss = self._compute_loss(batch, adj_logits, cycle_logits, embeddings)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            count += 1
        return total_loss / max(1, count)

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

class BinaryMultiTrainer(BaseTrainer):
    """
    New Binary Trainer.
    Cycle prediction is replaced by 'isRecurring' (Binary) prediction.
    """

    def __init__(self, model, config: MultiExpConfig):
        super().__init__(model, config)
        # For binary classification (Recurring vs Noise)
        self.binary_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def _compute_loss(self, batch, adj_logits, binary_logits, embeddings):
        # 1. Adjacency Loss
        mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
        eye = torch.eye(adj_logits.shape[1], device=self.config.device).unsqueeze(0)
        mask_2d = mask_2d & (eye == 0)

        adj_loss = self.adj_criterion(adj_logits, batch['adjacency_target'])
        adj_loss = (adj_loss * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)

        # 2. Binary Classification Loss
        # Target: cycle_target > 0 (Assuming 0 is Noise/None)
        # cycle_target is [B, S] long ints. We convert to float for BCE.
        is_recurring_target = (batch['cycle_target'] > 0).float().unsqueeze(-1)  # [B, S, 1]

        bin_loss = self.binary_criterion(binary_logits, is_recurring_target)
        # Mask out padding
        mask_1d = batch['padding_mask'].unsqueeze(-1)  # [B, S, 1]
        bin_loss = (bin_loss * mask_1d.float()).sum() / mask_1d.sum().clamp(min=1)

        # 3. Contrastive Loss
        con_loss = torch.tensor(0.0, device=adj_loss.device)
        if self.config.use_contrastive_loss and 'pattern_ids' in batch:
            con_loss = self.contrastive_criterion(embeddings, batch['pattern_ids'], batch['padding_mask'])

        total_loss = adj_loss + bin_loss + (self.config.contrastive_loss_weight * con_loss)
        return total_loss

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        count = 0

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                # Forward returns: adj_logits, binary_logits, embeddings
                adj_logits, binary_logits, embeddings = self.model(batch)
                loss = self._compute_loss(batch, adj_logits, binary_logits, embeddings)

            if torch.isnan(loss):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

        return total_loss / max(1, count)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0

        # Collections
        all_true_edges, all_pred_edges, all_prob_edges = [], [], []
        all_true_bin, all_pred_bin, all_prob_bin = [], [], []

        # New Collection: Clustering-as-Classifier
        # We need to collect MAX clustering probability per node
        all_clust_prob_max = []

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                adj_logits, binary_logits, embeddings = self.model(batch)
                loss = self._compute_loss(batch, adj_logits, binary_logits, embeddings)
            total_loss += loss.item()

            # --- Adjacency Metrics ---
            adj_probs = torch.sigmoid(adj_logits)

            # 1. Edge Level (Standard Clustering Metrics)
            mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
            eye = torch.eye(adj_probs.shape[1], device=self.config.device).unsqueeze(0)
            valid_edges = mask_2d & (eye == 0)

            all_prob_edges.extend(adj_probs[valid_edges].cpu().numpy())
            all_true_edges.extend(batch['adjacency_target'][valid_edges].cpu().numpy())

            # 2. Node Level (Clustering as Detection)
            # Find the max probability of connection to ANY other node (excluding self)
            # Mask diagonal first
            # adj_probs shape: [B, S, S]
            adj_probs_masked = adj_probs.clone()
            adj_probs_masked.masked_fill_(eye.bool(), 0.0)

            # Max over columns (dim 2) gives [B, S] - best neighbor score for each node
            max_neighbor_probs, _ = adj_probs_masked.max(dim=2)

            # Mask out padding nodes
            mask_1d = batch['padding_mask']
            all_clust_prob_max.extend(max_neighbor_probs[mask_1d].cpu().numpy())

            # --- Binary Metrics ---
            bin_probs = torch.sigmoid(binary_logits).squeeze(-1)  # [B, S]
            bin_targets = (batch['cycle_target'] > 0).float()

            all_prob_bin.extend(bin_probs[mask_1d].cpu().numpy())
            all_true_bin.extend(bin_targets[mask_1d].cpu().numpy())

        # Metrics Calc
        metrics = {"val_loss": total_loss / max(1, len(dataloader))}

        # A. Adjacency (Graph Quality)
        if all_true_edges:
            p_edges = (np.array(all_prob_edges) > 0.5).astype(int)
            metrics['pr_auc'] = average_precision_score(all_true_edges, all_prob_edges)
            metrics['adj_f1'] = f1_score(all_true_edges, p_edges)
            metrics['adj_p'] = precision_score(all_true_edges, p_edges)
            metrics['adj_r'] = recall_score(all_true_edges, p_edges)

        # B. Binary (Explicit Detection Head)
        if all_true_bin:
            p_bin = (np.array(all_prob_bin) > 0.5).astype(int)
            metrics['cycle_f1'] = f1_score(all_true_bin, p_bin)
            metrics['cycle_pr_auc'] = average_precision_score(all_true_bin, all_prob_bin)
            metrics['cycle_rec'] = recall_score(all_true_bin, p_bin)
            metrics['cycle_prec'] = precision_score(all_true_bin, p_bin)

            # C. Clustering-as-Detection (Implicit Detection)
            # How well does "having a neighbor" predict recurrence?
            p_clust = (np.array(all_clust_prob_max) > 0.5).astype(int)
            metrics['clust_f1'] = f1_score(all_true_bin, p_clust)
            metrics['clust_p'] = precision_score(all_true_bin, p_clust)
            metrics['clust_r'] = recall_score(all_true_bin, p_clust)
            metrics['clust_pr_auc'] = average_precision_score(all_true_bin, all_clust_prob_max)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics['val_loss'])
        else:
            self.scheduler.step()

        return metrics

    def log_metrics(self, metrics):
        logger.info(
            f" ADJ_PR_AUC: {metrics.get('pr_auc', 0):.3f}"
            f" ADJ_F1: {metrics.get('adj_f1', 0):.3f}"
            f" ADJ_P: {metrics.get('adj_p', 0):.3f}"
            f" ADJ_R: {metrics.get('adj_r', 0):.3f}"
            f" |"
            f" BIN_F1: {metrics.get('cycle_f1', 0):.3f}"
            f" BIN_PR_AUC: {metrics.get('cycle_pr_auc', 0):.3f}"
            f" BIN_P: {metrics.get('cycle_prec', 0):.3f}"
            f" BIN_R: {metrics.get('cycle_rec', 0):.3f}"
        )
        logger.info(
            f"CLUST:"
            f" PR_AUC={metrics.get('clust_pr_auc', 0):.3f}"
            f" F1={metrics.get('clust_f1', 0):.3f}"
            f" P={metrics.get('clust_p', 0):.3f}"
            f" R={metrics.get('clust_r', 0):.3f}"
        )
