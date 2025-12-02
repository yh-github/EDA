import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from multi.config import MultiExpConfig
from multi.trainer import BaseTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class BinaryMultiTrainer(BaseTrainer):
    """
    New Binary Trainer.
    Cycle prediction is replaced by 'isRecurring' (Binary) prediction.
    """

    def __init__(self, model, config: MultiExpConfig):
        super().__init__(model, config)

        # --- FIX: Apply Class Balancing to Binary Head ---
        if config.use_focal_loss:
            # Re-use the Focal Loss from BaseTrainer (configured with alpha/gamma)
            self.binary_criterion = self.adj_criterion
        else:
            # Re-use pos_weight from config
            weight_tensor = torch.tensor([config.pos_weight]).to(config.device)
            self.binary_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight_tensor)

    def _compute_loss(self, batch, adj_logits, binary_logits, embeddings):
        device = self.config.device

        # --- SPEEDUP: Generate Adjacency Target on GPU ---
        # Instead of relying on CPU collate, we generate it here from pattern_ids
        # pattern_ids: [B, S]. -1 indicates noise/padding.
        p_ids = batch['pattern_ids']

        # Expand to [B, S, 1] and [B, 1, S] for broadcasting
        p_v = p_ids.unsqueeze(2)
        p_h = p_ids.unsqueeze(1)

        # Match where IDs are equal and NOT -1
        # [B, S, S] boolean mask
        adj_target_gpu = (p_v == p_h) & (p_v != -1)
        adj_target_float = adj_target_gpu.float()

        # 1. Adjacency Loss
        mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
        eye = torch.eye(adj_logits.shape[1], device=device).unsqueeze(0)

        # Mask out padding AND self-loops (diagonal) for loss calculation
        mask_2d = mask_2d & (eye == 0)

        adj_loss = self.adj_criterion(adj_logits, adj_target_float)
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

            # Re-generate target on GPU for metrics
            p_ids = batch['pattern_ids']
            p_v = p_ids.unsqueeze(2)
            p_h = p_ids.unsqueeze(1)
            adj_target_gpu = (p_v == p_h) & (p_v != -1)

            # 1. Edge Level (Standard Clustering Metrics)
            mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
            eye = torch.eye(adj_probs.shape[1], device=self.config.device).unsqueeze(0)
            valid_edges = mask_2d & (eye == 0)

            all_prob_edges.extend(adj_probs[valid_edges].cpu().numpy())
            all_true_edges.extend(adj_target_gpu[valid_edges].cpu().numpy())

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