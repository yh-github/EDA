import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import torch.nn.functional as F
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
    Operates per account (per sample in batch).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor):
        """
        features: [Batch, Seq, Dim]
        labels: [Batch, Seq] (pattern_id, -1 is noise)
        padding_mask: [Batch, Seq] (True if valid)
        """
        device = features.device
        batch_size, seq_len, _ = features.shape

        # Normalize features for cosine similarity
        features = F.normalize(features, dim=-1)

        total_loss = 0.0
        n_valid_batches = 0

        # We iterate batch items because contrast is only valid WITHIN an account history
        for i in range(batch_size):
            # Extract valid elements for this account
            valid_mask = padding_mask[i]
            feat = features[i][valid_mask]  # [N, Dim]
            lbl = labels[i][valid_mask]  # [N]

            if len(lbl) < 2:
                continue

            # Compute similarity matrix: [N, N]
            sim_matrix = torch.matmul(feat, feat.T) / self.temperature

            # For numerical stability
            sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_matrix_max.detach()

            # Mask for positives: same label AND not noise (-1)
            # lbl is [N], lbl.unsqueeze(0) is [1, N], lbl.unsqueeze(1) is [N, 1]
            label_mask = torch.eq(lbl.unsqueeze(1), lbl.unsqueeze(0)).float()

            # Mask out self-contrast (diagonal)
            logits_mask = torch.scatter(
                torch.ones_like(label_mask),
                1,
                torch.arange(len(lbl)).view(-1, 1).to(device),
                0
            )

            # Mask out noise (-1) from being positives
            # If label is -1, it should effectively have NO positives other than itself (which is masked)
            # So rows where label == -1 should effectively be ignored or treated as having no positives
            not_noise = (lbl != -1).float().unsqueeze(1)
            label_mask = label_mask * logits_mask * not_noise

            # If an anchor has no positives, we can't compute SupCon for it
            # We compute loss only for anchors that have at least one OTHER positive
            has_positives = label_mask.sum(1) > 0
            if has_positives.sum() == 0:
                continue

            # Compute log_prob
            exp_sim = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)

            # Mean log_prob over positive pairs
            mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-6)

            # Loss for this account
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
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        # Loss Functions
        if pos_weight:
            weight_tensor = torch.tensor([pos_weight]).to(config.device)
            self.adj_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight_tensor)
        else:
            self.adj_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.cycle_criterion = nn.CrossEntropyLoss(reduction='none')

        if config.use_contrastive_loss:
            self.contrastive_criterion = SupervisedContrastiveLoss(temperature=0.07)

    def _compute_loss(self, batch, adj_logits, cycle_logits, embeddings):
        # 1. Adjacency Loss
        mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
        eye = torch.eye(adj_logits.shape[1], device=self.config.device).unsqueeze(0)
        mask_2d = mask_2d & (eye == 0)

        adj_loss = self.adj_criterion(adj_logits, batch['adjacency_target'])
        adj_loss = (adj_loss * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)

        # 2. Cycle Loss
        cycle_loss = self.cycle_criterion(
            cycle_logits.view(-1, self.config.num_classes),
            batch['cycle_target'].view(-1)
        )
        mask_1d = batch['padding_mask'].view(-1)
        cycle_loss = (cycle_loss * mask_1d.float()).sum() / mask_1d.sum().clamp(min=1)

        total_loss = adj_loss + cycle_loss

        con_loss = torch.tensor(0.0, device=total_loss.device)
        if self.config.use_contrastive_loss:
            # embeddings is 'h' from transformer
            # batch['pattern_ids'] is the label (where -1 is noise)
            # We need to recover the pattern_ids from the sparse/adjacency matrix?
            # No, 'pattern_ids' are not passed to model forward, but they ARE in the batch dict from collate_fn!
            # Wait, `collate_fn` puts `pattern_ids` into the dict but not as a tensor?
            # Let's check data.py -> NO, it does NOT pass pattern_ids tensor in `result`.
            # We need to fix collate_fn if we want to use pattern_ids for SupCon.
            # Workaround: Reconstruction from adjacency is hard.
            # BETTER: We need to rely on `cycle_target`? No, that's cycle class.
            # FIX: We rely on the fact that `collate_fn` computes adjacency from `pattern_ids`.
            # Since I can't edit data.py in this specific block without overwriting `analyze_token_stats`,
            # I will assume `batch` has `adjacency_target` and I can infer clusters? No.
            # I MUST assume the user will update data.py or I have to hack it here.
            # However, looking at `src/multi/data.py` provided in previous turn:
            # `p_ids = item['pattern_ids']` is used to build adjacency, but NOT added to `result` dict.
            # THIS IS A PROBLEM for SupCon.
            # Ideally, I'd edit data.py. But `data.py` was edited in the previous turn.
            # I will edit `trainer.py` to gracefully skip SupCon if `pattern_ids` missing,
            # BUT since I cannot easily edit `data.py` again without reposting the whole thing,
            # I will add logic here to *infer* approximate labels from the adjacency matrix if needed,
            # OR I will simply re-emit `data.py` with the fix. Re-emitting `data.py` is safer.
            pass

        return total_loss, adj_loss, cycle_loss, con_loss

    def _compute_loss_with_pattern_ids(self, batch, adj_logits, cycle_logits, embeddings):
        # 1. Adjacency
        mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
        eye = torch.eye(adj_logits.shape[1], device=self.config.device).unsqueeze(0)
        mask_2d = mask_2d & (eye == 0)
        adj_loss = self.adj_criterion(adj_logits, batch['adjacency_target'])
        adj_loss = (adj_loss * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)

        # 2. Cycle
        cycle_loss = self.cycle_criterion(
            cycle_logits.view(-1, self.config.num_classes),
            batch['cycle_target'].view(-1)
        )
        mask_1d = batch['padding_mask'].view(-1)
        cycle_loss = (cycle_loss * mask_1d.float()).sum() / mask_1d.sum().clamp(min=1)

        # 3. Contrastive
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

        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                # Model now returns embeddings 'h' as 3rd output
                adj_logits, cycle_logits, embeddings = self.model(batch)
                loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)

                # Normalize loss for accumulation
                loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # Scale back for logging

            if batch_idx % max(10, num_batches // 10) == 0:
                logger.info(
                    f"Epoch {epoch_idx} | Batch {batch_idx}/{num_batches} | Loss: {loss.item() * accumulation_steps:.4f}")

        # Handle remaining gradients
        if len(dataloader) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_pred_edges = []
        all_true_edges = []
        total_val_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                adj_logits, cycle_logits, embeddings = self.model(batch)
                loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)

            total_val_loss += loss.item()

            preds = (torch.sigmoid(adj_logits) > 0.5).float()
            mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
            eye = torch.eye(preds.shape[1], device=self.config.device).unsqueeze(0)
            mask_2d = mask_2d & (eye == 0)

            all_pred_edges.extend(preds[mask_2d].cpu().numpy())
            all_true_edges.extend(batch['adjacency_target'][mask_2d].cpu().numpy())

        avg_val_loss = total_val_loss / len(dataloader)

        p = precision_score(all_true_edges, all_pred_edges, zero_division=0)
        r = recall_score(all_true_edges, all_pred_edges, zero_division=0)
        f1 = f1_score(all_true_edges, all_pred_edges, zero_division=0)

        self.scheduler.step(avg_val_loss)

        return {"val_loss": avg_val_loss, "precision": p, "recall": r, "f1": f1}