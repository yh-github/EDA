import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import torch.nn.functional as fnn
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
            self.optimizer, mode='min', factor=0.5, patience=2 # no `verbose` parameter
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
            self.contrastive_criterion = SupervisedContrastiveLoss(temperature=0.07)

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

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            try:
                # --- The Critical Recovery Block ---
                with torch.amp.autocast('cuda'):
                    adj_logits, cycle_logits, embeddings = self.model(batch)
                    loss, _, _, _ = self._compute_loss_with_pattern_ids(batch, adj_logits, cycle_logits, embeddings)
                    loss = loss / accumulation_steps

                # If we get here, forward pass was successful.
                # Now try backward (where gradients expand)
                self.scaler.scale(loss).backward()

            except torch.cuda.OutOfMemoryError:
                # --- RECOVERY LOGIC ---
                self.optimizer.zero_grad()  # Drop any partial gradients
                torch.cuda.empty_cache()  # Force free memory

                # Log diagnostic info to help you tune config later
                seq_len = batch['input_ids'].shape[1]
                logger.warning(
                    f"⚠️ OOM detected at Epoch {epoch_idx}, Batch {batch_idx}. "
                    f"Skipping batch with Max Seq Len: {seq_len}. "
                    f"Mem: {torch.cuda.memory_allocated() / 1e9:.1f}GB"
                )
                num_batches_skipped += 1
                continue  # Skip to next batch

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Handle older PyTorch versions generic RuntimeError
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    logger.warning(f"⚠️ OOM (RuntimeError) at Batch {batch_idx}. Skipping.")
                    num_batches_skipped += 1
                    continue
                raise e

            # Accumulation Step
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            num_batches_processed += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch_idx} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * accumulation_steps:.4f}")

        # Handle remaining gradients
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
        all_true_edges = []
        total_val_loss = 0.0

        # We also wrap eval in try/except just in case validation has a huge outlier
        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            try:
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

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("⚠️ OOM during Evaluation. Skipping batch.")
                continue

        avg_val_loss = total_val_loss / len(dataloader)

        p = precision_score(all_true_edges, all_pred_edges, zero_division=0)
        r = recall_score(all_true_edges, all_pred_edges, zero_division=0)
        f1 = f1_score(all_true_edges, all_pred_edges, zero_division=0)

        self.scheduler.step(avg_val_loss)

        return {"val_loss": avg_val_loss, "precision": p, "recall": r, "f1": f1}