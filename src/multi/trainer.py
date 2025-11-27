import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from multi.config import MultiExpConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultiTrainer:
    def __init__(self, model, config: MultiExpConfig, pos_weight: float = None):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        # --- Mixed Precision Scaler ---
        # Updated to new API if available, or fallback
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

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            # --- Mixed Precision Context ---
            # Fix: Use torch.amp.autocast for newer PyTorch versions
            with torch.amp.autocast('cuda'):
                adj_logits, cycle_logits = self.model(batch)

                # --- LOSS CALCULATION ---
                # 1. Adjacency Loss
                mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)

                # Exclude self-loops (diagonal) from loss
                # This prevents the model from learning the trivial "I am me" connection
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

                loss = adj_loss + cycle_loss

            # --- Mixed Precision Backward ---
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % max(10, num_batches // 10) == 0:
                logger.info(f"Epoch {epoch_idx} | Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_pred_edges = []
        all_true_edges = []

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                adj_logits, _ = self.model(batch)

            preds = (torch.sigmoid(adj_logits) > 0.5).float()

            mask_2d = batch['padding_mask'].unsqueeze(1) & batch['padding_mask'].unsqueeze(2)
            eye = torch.eye(preds.shape[1], device=self.config.device).unsqueeze(0)
            mask_2d = mask_2d & (eye == 0)

            all_pred_edges.extend(preds[mask_2d].cpu().numpy())
            all_true_edges.extend(batch['adjacency_target'][mask_2d].cpu().numpy())

        p = precision_score(all_true_edges, all_pred_edges, zero_division=0)
        r = recall_score(all_true_edges, all_pred_edges, zero_division=0)
        f1 = f1_score(all_true_edges, all_pred_edges, zero_division=0)

        return {"precision": p, "recall": r, "f1": f1}