import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from transformers import AutoTokenizer
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.encoder import TransactionTransformer
from multi.data import get_dataloader  # Use shared data pipeline

logger = logging.getLogger(__name__)


class MultiPredictor:
    def __init__(self, model_path: str, runtime_config: MultiExpConfig = None):
        if runtime_config is None:
            runtime_config = MultiExpConfig()

        self.device = runtime_config.device
        self.field_config = MultiFieldConfig()

        logger.info(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            saved_config = checkpoint["config"]
            state_dict = checkpoint["state_dict"]
            # Override batch size with runtime request
            saved_config.batch_size = runtime_config.batch_size
            self.config = saved_config
        else:
            raise Exception('checkpoint not a dict')

        logger.info(f"Loading Tokenizer: {self.config.text_encoder_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_model)

        self.model = TransactionTransformer(self.config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.idx_to_cycle = {v: k for k, v in self.config.cycle_map.items()}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        fc = self.field_config

        # 1. Prepare DataFrame (Validation)
        req_cols = [fc.accountId, fc.date, fc.amount, fc.text]
        for col in req_cols:
            if col not in df.columns:
                raise KeyError(f"Input DataFrame missing required column: {col}")

        # Ensure dummy columns exist for Dataset compatibility
        if fc.patternCycle not in df.columns:
            df[fc.patternCycle] = 'None'
        if fc.patternId not in df.columns:
            df[fc.patternId] = -1

        # 2. Create DataLoader (Fast, Pre-tokenized)
        logger.info("Initializing DataLoader for inference...")
        loader = get_dataloader(df, self.config, shuffle=False)

        # 3. Prepare Result Columns
        # We will fill these arrays using the original indices
        n_rows = len(df)
        res_is_rec = np.zeros(n_rows, dtype=bool)
        res_probs = np.zeros(n_rows, dtype=np.float32)
        res_pids = np.array([None] * n_rows, dtype=object)
        res_cycles = np.array(["None"] * n_rows, dtype=object)

        logger.info(f"Running inference on {len(loader)} batches...")

        with torch.no_grad():
            for batch in loader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward Pass
                adj_logits, cycle_logits, _ = self.model(batch)

                # Extract Results
                self._process_batch(
                    batch, adj_logits, cycle_logits,
                    res_is_rec, res_probs, res_pids, res_cycles
                )

        # 4. Assign results back to dataframe
        # Note: 'get_dataloader' creates a copy of df internally and resets index.
        # But we captured 'original_index' from that internal DF.
        # If the input 'df' index was 0..N, it matches.
        # If input 'df' had custom index, we need to be careful.
        # Ideally, we return a new dataframe aligned with the input.

        # Since MultiTransactionDataset filters rows (amount=0), the loader
        # covers a SUBSET of the original DF.
        # We need to map results back to the specific rows processed.

        # We clone the input to avoid side effects
        out_df = df.copy()

        # We need to align the results arrays with the original DF.
        # The 'res_...' arrays are sized to the INPUT df (n_rows).
        # We filled them at the indices provided by the loader.

        out_df['pred_isRecurring'] = res_is_rec
        out_df['pred_recurring_prob'] = res_probs
        out_df['pred_patternId'] = res_pids
        out_df['pred_patternCycle'] = res_cycles

        return out_df

    def _process_batch(self, batch, adj_logits, cycle_logits, res_is_rec, res_probs, res_pids, res_cycles):
        probs = torch.sigmoid(adj_logits).cpu().numpy()
        cycle_softmax = torch.softmax(cycle_logits, dim=-1).cpu().numpy()

        # Indices in the original dataframe
        batch_indices = batch['original_index'].cpu().numpy()  # [B, Seq]
        padding_mask = batch['padding_mask'].cpu().numpy()  # [B, Seq]

        # Account IDs are not directly in batch, but we can reconstruct Pid string
        # using batch index for uniqueness within this run

        batch_size = probs.shape[0]

        for i in range(batch_size):
            # Extract valid sequence
            valid_len = padding_mask[i].sum()
            if valid_len < 2: continue

            # Slice valid data
            indices = batch_indices[i, :valid_len]
            adj = probs[i, :valid_len, :valid_len]
            node_cycles = cycle_softmax[i, :valid_len, :]

            # 1. Detection Scores
            rec_scores = 1.0 - node_cycles[:, 0]

            # Assign scores to global arrays
            # (Use flat indexing for numpy arrays)
            res_probs[indices] = rec_scores

            # 2. Clustering
            adj_binary = (adj > 0.5).astype(int)
            np.fill_diagonal(adj_binary, 0)
            graph = csr_matrix(adj_binary)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            for cluster_id in range(n_components):
                cluster_mask = (labels == cluster_id)
                cluster_indices = indices[cluster_mask]

                # Filter small noise
                if len(cluster_indices) < 2:
                    continue

                # 3. Cycle Determination
                # Sum probs for this cluster
                cluster_sum_probs = node_cycles[cluster_mask].sum(axis=0)
                best_cycle_idx = cluster_sum_probs.argmax()

                is_recurring = (best_cycle_idx != 0)

                if is_recurring:
                    # Construct unique ID: "idx_{first_row_index}_{cluster}"
                    # This ensures uniqueness across the whole dataframe
                    pid_str = f"grp_{cluster_indices[0]}_{cluster_id}"
                    cycle_str = self.idx_to_cycle.get(best_cycle_idx, "None")

                    # Update Result Arrays
                    res_is_rec[cluster_indices] = True
                    res_pids[cluster_indices] = pid_str
                    res_cycles[cluster_indices] = cycle_str