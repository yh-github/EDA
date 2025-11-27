import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from multi.config import MultiExpConfig
from multi.encoder import TransactionTransformer
from multi.data import collate_fn


class MultiPredictor:
    def __init__(self, model_path: str, config: MultiExpConfig, embedding_service):
        self.config = config
        self.device = config.device
        self.embedding_service = embedding_service

        # Load Model
        print(f"Loading model from {model_path}...")
        self.model = TransactionTransformer(config)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Reverse Cycle Map for decoding
        self.idx_to_cycle = {v: k for k, v in config.cycle_map.items()}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a raw DataFrame, runs inference, and returns the DataFrame
        enriched with 'pred_isRecurring', 'pred_patternId', 'pred_patternCycle'.
        """
        # Prepare Data Groups (Account + Direction)
        # Note: We must follow the same grouping logic as training
        df['direction'] = np.sign(df['amount'])
        # Filter out zero amounts or handle them
        df = df[df['direction'] != 0].copy()

        # Assign unique group IDs to keep track
        groups = [group for _, group in df.groupby(['accountId', 'direction'])]

        results = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(groups), batch_size):
            batch_groups = groups[i: i + batch_size]

            # Prepare batch input
            batch_data = self._prepare_batch_data(batch_groups)

            # Collate (Padding + Embedding)
            batch = collate_fn(batch_data, self.embedding_service, self.config)

            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Run Inference
            with torch.no_grad():
                adj_logits, cycle_logits = self.model(batch)

            # Post-Process Batch
            self._process_batch_results(batch_groups, adj_logits, cycle_logits, results)

        # Merge results back to original DF
        result_df = pd.concat(results).sort_index()
        return result_df

    def _prepare_batch_data(self, groups):
        """Convert DataFrame groups into the dict format expected by collate_fn"""
        batch_list = []
        for group in groups:
            # Sort by date usually helps visualization, though model is permutation invariant
            # (except for Positional Encoding 'days')
            group = group.sort_values('date')

            texts = (group['bankRawDescription'].fillna('') + " " + group['counter_party'].fillna('')).tolist()
            amounts = group['amount'].values.astype(np.float32)
            log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

            dates = pd.to_datetime(group['date'])
            days = (dates - dates.min()).dt.days.values.astype(np.float32)

            # Dummy targets for inference
            pattern_ids = np.zeros(len(group))
            cycles = np.zeros(len(group))

            batch_list.append({
                "texts": texts,
                "amounts": log_amounts,
                "days": days,
                "pattern_ids": pattern_ids,
                "cycles": cycles,
                "original_index": group.index.tolist()  # Keep track of original rows
            })
        return batch_list

    def _process_batch_results(self, groups, adj_logits, cycle_logits, results_list):
        """
        Decodes the Adjacency Matrix and Cycle Logits into Pattern IDs.
        """
        # adj_logits: (Batch, N, N)
        # cycle_logits: (Batch, N, NumClasses)

        probs = torch.sigmoid(adj_logits).cpu().numpy()
        cycle_preds = torch.softmax(cycle_logits, dim=-1).cpu().numpy()

        for b_idx, group in enumerate(groups):
            n = len(group)
            if n == 0: continue

            # 1. Get Sub-Matrix for this account
            # (Ignore padding)
            adj = probs[b_idx, :n, :n]
            node_cycles = cycle_preds[b_idx, :n, :]

            # 2. Thresholding
            # A link exists if prob > 0.5
            adj_binary = (adj > 0.5).astype(int)

            # Remove self-loops (diagonal) just in case
            np.fill_diagonal(adj_binary, 0)

            # 3. Connected Components (Clustering)
            # This finds the "islands" in the graph
            graph = csr_matrix(adj_binary)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            # 4. Cycle Voting per Cluster
            final_pattern_ids = []
            final_cycles = []
            final_is_rec = []

            for cluster_id in range(n_components):
                # Get indices of transactions in this cluster
                indices = np.where(labels == cluster_id)[0]

                # Vote on Cycle: Sum probabilities of all members
                cluster_cycle_probs = node_cycles[indices].sum(axis=0)
                best_cycle_idx = cluster_cycle_probs.argmax()

                # Rule: If best cycle is 0 ('None'), then it's not recurring
                # Rule: If cluster size < 2, usually not recurring (unless 'once', but that's rare in training)
                is_recurring = (best_cycle_idx != 0) and (len(indices) > 1)

                cycle_label = self.idx_to_cycle[best_cycle_idx]

                # Generate a unique Pattern ID for this inference run
                # We use a temp string format: "{AccountID}_{Direction}_{ClusterID}"
                pat_id_str = f"{group['accountId'].iloc[0]}_{group['direction'].iloc[0]}_{cluster_id}" if is_recurring else None

                # Assign to all members
                for _ in indices:
                    final_is_rec.append(is_recurring)
                    final_pattern_ids.append(pat_id_str)
                    final_cycles.append(cycle_label if is_recurring else None)

            # Re-order based on clustering output logic is tricky because 'labels' corresponds to 0..N
            # We map back directly using indices

            # Create a mini dataframe to merge inputs
            res_df = group.copy()
            res_df['pred_isRecurring'] = False
            res_df['pred_patternId'] = None
            res_df['pred_patternCycle'] = None

            # Need to map the 'labels' array back to the dataframe rows
            # 'labels' is length N, corresponding to group.iloc[0]...group.iloc[N]

            # We iterate 0..N to assign values
            # Efficient way:
            pred_cycles_list = []
            pred_pids_list = []
            pred_is_rec_list = []

            for i in range(n):
                cluster_id = labels[i]

                # Re-calculate vote for this specific cluster (could optimize this out but this is safe)
                indices = np.where(labels == cluster_id)[0]
                cluster_cycle_probs = node_cycles[indices].sum(axis=0)
                best_cycle_idx = cluster_cycle_probs.argmax()

                is_recurring = (best_cycle_idx != 0) and (len(indices) > 1)

                if is_recurring:
                    pid = f"{group['accountId'].iloc[0]}_{group['direction'].iloc[0]}_{cluster_id}"
                    cycle = self.idx_to_cycle[best_cycle_idx]
                else:
                    pid = None
                    cycle = None

                pred_is_rec_list.append(is_recurring)
                pred_pids_list.append(pid)
                pred_cycles_list.append(cycle)

            res_df['pred_isRecurring'] = pred_is_rec_list
            res_df['pred_patternId'] = pred_pids_list
            res_df['pred_patternCycle'] = pred_cycles_list

            results_list.append(res_df)