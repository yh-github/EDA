import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from transformers import AutoTokenizer
from multi.config import MultiExpConfig
from multi.encoder import TransactionTransformer
from multi.data import collate_fn


class MultiPredictor:
    def __init__(self, model_path: str, config: MultiExpConfig):
        self.config = config
        self.device = config.device

        # The encoder uses a live Transformer, so we need the Tokenizer, not EmbeddingService
        print(f"Loading Tokenizer: {config.text_encoder_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)

        print(f"Loading model from {model_path}...")
        self.model = TransactionTransformer(config)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.idx_to_cycle = {v: k for k, v in config.cycle_map.items()}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Prepare Groups
        df['direction'] = np.sign(df['amount'])
        # Filter 0 amounts
        df = df[df['direction'] != 0].copy()

        # Sort to ensure order matches grouping
        groups = [group for _, group in df.groupby(['accountId', 'direction'])]

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(groups), batch_size):
            batch_groups = groups[i: i + batch_size]
            batch_data = self._prepare_batch_data(batch_groups)

            if not batch_data: continue

            # FIX: Pass self.tokenizer, NOT embedding_service
            batch = collate_fn(batch_data, self.tokenizer, self.config)
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                adj_logits, cycle_logits = self.model(batch)

            self._process_batch_results(batch_groups, adj_logits, cycle_logits, results)

        if not results:
            return pd.DataFrame()

        return pd.concat(results).sort_index()

    def _prepare_batch_data(self, groups):
        batch_list = []
        for group in groups:
            # Sort by date for temporal consistency
            group = group.sort_values('date')

            # Text Features
            texts = (group['bankRawDescription'].fillna('') + " " + group['counter_party'].fillna('')).tolist()

            # Amount Features
            amounts = group['amount'].values.astype(np.float32)
            log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

            # Date Features
            dates = pd.to_datetime(group['date'])
            days = (dates - dates.min()).dt.days.values.astype(np.float32)

            # Dummy targets for inference (required by collate_fn shapes)
            batch_list.append({
                "texts": texts,
                "amounts": log_amounts,
                "days": days,
                "pattern_ids": np.zeros(len(group)) - 1,  # -1 indicates no pattern/unknown
                "cycles": np.zeros(len(group)),
            })
        return batch_list

    def _process_batch_results(self, groups, adj_logits, cycle_logits, results_list):
        probs = torch.sigmoid(adj_logits).cpu().numpy()
        cycle_softmax = torch.softmax(cycle_logits, dim=-1).cpu().numpy()

        for b_idx, group in enumerate(groups):
            n = len(group)
            if n == 0: continue

            # Extract sub-matrices for this group
            adj = probs[b_idx, :n, :n]
            node_cycles = cycle_softmax[b_idx, :n, :]

            # Recurring Score = Prob that cycle is NOT 'None'
            recurring_scores = 1.0 - node_cycles[:, 0]

            # Clustering Logic
            adj_binary = (adj > 0.5).astype(int)
            np.fill_diagonal(adj_binary, 0)
            graph = csr_matrix(adj_binary)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            final_pids = [None] * n
            final_cycles = ["None"] * n
            final_is_rec = [False] * n

            for cluster_id in range(n_components):
                indices = np.where(labels == cluster_id)[0]

                # Single nodes are noise unless strongly cyclic?
                # For now, enforce size > 1 for a cluster
                if len(indices) < 2:
                    continue

                # Vote on Cycle
                cluster_sum_probs = node_cycles[indices].sum(axis=0)
                best_cycle_idx = cluster_sum_probs.argmax()

                # If best cycle is 0 ('None'), it's a cluster of noise
                is_recurring = (best_cycle_idx != 0)

                if is_recurring:
                    pid_str = f"{group['accountId'].iloc[0]}_{group['direction'].iloc[0]}_{cluster_id}"
                    cycle_str = self.idx_to_cycle.get(best_cycle_idx, "None")

                    for idx in indices:
                        final_pids[idx] = pid_str
                        final_cycles[idx] = cycle_str
                        final_is_rec[idx] = True

            # Assign results back to dataframe
            res_df = group.copy()
            res_df['pred_isRecurring'] = final_is_rec
            res_df['pred_recurring_prob'] = recurring_scores
            res_df['pred_patternId'] = final_pids
            res_df['pred_patternCycle'] = final_cycles

            results_list.append(res_df)