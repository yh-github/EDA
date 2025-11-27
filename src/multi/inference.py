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
from multi.data import collate_fn

logger = logging.getLogger(__name__)


class MultiPredictor:
    def __init__(self, model_path: str, runtime_config: MultiExpConfig = None):
        if runtime_config is None:
            runtime_config = MultiExpConfig()

        self.device = runtime_config.device
        self.field_config = MultiFieldConfig()

        logger.info(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            saved_config = checkpoint["config"]
            state_dict = checkpoint["state_dict"]
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

        req_cols = [fc.accountId, fc.date, fc.amount, fc.text]
        for col in req_cols:
            if col not in df.columns:
                raise KeyError(f"Input DataFrame missing required column: {col}")

        df['direction'] = np.sign(df[fc.amount])
        df = df[df['direction'] != 0].copy()

        groups = [group for _, group in df.groupby([fc.accountId, 'direction'])]

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(groups), batch_size):
            batch_groups = groups[i: i + batch_size]
            batch_data = self._prepare_batch_data(batch_groups)

            if not batch_data: continue

            batch = collate_fn(batch_data, self.tokenizer, self.config)
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                adj_logits, cycle_logits = self.model(batch)

            self._process_batch_results(batch_groups, adj_logits, cycle_logits, results)

        if not results:
            return pd.DataFrame()

        return pd.concat(results).sort_index()

    def _prepare_batch_data(self, groups):
        fc = self.field_config
        use_cp = self.config.use_counter_party
        batch_list = []

        for group in groups:
            group = group.sort_values(fc.date)

            texts = group[fc.text].fillna('').tolist()

            cps = []
            if use_cp:
                if fc.counter_party in group.columns:
                    cps = group[fc.counter_party].fillna('').tolist()
                else:
                    cps = [""] * len(texts)

            amounts = group[fc.amount].values.astype(np.float32)
            log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

            dates = pd.to_datetime(group[fc.date])
            min_date = dates.iloc[0]
            days = (dates - min_date).dt.days.values.astype(np.float32)

            batch_list.append({
                "texts": texts,
                "cps": cps,
                "amounts": log_amounts,
                "days": days,
                "pattern_ids": np.zeros(len(group)) - 1,
                "cycles": np.zeros(len(group)),
            })
        return batch_list

    def _process_batch_results(self, groups, adj_logits, cycle_logits, results_list):
        probs = torch.sigmoid(adj_logits).cpu().numpy()
        cycle_softmax = torch.softmax(cycle_logits, dim=-1).cpu().numpy()

        for b_idx, group in enumerate(groups):
            n = len(group)
            if n == 0: continue

            adj = probs[b_idx, :n, :n]
            node_cycles = cycle_softmax[b_idx, :n, :]

            recurring_scores = 1.0 - node_cycles[:, 0]

            adj_binary = (adj > 0.5).astype(int)
            np.fill_diagonal(adj_binary, 0)
            graph = csr_matrix(adj_binary)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            final_pids = [None] * n
            final_cycles = ["None"] * n
            final_is_rec = [False] * n

            for cluster_id in range(n_components):
                indices = np.where(labels == cluster_id)[0]

                if len(indices) < 2:
                    continue

                cluster_sum_probs = node_cycles[indices].sum(axis=0)
                best_cycle_idx = cluster_sum_probs.argmax()

                is_recurring = (best_cycle_idx != 0)

                if is_recurring:
                    pid_str = f"{group[self.field_config.accountId].iloc[0]}_{group['direction'].iloc[0]}_{cluster_id}"
                    cycle_str = self.idx_to_cycle.get(best_cycle_idx, "None")

                    for idx in indices:
                        final_pids[idx] = pid_str
                        final_cycles[idx] = cycle_str
                        final_is_rec[idx] = True

            res_df = group.copy()
            res_df['pred_isRecurring'] = final_is_rec
            res_df['pred_recurring_prob'] = recurring_scores
            res_df['pred_patternId'] = final_pids
            res_df['pred_patternCycle'] = final_cycles

            results_list.append(res_df)