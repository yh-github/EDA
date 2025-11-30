import logging
import torch
import numpy as np
import pandas as pd
from multi.config import MultiFieldConfig

logger = logging.getLogger(__name__)


def _align_df_for_dataset(df: pd.DataFrame, fields: MultiFieldConfig) -> pd.DataFrame:
    """
    Simulates the filtering and indexing performed inside MultiTransactionDataset.
    This ensures we have a 1-to-1 mapping with the DataLoader output.
    """
    df = df.copy()

    # 1. Simulate the Dataset's filtering logic
    df['direction'] = np.sign(df[fields.amount])
    df = df[df['direction'] != 0]

    # 2. Reset index to ensure clean iteration
    df = df.reset_index(drop=True)
    return df


def analyze_classification_mistakes(model, val_df, loader, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors specifically.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")

    fc = MultiFieldConfig()
    # PRE-ALIGNMENT: This dataframe now exactly matches the rows in the DataLoader
    aligned_df = _align_df_for_dataset(val_df, fc)

    model.eval()
    device = config.device

    fps = []
    fns = []

    # We use a global index tracker to walk through 'aligned_df'
    # NOTE: The DataLoader groups transactions by Account. The aligned_df is flat.
    # The Dataset internal logic groups by Account.
    # Therefore, row 'i' in aligned_df might NOT match row 'i' in the loader sequence
    # because the loader yields (Batch, Seq) of grouped accounts.

    # Correction: MultiTransactionDataset groups by (Account, Direction).
    # It stores self.groups = [df_group1, df_group2, ...]
    # The DataLoader randomizes these groups if shuffle=True.
    # Since we use shuffle=False, the order of groups is deterministic.

    # We need to replicate the Grouping logic to match the batches.
    groups = [group for _, group in aligned_df.groupby([fc.accountId, 'direction'])]

    # Flatten groups back into a list of (group_idx, row_in_group) mapping?
    # No, we just need to iterate through 'groups' in the same order the loader does.

    current_group_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            _, cycle_logits, _ = model(batch_gpu)

            # Predictions
            preds = torch.argmax(cycle_logits, dim=-1).cpu().numpy()
            targets = batch_gpu['cycle_target'].cpu().numpy()
            mask_np = batch_gpu['padding_mask'].cpu().numpy()

            probs = torch.softmax(cycle_logits, dim=-1)
            # FIX: Unpack the tuple (values, indices) BEFORE calling .cpu()
            max_probs_tensor, _ = torch.max(probs, dim=-1)
            probs_np = max_probs_tensor.cpu().numpy()

            batch_size, seq_len = preds.shape

            # The loader returns a batch of groups.
            # batch[0] corresponds to groups[current_group_idx]
            # batch[1] corresponds to groups[current_group_idx + 1]

            for b in range(batch_size):
                if current_group_idx >= len(groups):
                    break

                # Get the source dataframe slice for this group
                # Note: The Dataset also sorts by date and truncates to max_seq_len.
                group_df = groups[current_group_idx].sort_values(fc.date, ascending=True)
                if len(group_df) > config.max_seq_len:
                    group_df = group_df.iloc[-config.max_seq_len:]

                current_group_idx += 1

                # Iterate through the sequence
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    # Safety check: if mask is true, we should have a row
                    if s >= len(group_df):
                        continue

                    pred_cls = preds[b, s]
                    true_cls = targets[b, s]
                    prob = probs_np[b, s]

                    row = group_df.iloc[s]

                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    item = {
                        'txt': row[fc.text],
                        'amt': row[fc.amount],
                        'date': row[fc.date],
                        'prob': prob,
                        'pred_cycle': pred_cls,
                        'true_cycle': true_cls
                    }

                    if pred_is_rec and not true_is_rec:
                        fps.append(item)
                    elif not pred_is_rec and true_is_rec:
                        fns.append(item)

    # Log Results
    logger.info("-" * 60)
    logger.info(f"CLASSIFICATION MISTAKES (isRecurring): FP={len(fps)}, FN={len(fns)}")

    # Sort by confidence
    fps_sorted = sorted(fps, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            # Shorten description
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt}")

    fns_sorted = sorted(fns, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt} (True: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, val_df, loader, config, num_examples=5):
    """
    Runs inference and logs Adjacency False Positives/Negatives.
    """
    logger.info("Running Adjacency Mistake Analysis...")

    fc = MultiFieldConfig()
    aligned_df = _align_df_for_dataset(val_df, fc)

    # Replicate grouping
    groups = [group for _, group in aligned_df.groupby([fc.accountId, 'direction'])]
    current_group_idx = 0

    model.eval()
    device = config.device

    fps = []
    fns = []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            adj_logits, _, _ = model(batch_gpu)
            probs = torch.sigmoid(adj_logits)

            preds = (probs > 0.5).float()
            targets = batch_gpu['adjacency_target']

            # Create masks
            b_size, s_len, _ = probs.shape
            triu_mask = torch.triu(torch.ones(s_len, s_len, device=device), diagonal=1).bool()
            triu_mask = triu_mask.unsqueeze(0).expand(b_size, -1, -1)

            valid_mask = batch_gpu['padding_mask'].unsqueeze(1) & batch_gpu['padding_mask'].unsqueeze(2)
            final_mask = triu_mask & valid_mask

            # Find mismatches
            mismatches = (preds != targets) & final_mask
            b_indices, i_indices, j_indices = torch.where(mismatches)

            if len(b_indices) == 0:
                current_group_idx += b_size
                continue

            # Move to CPU
            b_indices = b_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                # Identify Group
                # The loop index 'b' is relative to the batch.
                # We need to map it to the global groups list.
                # Since we increment current_group_idx by b_size at the end of loop,
                # the index is simply current_group_idx + b.
                # HOWEVER, this loop iterates mismatches, so we can't increment inside.

                actual_group_idx = current_group_idx + b
                if actual_group_idx >= len(groups): break

                group_df = groups[actual_group_idx].sort_values(fc.date, ascending=True)
                if len(group_df) > config.max_seq_len:
                    group_df = group_df.iloc[-config.max_seq_len:]

                if i >= len(group_df) or j >= len(group_df): continue

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                row_i = group_df.iloc[i]
                row_j = group_df.iloc[j]

                item = {
                    'prob': prob,
                    'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                    'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                }

                if pred == 1 and truth == 0:
                    fps.append(item)
                elif pred == 0 and truth == 1:
                    fns.append(item)

            # Advance the global group pointer
            current_group_idx += b_size

    logger.info("-" * 50)
    logger.info(f"ADJACENCY MISTAKES (Found {len(fps)} FPs, {len(fns)} FNs)")

    fps_sorted = sorted(fps, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top Adjacency FPs (Model thought these were same):")
        for x in fps_sorted:
            t1 = (str(x['txt1'])[:25] + '..') if len(str(x['txt1'])) > 25 else str(x['txt1'])
            t2 = (str(x['txt2'])[:25] + '..') if len(str(x['txt2'])) > 25 else str(x['txt2'])
            logger.info(f"  [{x['prob']:.4f}] {t1} (${x['amt1']}) <--> {t2} (${x['amt2']})")

    fns_sorted = sorted(fns, key=lambda z: z['prob'])[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top Adjacency FNs (Model missed these):")
        for x in fns_sorted:
            t1 = (str(x['txt1'])[:25] + '..') if len(str(x['txt1'])) > 25 else str(x['txt1'])
            t2 = (str(x['txt2'])[:25] + '..') if len(str(x['txt2'])) > 25 else str(x['txt2'])
            logger.info(f"  [{x['prob']:.4f}] {t1} (${x['amt1']}) <--> {t2} (${x['amt2']})")

    logger.info("-" * 50)