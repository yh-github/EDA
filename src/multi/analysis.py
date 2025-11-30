import logging
import torch
import numpy as np
import pandas as pd
from multi.config import MultiFieldConfig
from multi.data import get_dataloader, MultiTransactionDataset

logger = logging.getLogger(__name__)


def _align_df_for_dataset(df: pd.DataFrame, fields: MultiFieldConfig) -> pd.DataFrame:
    """
    Simulates the filtering and indexing performed inside MultiTransactionDataset.
    This ensures that batch['original_index'] maps correctly to this dataframe.
    """
    df = df.copy()

    # 1. Simulate the Dataset's filtering logic
    # (Referencing MultiTransactionDataset.__init__)
    df['direction'] = np.sign(df[fields.amount])
    df = df[df['direction'] != 0]

    # 2. Reset index to match the Dataset's internal 0..N indexing
    df = df.reset_index(drop=True)
    return df


def analyze_classification_mistakes(model, val_df, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors specifically.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")

    # Align Dataframe indices with Dataset indices
    fc = MultiFieldConfig()
    aligned_df = _align_df_for_dataset(val_df, fc)

    model.eval()
    # Note: get_dataloader will create a NEW dataset instance, which will do the filtering again.
    # Because we pass the raw val_df to get_dataloader, it handles logic internally.
    # But for lookup, we must use 'aligned_df'.
    val_loader = get_dataloader(val_df, config, shuffle=False, n_workers=0)
    device = config.device

    fps = []
    fns = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            _, cycle_logits, _ = model(batch_gpu)

            # Predictions: [Batch, Seq]
            preds = torch.argmax(cycle_logits, dim=-1)
            targets = batch_gpu['cycle_target']
            padding_mask = batch_gpu['padding_mask']

            probs = torch.softmax(cycle_logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)

            # Move to CPU
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            mask_np = padding_mask.cpu().numpy()
            indices_np = batch['original_index'].numpy()
            probs_np = max_probs.cpu().numpy()

            batch_size, seq_len = preds.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    pred_cls = preds_np[b, s]
                    true_cls = targets_np[b, s]
                    prob = probs_np[b, s]

                    # Convert to Binary Logic: 0 is None/Noise, >0 is Recurring
                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    # Get original row using the Dataset's internal index
                    orig_idx = indices_np[b, s]

                    try:
                        row = aligned_df.iloc[orig_idx]
                    except IndexError:
                        logger.error(f"Index alignment error: {orig_idx} not in df len {len(aligned_df)}")
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
                        fps.append(item)  # False Positive
                    elif not pred_is_rec and true_is_rec:
                        fns.append(item)  # False Negative

    # Log Results
    logger.info("-" * 60)
    logger.info(f"CLASSIFICATION MISTAKES (isRecurring): FP={len(fps)}, FN={len(fns)}")

    # Sort by confidence
    fps_sorted = sorted(fps, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.2f}] Amt:{x['amt']:<8} Date:{x['date']} Desc:'{x['txt']}'")

    fns_sorted = sorted(fns, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            logger.info(
                f"  [{x['prob']:.2f}] Amt:{x['amt']:<8} Date:{x['date']} Desc:'{x['txt']}' (TrueCycle: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, val_df, config, num_examples=5):
    """
    Runs inference and logs Adjacency False Positives/Negatives.
    """
    logger.info("Running Adjacency Mistake Analysis...")

    fc = MultiFieldConfig()
    aligned_df = _align_df_for_dataset(val_df, fc)

    model.eval()
    val_loader = get_dataloader(val_df, config, shuffle=False, n_workers=0)
    device = config.device

    fps = []  # False Positives (Pred=1, True=0)
    fns = []  # False Negatives (Pred=0, True=1)

    with torch.no_grad():
        for batch in val_loader:
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

            # Find mismatches: (Pred != Target) & Valid & UpperTri
            mismatches = (preds != targets) & final_mask

            # Get indices of mismatches
            b_indices, i_indices, j_indices = torch.where(mismatches)

            if len(b_indices) == 0:
                continue

            # Move necessary data to CPU
            b_indices = b_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()

            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            indices_np = batch['original_index'].numpy()

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                # Get original DF indices
                idx_i = indices_np[b, i]
                idx_j = indices_np[b, j]

                try:
                    row_i = aligned_df.iloc[idx_i]
                    row_j = aligned_df.iloc[idx_j]
                except IndexError:
                    continue

                item = {
                    'prob': prob,
                    'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                    'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                }

                if pred == 1 and truth == 0:
                    fps.append(item)
                elif pred == 0 and truth == 1:
                    fns.append(item)

    logger.info("-" * 50)
    logger.info(f"ADJACENCY MISTAKES (Found {len(fps)} FPs, {len(fns)} FNs)")

    fps_sorted = sorted(fps, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top Adjacency FPs (Model thought these were same):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.4f}] '{x['txt1']}' (${x['amt1']}) <--> '{x['txt2']}' (${x['amt2']})")

    fns_sorted = sorted(fns, key=lambda z: z['prob'])[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top Adjacency FNs (Model missed these):")
        for x in fns_sorted:
            logger.info(f"  [{x['prob']:.4f}] '{x['txt1']}' (${x['amt1']}) <--> '{x['txt2']}' (${x['amt2']})")

    logger.info("-" * 50)