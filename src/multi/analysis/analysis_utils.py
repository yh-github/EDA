import logging
import torch
import pandas as pd
from common.config import get_device
from multi.config import MultiFieldConfig

logger = logging.getLogger(__name__)


def _align_df_for_dataset(df: pd.DataFrame, fields: MultiFieldConfig) -> pd.DataFrame:
    # Ensure dataframe is indexed appropriately for fast lookup
    if '_true_index' in df.columns:
        return df.set_index('_true_index')
    return df


def analyze_classification_mistakes(model, val_df, loader, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors specifically.
    Refactored to use 'original_index' for alignment.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")

    fc = MultiFieldConfig()

    # We use the original dataframe for lookups
    # Ensure it has a reliable index.
    # If val_df was reset, we need to map back using _true_index if available,
    # or assume val_df index is the ground truth.
    # The dataset generates 'original_index' based on val_df['_true_index'] if it exists,
    # or val_df.index otherwise.

    lookup_df = val_df.copy()
    if '_true_index' in lookup_df.columns:
        lookup_df = lookup_df.set_index('_true_index')

    device = get_device()
    model.to(device)
    model.eval()

    fps = []
    fns = []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            _, cycle_logits, _ = model(batch_gpu)

            # Predictions
            preds = torch.argmax(cycle_logits, dim=-1).cpu().numpy()
            targets = batch_gpu['cycle_target'].cpu().numpy()
            mask_np = batch_gpu['padding_mask'].cpu().numpy()
            original_indices = batch_gpu['original_index'].cpu().numpy()

            probs = torch.softmax(cycle_logits, dim=-1)
            max_probs_tensor, _ = torch.max(probs, dim=-1)
            probs_np = max_probs_tensor.cpu().numpy()

            batch_size, seq_len = preds.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    idx = original_indices[b, s]
                    if idx == -1: continue  # Padding

                    pred_cls = preds[b, s]
                    true_cls = targets[b, s]
                    prob = probs_np[b, s]

                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    # Retrieve Row
                    try:
                        row = lookup_df.loc[idx]
                    except KeyError:
                        logger.warning(f"Index {idx} not found in validation dataframe. Skipping.")
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

    logger.info("-" * 60)
    logger.info(f"CLASSIFICATION MISTAKES (isRecurring): FP={len(fps)}, FN={len(fns)}")

    fps_sorted = sorted(fps, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt}")

    fns_sorted = sorted(fns, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt} (True: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, val_df, loader, config, num_examples=5):
    """
    Runs inference and logs Adjacency False Positives/Negatives.
    Refactored to use 'original_index' for alignment.
    """
    logger.info("Running Adjacency Mistake Analysis...")

    fc = MultiFieldConfig()

    lookup_df = val_df.copy()
    if '_true_index' in lookup_df.columns:
        lookup_df = lookup_df.set_index('_true_index')

    device = get_device()
    model.to(device)
    model.eval()

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
            original_indices = batch_gpu['original_index'].cpu().numpy()

            # Create masks
            b_size, s_len, _ = probs.shape
            triu_mask = torch.triu(torch.ones(s_len, s_len, device=device), diagonal=1).bool()
            triu_mask = triu_mask.unsqueeze(0).expand(b_size, -1, -1)

            valid_mask = batch_gpu['padding_mask'].unsqueeze(1) & batch_gpu['padding_mask'].unsqueeze(2)
            final_mask = triu_mask & valid_mask

            mismatches = (preds != targets) & final_mask
            b_indices, i_indices, j_indices = torch.where(mismatches)

            if len(b_indices) == 0:
                continue

            b_indices = b_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                idx_i = original_indices[b, i]
                idx_j = original_indices[b, j]

                if idx_i == -1 or idx_j == -1: continue

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                try:
                    row_i = lookup_df.loc[idx_i]
                    row_j = lookup_df.loc[idx_j]
                except KeyError:
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