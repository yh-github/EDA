import logging
import torch
import numpy as np
import pandas as pd
from multi.config import MultiFieldConfig

logger = logging.getLogger(__name__)


def analyze_classification_mistakes(model, df, loader, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors using direct index lookup.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")

    fc = MultiFieldConfig()
    model.eval()
    device = config.device

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
            probs = torch.softmax(cycle_logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1).cpu().numpy()

            # Retrieve Indices from batch
            indices_np = batch['original_index'].numpy()

            batch_size, seq_len = preds.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    pred_cls = preds[b, s]
                    true_cls = targets[b, s]
                    prob = max_probs[b, s]

                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    # --- DIRECT LOOKUP ---
                    true_idx = indices_np[b, s]
                    try:
                        row = df.loc[true_idx]

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

                    except KeyError:
                        logger.warning(f"Index {true_idx} not found in DataFrame.")
                        continue

    # Log Results
    logger.info("-" * 60)
    logger.info(f"CLASSIFICATION MISTAKES (isRecurring): FP={len(fps)}, FN={len(fns)}")

    fps_sorted = sorted(fps, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt}")

    fns_sorted = sorted(fns, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            txt = (str(x['txt'])[:40] + '..') if len(str(x['txt'])) > 40 else str(x['txt'])
            logger.info(f"  [{x['prob']:.2f}] ${x['amt']:<6.2f} {x['date']} | {txt} (True: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, df, loader, config, num_examples=5):
    """
    Runs inference and logs Adjacency False Positives/Negatives using direct index lookup.
    """
    logger.info("Running Adjacency Mistake Analysis...")

    fc = MultiFieldConfig()
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
                continue

            # Move to CPU
            b_indices = b_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Retrieve Indices from batch
            indices_np = batch['original_index'].numpy()

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                # --- DIRECT LOOKUP ---
                idx_i = indices_np[b, i]
                idx_j = indices_np[b, j]

                try:
                    row_i = df.loc[idx_i]
                    row_j = df.loc[idx_j]

                    item = {
                        'prob': prob,
                        'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                        'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                    }

                    if pred == 1 and truth == 0:
                        fps.append(item)
                    elif pred == 0 and truth == 1:
                        fns.append(item)

                except KeyError:
                    continue

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