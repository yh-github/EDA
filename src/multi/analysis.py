import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def analyze_classification_mistakes(model, loader, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors specifically.
    Accepts a DataLoader to avoid re-tokenizing data.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")

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

            # Probs
            probs = torch.softmax(cycle_logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            probs_np = max_probs.cpu().numpy()

            # Data for reporting
            amounts = batch['amounts'].numpy()  # Log amounts
            days = batch['days'].numpy()

            batch_size, seq_len = preds.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    pred_cls = preds[b, s]
                    true_cls = targets[b, s]
                    prob = probs_np[b, s]

                    # Reconstruct Amount (Approx) from Log-Abs
                    log_amt = amounts[b, s, 0]
                    approx_amt = np.exp(log_amt) - 1
                    day_val = days[b, s, 0]

                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    item = {
                        'amt': approx_amt,
                        'day': day_val,
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

    fps_sorted = sorted(fps, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.2f}] Amt:~${x['amt']:.2f} Day:{x['day']:.0f} (Cycle Pred: {x['pred_cycle']})")

    fns_sorted = sorted(fns, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            logger.info(f"  [{x['prob']:.2f}] Amt:~${x['amt']:.2f} Day:{x['day']:.0f} (TrueCycle: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, loader, config, num_examples=5):
    """
    Runs inference and logs Adjacency False Positives/Negatives.
    """
    logger.info("Running Adjacency Mistake Analysis...")

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

            amounts = batch['amounts'].numpy()

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

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                amt1 = np.exp(amounts[b, i, 0]) - 1
                amt2 = np.exp(amounts[b, j, 0]) - 1

                item = {
                    'prob': prob,
                    'amt1': amt1,
                    'amt2': amt2
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
            logger.info(f"  [{x['prob']:.4f}] Amt1:~${x['amt1']:.2f} <--> Amt2:~${x['amt2']:.2f}")

    fns_sorted = sorted(fns, key=lambda z: z['prob'])[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top Adjacency FNs (Model missed these):")
        for x in fns_sorted:
            logger.info(f"  [{x['prob']:.4f}] Amt1:~${x['amt1']:.2f} <--> Amt2:~${x['amt2']:.2f}")

    logger.info("-" * 50)