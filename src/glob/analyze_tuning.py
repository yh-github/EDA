import logging
import sys
import pandas as pd
import diskcache
from pathlib import Path

# --- Configuration ---
# Point this to the cache you want to analyze
CACHE_DIR = Path('cache/results/')
# Set a threshold for what you consider a "failed" run
MIN_F1_FOR_ANALYSIS = 0.3
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_results_to_dataframe(cache_path: Path) -> pd.DataFrame | None:
    """Loads all results from a diskcache into a flat pandas DataFrame."""

    logger.info(f"Loading results from cache: {cache_path}")
    if not cache_path.exists():
        logger.error(f"Error: Cache directory not found at {cache_path}")
        return None

    try:
        with diskcache.Cache(str(cache_path)) as cache:
            # Use the correct iteration method you found
            all_results = [cache[k] for k in cache]
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return None

    if not all_results:
        logger.warning("Cache is empty. No results to analyze.")
        return None

    logger.info(f"Found {len(all_results)} results.")

    try:
        all_results_df = pd.DataFrame(all_results)
        metrics_df = pd.json_normalize(all_results_df['metrics'])
        params_df = pd.json_normalize(all_results_df['params'])
        summary_df = pd.concat([params_df, metrics_df], axis=1).fillna(0)
        summary_df = summary_df.drop(columns=['params', 'metrics'], errors='ignore')
        return summary_df
    except Exception as e:
        logger.error(f"Error flattening data into DataFrame: {e}")
        return None


def main(dir_index: str):
    df = load_results_to_dataframe(CACHE_DIR / dir_index)

    if df is None or df.empty:
        logger.info("Exiting.")
        return

    # Check if 'use_categorical_amount' was part of this experiment
    if 'feat_proc_params.use_categorical_amount' in df.columns:
        logger.info("Normalizing conditional parameters for 'use_categorical_amount'...")
        # Create a mask for all rows where the feature was OFF
        mask = df['feat_proc_params.use_categorical_amount'] == False

        # If k_top exists, set it to 0 for all "OFF" rows
        if 'feat_proc_params.k_top' in df.columns:
            df.loc[mask, 'feat_proc_params.k_top'] = 0

        # If n_bins exists, set it to 0 for all "OFF" rows
        if 'feat_proc_params.n_bins' in df.columns:
            df.loc[mask, 'feat_proc_params.n_bins'] = 0

    # --- 1. Find the "Best" and "Most Stable" Models ---

    # This ensures 'cv_loss' etc. are not treated as parameters
    metric_cols = [
        'cv_val_best_f1', 'cv_val_best_threshold',
        'cv_f1', 'cv_f1_std', 'cv_roc_auc', 'cv_loss',
        'loss', 'accuracy', 'f1', 'roc_auc', 'error'
    ]
    present_metric_cols = [col for col in metric_cols if col in df.columns]

    # Param cols are everything else
    param_cols = [col for col in df.columns if col not in present_metric_cols]

    varying_param_cols = []
    for col in param_cols:
        try:
            is_varying = df[col].nunique() > 1
        except TypeError:
            try:
                is_varying = df[col].astype(str).nunique() > 1
            except Exception as e:
                logger.warning(f"Could not determine uniqueness for column '{col}'. Skipping. Error: {e}")
                is_varying = False

        if is_varying:
            varying_param_cols.append(col)

    logger.info(f"Analyzing {len(varying_param_cols)} varying parameters: {varying_param_cols}")

    cols_to_show = present_metric_cols + varying_param_cols
    top_10_models = df.sort_values(by='cv_f1', ascending=False).head(10)

    logger.info("\n--- Top 10 Best-Performing Models (Only Showing Varying Params) ---")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logger.info(top_10_models[cols_to_show].to_string())

    logger.info("\n--- Analysis ---")
    logger.info("Look for a model with high 'cv_f1' AND low 'cv_f1_std'.")

    # --- 2. Analyze Trends (GroupBy) ---

    # --- helper columns for aggregation ---
    df['is_failure'] = df['cv_f1'] < MIN_F1_FOR_ANALYSIS
    # Create a copy of 'cv_f1' where failures are NaN, so mean() ignores them
    df['success_f1'] = df['cv_f1'].where(df['cv_f1'] >= MIN_F1_FOR_ANALYSIS)

    if df['is_failure'].any():
        logger.info(
            f"\n--- NOTE: Analyzing trends. 'Failed' runs (F1 < {MIN_F1_FOR_ANALYSIS}) are counted separately. ---")

    for col in varying_param_cols:
        if col in df.columns:
            try:
                logger.info(f"\n--- Analysis by {col} ---")

                # Handle unhashable types (like lists) by grouping by their string representation
                if df[col].apply(type).isin([list, dict]).any():
                    logger.info("(Grouping by string representation for list/dict params)")
                    group_key = df[col].astype(str)
                else:
                    group_key = df[col]

                # Perform the complex aggregation
                # Define the aggregations dictionary dynamically
                agg_dict = {
                    'total_runs': ('cv_f1', 'count'),
                    'failed_runs': ('is_failure', 'sum'),
                    'avg_success_f1': ('success_f1', 'mean'),
                    'avg_f1_std': ('success_f1', 'std'),
                }

                # Add aggregations for the new metrics if they exist
                if 'cv_val_best_f1' in df.columns:
                    agg_dict['avg_potential_f1'] = ('success_potential_f1', 'mean')

                if 'cv_val_best_threshold' in df.columns:
                    agg_dict['avg_optim_thresh'] = ('cv_val_best_threshold', 'mean')

                if 'cv_best_epoch' in df.columns:
                    agg_dict['avg_best_epoch'] = ('cv_best_epoch', 'mean')

                if 'cv_epoch_1_stop' in df.columns:
                    agg_dict['epoch_1_fail_rate'] = ('cv_epoch_1_stop', 'mean')

                analysis = df.groupby(group_key).agg(**agg_dict)

                analysis['failure_rate'] = (analysis['failed_runs'] / analysis['total_runs']).apply(
                    lambda x: f"{x:.0%}")

                # 1. Fill NaNs in the sorting key (avg_success_f1) with 0 so we can sort.
                #    We will fill them with a string for display *after* sorting.
                analysis_sorted = analysis.fillna({'avg_success_f1': 0}).sort_values(
                    by='avg_success_f1',
                    ascending=False
                )

                # 2. Now that it's sorted, format for display
                analysis_sorted['avg_success_f1'] = analysis['avg_success_f1'].apply(
                    lambda x: f"{x:.6f}" if x > 0 else 'N/A (All Failed)'
                )
                analysis_sorted['avg_f1_std'] = analysis['avg_f1_std'].fillna(0).apply(
                    lambda x: f"{x:.6f}"
                )

                display_cols = ['total_runs', 'failed_runs', 'failure_rate', 'avg_success_f1', 'avg_f1_std']

                if 'epoch_1_fail_rate' in analysis.columns:
                    display_cols.append('epoch_1_fail_rate')

                display_cols += ['avg_success_f1', 'avg_f1_std']

                if 'avg_best_epoch' in analysis.columns:
                    display_cols.append('avg_best_epoch')

                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                    logger.info(analysis_sorted[display_cols].to_string())

            except Exception as e:
                logger.warning(f"General error analyzing column '{col}': {e}", exc_info=True)

if __name__ == "__main__":
    ind = sys.argv[1] if len(sys.argv) > 1 else ""
    main(ind)