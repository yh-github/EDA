import logging
import pandas as pd
import diskcache
from pathlib import Path

# --- Configuration ---
# Point this to the cache you want to analyze
CACHE_DIR = Path('cache/results/')
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


def main():
    df = load_results_to_dataframe(CACHE_DIR)

    if df is None or df.empty:
        logger.info("Exiting.")
        return

    # --- 1. Find the "Best" and "Most Stable" Models ---

    # This ensures 'cv_loss' etc. are not treated as parameters
    metric_cols = [
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

    for col in varying_param_cols:
        if col in df.columns:
            try:
                logger.info(f"\n--- Average F1 by {col} ---")
                # This will fail for lists
                perf = df.groupby(col)['cv_f1'].mean().sort_values(ascending=False)
                logger.info(perf.to_string())
            except TypeError as e:
                if "unhashable type" in str(e):
                    # If it's a list, group by its string representation
                    logger.info(f"--- Average F1 by {col} (as string) ---")
                    perf_str = df.groupby(df[col].astype(str))['cv_f1'].mean().sort_values(ascending=False)
                    logger.info(perf_str.to_string())
                else:
                    logger.warning(f"Could not group by column '{col}'. Error: {e}")
            except Exception as e:
                logger.warning(f"General error analyzing column '{col}': {e}")


if __name__ == "__main__":
    main()