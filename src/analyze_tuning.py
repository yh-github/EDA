import logging
import pandas as pd
import diskcache
from pathlib import Path

CACHE_DIR = Path('cache/results/')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_results_to_dataframe(cache_path: Path) -> pd.DataFrame | None:
    """Loads all results from a diskcache into a flat pandas DataFrame."""

    logger.info(f"Loading results from cache: {cache_path}")
    if not cache_path.exists():
        logger.error(f"Error: Cache directory not found at {cache_path}")
        return None

    try:
        with diskcache.Cache(cache_path) as cache:
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

    # --- THIS IS THE NEW LOGIC ---

    # 1. Define key metrics we *always* want to see
    metric_cols = [
        'cv_f1',
        'cv_f1_std',
        'cv_roc_auc'
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]

    # 2. Find all parameter columns that *actually vary* across runs
    varying_param_cols = []

    # Get all parameter columns (anything not a metric or error)
    param_cols = [col for col in df.columns if col not in metric_cols and col != 'error']

    for col in param_cols:
        # Check if the column has more than 1 unique value
        if df[col].nunique() > 1:
            varying_param_cols.append(col)

    # 3. Combine them into the final list
    cols_to_show = metric_cols + varying_param_cols
    logger.info(f"Analyzing {len(varying_param_cols)} varying parameters: {varying_param_cols}")
    # --- END OF NEW LOGIC ---

    top_10_models = df.sort_values(by='cv_f1', ascending=False).head(10)

    logger.info("\n--- Top 10 Best-Performing Models (Only Showing Varying Params) ---")

    # Use max_cols=None to prevent truncation of the columns we've selected
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logger.info(top_10_models[cols_to_show].to_string())

    logger.info("\n--- Analysis ---")
    logger.info("Look for a model with high 'cv_f1' AND low 'cv_f1_std'.")

    # --- 2. Analyze Trends (GroupBy) ---
    # This section remains the same, but we'll add checks

    if 'emb_params.model_name' in df.columns:
        logger.info("\n--- Average F1 by Embedding Model ---")
        model_perf = df.groupby('emb_params.model_name')['cv_f1'].mean()
        logger.info(model_perf.to_string())

    if 'model_params.dropout_rate' in df.columns:
        logger.info("\n--- Average F1 by Dropout Rate ---")
        dropout_perf = df.groupby('model_params.dropout_rate')['cv_f1'].mean()
        logger.info(dropout_perf.to_string())

    # Add any other ablations you're running
    if 'feat_proc_params.use_cyclical_dates' in df.columns:
        logger.info("\n--- Ablation: Cyclical Dates ---")
        ablation_perf = df.groupby('feat_proc_params.use_cyclical_dates')['cv_f1'].mean()
        logger.info(ablation_perf.to_string())

    if 'feat_proc_params.use_categorical_dates' in df.columns:
        logger.info("\n--- Ablation: Categorical Dates ---")
        ablation_perf = df.groupby('feat_proc_params.use_categorical_dates')['cv_f1'].mean()
        logger.info(ablation_perf.to_string())


if __name__ == "__main__":
    main()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_results_to_dataframe(cache_path: Path) -> pd.DataFrame | None:
    """Loads all results from a diskcache into a flat pandas DataFrame."""

    logger.info(f"Loading results from cache: {cache_path}")
    if not cache_path.exists():
        logger.error(f"Error: Cache directory not found at {cache_path}")
        return None

    try:
        with diskcache.Cache(cache_path) as cache:
            all_results = [cache[k] for k in cache]
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return None

    if not all_results:
        logger.warning("Cache is empty. No results to analyze.")
        return None

    logger.info(f"Found {len(all_results)} results.")

    # --- Flatten the results ---
    # This logic is from the HyperTuner's final report
    try:
        all_results_df = pd.DataFrame(all_results)

        # 'metrics' and 'params' are columns containing dictionaries
        # pd.json_normalize flattens them into proper columns
        metrics_df = pd.json_normalize(all_results_df['metrics'])
        params_df = pd.json_normalize(all_results_df['params'])

        # Combine the flat param/metric data
        summary_df = pd.concat([params_df, metrics_df], axis=1).fillna(0)

        # Drop columns we don't need for analysis
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

    # Sort by F1 score, but also show stability (std)
    # Your run_cross_validation already provides 'cv_f1_std'
    cols_to_show = [
        'cv_f1',
        'cv_f1_std',
        'cv_roc_auc',
        'emb_params.model_name',
        'exp_params.learning_rate',
        'model_params.mlp_hidden_layers',
        'feat_proc_params.use_cyclical_dates',
        'feat_proc_params.use_categorical_dates',
        'feat_proc_params.use_continuous_amount',
        'feat_proc_params.use_categorical_amount'
    ]

    # Filter for columns that actually exist in the DataFrame
    cols_to_show = [col for col in cols_to_show if col in df.columns]

    top_10_models = df.sort_values(by='cv_f1', ascending=False).head(10)

    logger.info("\n--- Top 10 Best-Performing Models ---")
    logger.info(top_10_models[cols_to_show].to_string())

    logger.info("\n--- Analysis ---")
    logger.info("Look for a model with high 'cv_f1' AND low 'cv_f1_std'.")
    logger.info("The #1 model isn't always the best if its 'cv_f1_std' is high.")

    # --- 2. Analyze Trends (GroupBy) ---
    # This is how you answer "Which parameters matter most?"

    # Example 1: Which embedding model was best on average?
    if 'emb_params.model_name' in df.columns:
        logger.info("\n--- Average F1 by Embedding Model ---")
        model_perf = df.groupby('emb_params.model_name')['cv_f1'].mean()
        logger.info(model_perf.to_string())

    # Example 2: How did dropout rate affect things?
    if 'model_params.dropout_rate' in df.columns:
        logger.info("\n--- Average F1 by Dropout Rate ---")
        dropout_perf = df.groupby('model_params.dropout_rate')['cv_f1'].mean()
        logger.info(dropout_perf.to_string())

    # Example 3: How do features affect the run? (for your ablation study)
    if 'feat_proc_params.use_cyclical_dates' in df.columns:
        logger.info("\n--- Ablation: Cyclical Dates ---")
        ablation_perf = df.groupby('feat_proc_params.use_cyclical_dates')['cv_f1'].mean()
        logger.info(ablation_perf.to_string())


if __name__ == "__main__":
    main()