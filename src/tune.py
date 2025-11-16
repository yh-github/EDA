import json
import logging
from dataclasses import asdict
from pathlib import Path

import diskcache
import pandas as pd
from sklearn.model_selection import ParameterGrid
from log_utils import setup_logging
from runner import ExpRunner
from config import ExperimentConfig, FieldConfig, EmbModel
from embedder import EmbeddingService
from feature_processor import FeatProcParams
from classifier import HybridModel

CACHE_DIR = Path('cache/results/')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Setup Logging ---
setup_logging(Path('logs/'), "tuning")
logger = logging.getLogger(__name__)


def get_param_key(param_set: dict) -> str:
    """
    Creates a unique, consistent string key for a parameter set.
    'param_set' is a dict of dataclasses, e.g.,
    {'exp_params': ExperimentConfig(...), 'emb_params': ...}
    """
    params_as_dict = {key: asdict(value) for key, value in param_set.items()}
    return json.dumps(params_as_dict, sort_keys=True)


def main():
    logger.info("=" * 50)
    logger.info("--- STARTING HYPERPARAMETER TUNING (V3-Structured) ---")
    logger.info("=" * 50)

    # --- 2. Define Data Path ---
    DATA_PATH = Path('data/rec_data2.csv')

    # --- 3. Define Structured Parameter Grids ---

    # 3a. Define "sub-grids" for each config object
    exp_config_grid = {
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [128, 256]
    }

    emb_params_grid = {
        'model_name': [EmbModel.ALBERT, EmbModel.MiniLM_L12, EmbModel.FINBERT]
    }

    feat_proc_params_grid = {
        'k_top': [20, 50, 100],
        'n_bins': [20, 50, 100]
    }

    model_params_grid = {
        'mlp_hidden_layers': [[64], [128, 64]],
        'dropout_rate': [0.25, 0.4]
    }

    # 3b. Use sub-grids to create lists of *config objects*
    # This is "fail-fast": it will error here if names are wrong
    base_exp_config = ExperimentConfig()  # For default values
    exp_configs = [
        ExperimentConfig(
            **{**asdict(base_exp_config), **p}  # Merge defaults with grid params
        ) for p in ParameterGrid(exp_config_grid)
    ]

    emb_params_list = [EmbeddingService.Params(**p) for p in ParameterGrid(emb_params_grid)]
    feat_proc_params_list = [FeatProcParams(**p) for p in ParameterGrid(feat_proc_params_grid)]
    model_params_list = [HybridModel.MlpHyperParams(**p) for p in ParameterGrid(model_params_grid)]

    # 3c. Create the "master grid" from the lists of objects
    master_grid_config = {
        'exp_params': exp_configs,
        'emb_params': emb_params_list,
        'feat_proc_params': feat_proc_params_list,
        'model_params': model_params_list
    }

    grid_list = list(ParameterGrid(master_grid_config))
    num_combinations = len(grid_list)

    logger.info(f"Defined structured parameter grid. Total combinations to test: {num_combinations}")

    # --- 4. Load Data and Base Configs ---
    try:
        full_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        logger.error(f"FATAL: Failed to load data from {DATA_PATH}. Error: {e}")
        return

    field_config = FieldConfig()

    df_cleaned = full_df.dropna(
        subset=[
            field_config.date,
            field_config.amount,
            field_config.text,
            field_config.label
        ]
    )
    logger.info(f"Loaded {len(full_df)} rows, {len(df_cleaned)} after cleaning.")

    # --- 5. Create Train/Val/Test Split ---
    base_runner = ExpRunner.create(
        exp_params=base_exp_config,
        full_df=df_cleaned,
        emb_params=emb_params_list[0],  # Just a placeholder
        feat_proc_params=FeatProcParams.all_off(),
        model_params=model_params_list[0]
    )

    logger.info("Creating Train/Val/Test split for tuning...")
    df_train, df_val, df_test = base_runner.create_train_val_test_split(
        test_size=0.2, val_size=0.2
    )

    df_train_val = pd.concat([df_train, df_val])

    logger.info(f"Tuning will be performed on {len(df_train_val)} rows.")
    logger.info(f"Final holdout test set has {len(df_test)} rows.")

    # --- 6. Run Tuning Loop ---
    # --- NEW: Initialize DiskCache ---
    logger.info(f"Initializing results cache at {CACHE_DIR}")
    # Using 'with' ensures the cache is properly closed
    with diskcache.Cache(str(CACHE_DIR)) as results_cache:

        n_splits = 3
        runs_to_skip = 0

        # --- NEW: Pre-scan grid to count skips ---
        keys_to_run = {}  # Use a dict to map index to key
        for i, param_set in enumerate(grid_list):
            param_key_str = get_param_key(param_set)
            if param_key_str in results_cache:
                runs_to_skip += 1
            else:
                keys_to_run[i] = param_key_str

        logger.info(f"Found {len(results_cache)} completed runs in cache.")
        logger.info(f"Will run {len(keys_to_run)} new combinations, skipping {runs_to_skip}.")

        # --- NEW: Iterate over the pre-filtered keys_to_run ---
        run_counter = 0
        total_new_runs = len(keys_to_run)

        for i, param_key_str in keys_to_run.items():
            run_counter += 1
            param_set = grid_list[i]  # Get the original param_set

            logger.info("-" * 50)
            logger.info(f"--- Run {run_counter}/{total_new_runs} (Grid Index {i + 1}/{num_combinations}) [RUNNING] ---")
            logger.info(f"Testing params:")
            logger.info(f"  exp_params: {param_set['exp_params']}")
            logger.info(f"  emb_params: {param_set['emb_params']}")
            logger.info(f"  feat_proc_params: {param_set['feat_proc_params']}")
            logger.info(f"  model_params: {param_set['model_params']}")

            try:
                runner = ExpRunner.create(
                    exp_params=param_set['exp_params'],
                    full_df=df_cleaned,
                    emb_params=param_set['emb_params'],
                    feat_proc_params=param_set['feat_proc_params'],
                    model_params=param_set['model_params'],
                    field_config=field_config
                )
                cv_metrics = runner.run_cross_validation(
                    df_train_val=df_train_val,
                    n_splits=n_splits
                )

                params_as_dict = {key: asdict(value) for key, value in param_set.items()}
                run_result = {'params': params_as_dict, 'metrics': cv_metrics}

                # --- NEW: Store result in cache ---
                results_cache[param_key_str] = run_result

                logger.info(f"Run {run_counter}/{total_new_runs} complete. Avg F1: {cv_metrics.get('cv_f1', 0.0):.4f}")

            except Exception as e:
                logger.error(f"Run {run_counter}/{total_new_runs} FAILED with params {param_set}. Error: {e}",
                             exc_info=True)
                error_result = {
                    'params': {key: asdict(value) for key, value in param_set.items()},
                    'metrics': {'cv_f1': 0.0, 'error': str(e)}
                }

                # --- NEW: Store error result in cache ---
                results_cache[param_key_str] = error_result

        # --- 7. Find and Report Best Parameters ---
        logger.info("=" * 50)
        logger.info("--- TUNING COMPLETE ---")

        # --- NEW: Read all results from the cache for summary ---
        all_results = []
        for result in results_cache.iterall():
            all_results.append(result)

        if not all_results:
            logger.warning("No results were found in the cache.")
            return

        # Convert to DataFrame for easy analysis
        all_results_df = pd.DataFrame(all_results)
        metrics_df = pd.json_normalize(all_results_df['metrics'])
        params_df = pd.json_normalize(all_results_df['params'])
        summary_df = pd.concat([params_df, metrics_df], axis=1)

        summary_df = summary_df.fillna(0)  # Handle errors

        if 'cv_f1' not in summary_df.columns:
            logger.error("No 'cv_f1' metric found in results. Cannot determine best run.")
            logger.info("Available columns:", summary_df.columns)
            return

        best_run = summary_df.sort_values(by='cv_f1', ascending=False).iloc[0]

        logger.info("Best parameters found (from all runs in cache):")
        logger.info(f"  Avg F1: {best_run['cv_f1']:.4f}")
        logger.info(f"  Avg ROC-AUC: {best_run['cv_roc_auc']:.4f}")
        logger.info("  Params:")

        # Find the original params dict from the best run index
        best_params_dict = all_results_df.iloc[best_run.name]['params']
        logger.info(json.dumps(best_params_dict, indent=2))
        logger.info("=" * 50)


if __name__ == "__main__":
    main()