import logging
import json
from pathlib import Path
from typing import Self, Any
import pandas as pd
import diskcache
from sklearn.model_selection import ParameterGrid
from dataclasses import asdict
from log_utils import setup_logging
from runner import ExpRunner, ModelParams
from config import ExperimentConfig, FieldConfig, EmbModel
from embedder import EmbeddingService
from feature_processor import FeatProcParams
from classifier import HybridModel

logger = logging.getLogger(__name__)
exclude_none_values = lambda x: {k: v for (k, v) in x if v is not None}

class HyperTuner:
    """
    Encapsulates all logic for running a hyperparameter tuning grid search.
    It handles data loading, splitting, caching, and running experiments.
    """

    @classmethod
    def load(cls,
        index: int, model_config_class: Any = HybridModel.MlpHyperParams, unique_cache=True, filter_direction=False
    ) ->Self:
        setup_logging(Path('logs/'), f"tuning{index}")
        postfix = ""
        if unique_cache:
            postfix = f"{index}/"
        return cls(
            data_path=Path('data/rec_data2.csv'),
            cache_dir=Path(f'cache/results/{postfix}'),
            filter_direction=filter_direction,
            field_config=FieldConfig(),
            model_config_class=model_config_class
        )

    def __init__(self,
         data_path: Path,
         cache_dir: Path,
         filter_direction: bool,
         model_config_class:ModelParams,
         field_config: FieldConfig = FieldConfig()
    ):
        """
        Initializes the Tuner by loading and splitting the data one time.

        :param data_path: Path to the full dataset (e.g., rec_data2.csv)
        :param cache_dir: Path to the diskcache directory for this experiment.
        :param field_config: A FieldConfig instance.
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.field_config = field_config
        self.model_config_class = model_config_class

        logger.info(f"HyperTuner initialized. Cache at: {self.cache_dir}")

        # Load and split data *once* during initialization
        self.filter_direction = filter_direction
        self._load_and_split_data()

    def _load_and_split_data(self):
        """Loads and splits the data into train_val/test sets."""
        logger.info(f"Loading data from {self.data_path}...")
        try:
            full_df = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"FATAL: Failed to load data. Error: {e}")
            raise

        df_cleaned = full_df.dropna(
            subset=[
                self.field_config.date,
                self.field_config.amount,
                self.field_config.text,
                self.field_config.label
            ]
        )
        if self.filter_direction:
            df_cleaned = df_cleaned[(df_cleaned[self.field_config.amount] * self.filter_direction) > 0]
        logger.info(f"Loaded {len(full_df)} rows, {len(df_cleaned)} after cleaning.")

        # We need a base runner just to access the split method
        base_runner = ExpRunner.create(
            exp_params=ExperimentConfig(),
            full_df=df_cleaned,
            emb_params=EmbeddingService.Params(model_name=EmbModel.ALBERT),
            feat_proc_params=FeatProcParams.all_off(),
            model_params=self.model_config_class
        )

        logger.info("Creating Train/Val/Test split for tuning...")
        df_train, df_val, df_test = base_runner.create_train_val_test_split(
            test_size=0.2, val_size=0.2
        )

        # Store the sets that the run() method will need
        self.df_train_val = pd.concat([df_train, df_val])
        self.df_cleaned = df_cleaned  # Needed by the ExpRunner
        logger.info(f"Tuning will be performed on {len(self.df_train_val)} rows.")

    def _materialize_grid(self, structured_grid_config: dict) -> list[dict]:
        """
        Converts the user-friendly structured grid config into a final
        list of parameter-set dictionaries.
        """
        # Default objects for merging with grid params
        base_exp_config = ExperimentConfig()

        # Maps the keys in the grid to their corresponding dataclass
        # and any default values (like from base_exp_config).
        config_class_map = {
            'exp_params': (ExperimentConfig, asdict(base_exp_config)),
            'emb_params': (EmbeddingService.Params, {}),
            'feat_proc_params': (FeatProcParams, {}),
            'model_params': (self.model_config_class, {})
        }

        materialized_grid = {}

        for key, sub_grid in structured_grid_config.items():
            if key not in config_class_map:
                logger.warning(f"Unknown grid key '{key}'. Skipping.")
                continue

            ConfigClass, defaults = config_class_map[key]

            # Create the list of config objects for this sub-grid
            config_object_list = [
                ConfigClass(
                    **{**defaults, **params}  # Merge defaults with grid params
                ) for params in ParameterGrid(sub_grid)
            ]
            materialized_grid[key] = config_object_list

        # Return the final list of all combinations
        return list(ParameterGrid(materialized_grid))

    def _get_param_key(self, param_set: dict) -> str:
        """Creates a unique, consistent string key for a parameter set."""
        params_as_dict = {
            key: asdict(value, dict_factory=exclude_none_values)
            for key, value in param_set.items()
        }
        return json.dumps(params_as_dict, sort_keys=True)

    def run(self, structured_grid_config: dict, n_splits: int = 3):
        """
        Runs the full tuning experiment for a given parameter grid.

        :param structured_grid_config: The dict defining the "sub-grids".
        :param n_splits: The number of CV splits to run.
        """
        logger.info(f"Starting tuning run. Processing grid...")

        grid_list = self._materialize_grid(structured_grid_config)
        num_combinations = len(grid_list)
        logger.info(f"Total combinations to test: {num_combinations}")

        with diskcache.Cache(str(self.cache_dir)) as results_cache:
            keys_to_run = {}
            runs_to_skip = 0

            # Pre-scan grid to find completed runs
            for i, param_set in enumerate(grid_list):
                param_key_str = self._get_param_key(param_set)
                if param_key_str in results_cache:
                    runs_to_skip += 1
                else:
                    keys_to_run[i] = param_key_str

            logger.info(f"Found {len(results_cache)} completed runs in cache.")
            logger.info(f"Will run {len(keys_to_run)} new combinations, skipping {runs_to_skip}.")

            run_counter = 0
            total_new_runs = len(keys_to_run)

            # Run the experiments
            for i, param_key_str in keys_to_run.items():
                run_counter += 1
                param_set = grid_list[i]  # Get the original param_set

                logger.info("-" * 50)
                logger.info(
                    f"--- Run {run_counter}/{total_new_runs} (Grid Index {i + 1}/{num_combinations}) [RUNNING] ---")
                logger.info(f"Testing params: {param_set}")

                try:
                    runner = ExpRunner.create(
                        exp_params=param_set['exp_params'],
                        full_df=self.df_cleaned,
                        emb_params=param_set['emb_params'],
                        feat_proc_params=param_set['feat_proc_params'],
                        model_params=param_set['model_params'],
                        field_config=self.field_config
                    )
                    cv_metrics = runner.run_cross_validation(
                        df_train_val=self.df_train_val,
                        n_splits=n_splits
                    )

                    params_as_dict = {key: asdict(value) for key, value in param_set.items()}
                    run_result = {'params': params_as_dict, 'metrics': cv_metrics}
                    results_cache[param_key_str] = run_result
                    logger.info(
                        f"Run {run_counter}/{total_new_runs} complete. Avg F1: {cv_metrics.get('cv_f1', 0.0):.4f}")

                except Exception as e:
                    logger.error(f"Run {run_counter}/{total_new_runs} FAILED. Error: {e}", exc_info=True)
                    error_result = {
                        'params': {key: asdict(value) for key, value in param_set.items()},
                        'metrics': {'cv_f1': 0.0, 'error': str(e)}
                    }
                    results_cache[param_key_str] = error_result

            # --- Final Report ---
            logger.info("=" * 50)
            logger.info("--- TUNING COMPLETE ---")

            all_results = [results_cache[k] for k in results_cache]

            if not all_results:
                logger.warning("No results were found in the cache.")
                return

            all_results_df = pd.DataFrame(all_results)
            metrics_df = pd.json_normalize(all_results_df['metrics'])
            params_df = pd.json_normalize(all_results_df['params'])
            summary_df = pd.concat([params_df, metrics_df], axis=1).fillna(0)

            if 'cv_f1' not in summary_df.columns:
                logger.error("No 'cv_f1' metric found. Cannot determine best run.")
                return

            best_run = summary_df.sort_values(by='cv_f1', ascending=False).iloc[0]

            logger.info("Best parameters found (from all runs in cache):")
            logger.info(f"  Avg F1: {best_run['cv_f1']:.4f}")
            logger.info(f"  Avg ROC-AUC: {best_run['cv_roc_auc']:.4f}")
            logger.info("  Params:")
            best_params_dict = all_results_df.iloc[best_run.name]['params']
            logger.info(json.dumps(best_params_dict, indent=2))
            logger.info("=" * 50)