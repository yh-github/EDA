from common.exp_utils import get_cli_args
from common.feature_processor import FeatProcParams
from tft.tft_experiment import TFTTuningExperiment
# Re-use the non-overlapping data strategy from Tune 4
from experiments.tft_tune4 import prepare_non_overlapping_data
from tft.tft_data_clustered import build_clustered_tft_dataset

# --- CONFIG ---
# Keep encoder length consistent with the successful Tune 4
MAX_ENCODER_LEN = 64
BATCH_SIZE = 2048
# Increase epochs slightly as we are narrowing in on a stable region
MAX_EPOCHS = 25
N_TRIALS = 20
STUDY_NAME = "tft_tune5_refinement"
BEST_MODEL_PATH = "cache/tft_models/best_tune5_model.pt"

if __name__ == "__main__":
    # Refined Search Space based on Tune 4 Logs (Trial 10 Winner)
    search_space = {
        # Winner was 1.7e-4. Narrow range to [1e-4, 3e-4]
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True),

        # Winner was 256. 128 was clearly worse. Explore 256 vs 512.
        "hidden_size": lambda trial: trial.suggest_categorical("hidden_size", [256, 512]),

        # Winner was 0.28. Range [0.25, 0.35] covers top trials.
        "dropout": lambda trial: trial.suggest_float("dropout", 0.25, 0.35),

        # Winner used 2, but 4 was close.
        "attention_head_size": lambda trial: trial.suggest_categorical("attention_head_size", [2, 4]),

        # Winner was 0.53.
        "gradient_clip_val": lambda trial: trial.suggest_float("gradient_clip_val", 0.3, 0.7)
    }

    min_encoder_len = int(get_cli_args().get(1, '5'))

    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        min_encoder_len=min_encoder_len,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS,
        search_space=search_space,

        # Use the successful non-overlapping strategy
        prepare_data_fn=prepare_non_overlapping_data,

        # Use the clustered dataset builder
        build_dataset_fn=build_clustered_tft_dataset,

        use_aggregation=False,

        feat_params=FeatProcParams(
            use_is_positive=False, use_categorical_dates=True, use_cyclical_dates=True,
            use_continuous_amount=True, use_categorical_amount=False, k_top=0, n_bins=0,
            use_behavioral_features=True
        )
    )
    experiment.run()