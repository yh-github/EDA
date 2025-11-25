from common.exp_utils import get_cli_args
from tft.tft_experiment import TFTTuningExperiment

# --- CONFIG ---
MAX_ENCODER_LEN = 150
BATCH_SIZE = 2048
MAX_EPOCHS = 10
N_TRIALS = 20
STUDY_NAME = "tft_optimization_outgoing"
BEST_MODEL_PATH = "cache/tft_models/best_tune1.1_model.pt"

if __name__ == "__main__":
    min_encoder_len = int(get_cli_args().get(1, '10'))

    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        min_encoder_len=min_encoder_len,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS
        # Uses default prepare_tft_data and build_tft_dataset
    )
    experiment.run()