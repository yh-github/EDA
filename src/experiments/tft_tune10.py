from common.config import EmbModel
from common.embedder import EmbeddingService
from common.exp_utils import get_cli_args
from common.feature_processor import FeatProcParams
from tft.tft_experiment import TFTTuningExperiment

# --- CONFIG ---
MAX_ENCODER_LEN = 150
BATCH_SIZE = 2048
MAX_EPOCHS = 10
N_TRIALS = 20
STUDY_NAME = "tft_base"
BEST_MODEL_PATH = "cache/tft_models/best_tune10_model.pt"

if __name__ == "__main__":
    min_encoder_len = int(get_cli_args().get(1, '10'))

    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        min_encoder_len=min_encoder_len,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS,
        data_path='data/combined_transactions_flat.csv',
        downsample=0.3,
        # Uses default prepare_tft_data and build_tft_dataset
        feat_params = FeatProcParams(
            use_is_positive=False, use_categorical_dates=True, use_cyclical_dates=True,
            use_continuous_amount=True, use_categorical_amount=False, k_top=0, n_bins=0,
            use_behavioral_features=True
        ),
        emb_params = EmbeddingService.Params(model_name=EmbModel.MPNET, max_length=32)

    )
    experiment.run()