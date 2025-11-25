from tft.tft_experiment import TFTTuningExperiment

# --- CONFIG ---
MAX_ENCODER_LEN = 150
BATCH_SIZE = 2048
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_optimization_outgoing"
BEST_MODEL_PATH = "cache/tft_models/best_tune1.1_model.pt"

if __name__ == "__main__":
    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS
        # Uses default prepare_tft_data and build_tft_dataset
    )
    experiment.run()