from tft.tft_experiment import TFTTuningExperiment
from tft.tft_data_clustered import build_clustered_tft_dataset, prepare_clustered_tft_data

# --- CONFIG ---
MAX_ENCODER_LEN = 64
BATCH_SIZE = 2048
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_amt_clusters_v1"
BEST_MODEL_PATH = "cache/tft_models/best_tune2_model.pt"

if __name__ == "__main__":
    # Define custom search space
    search_space = {
        "hidden_size": lambda trial: trial.suggest_categorical("hidden_size", [64, 128, 256])
    }

    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS,
        search_space=search_space,
        # Inject Clustered Strategies
        prepare_data_fn=prepare_clustered_tft_data,
        build_dataset_fn=build_clustered_tft_dataset,
        use_aggregation=True
    )
    experiment.run()