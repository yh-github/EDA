import argparse
import optuna
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("inspector")


def inspect_study(db_path, study_name=None):
    storage_url = f"sqlite:///{db_path}"

    # List all studies if name not provided
    if not study_name:
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        except Exception as e:
            print(f"Error reading DB at {db_path}: {e}")
            return

        print(f"Found {len(summaries)} studies in {db_path}:")
        for s in summaries:
            best_val = s.best_trial.value if s.best_trial else "N/A"
            print(f"  - {s.study_name} (Trials: {s.n_trials}, Best: {best_val})")

        if not summaries:
            return

        # Pick the most recent one by default
        study_name = summaries[-1].study_name
        print(f"\nDefaulting to most recent study: {study_name}")

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"\n=== Study: {study_name} ===")
    try:
        print(f"Best Value: {study.best_value}")
        print(f"Best Params: {study.best_params}")
    except ValueError:
        print("No completed trials yet.")

    print("\n--- Trial History (Top 20) ---")
    df = study.trials_dataframe()

    if df.empty:
        print("No trials found.")
        return

    # Clean up columns for display
    cols = ['number', 'value', 'state', 'datetime_start', 'duration']
    param_cols = [c for c in df.columns if c.startswith('params_')]

    # Ensure columns exist
    existing_cols = [c for c in cols if c in df.columns]

    df_show = df[existing_cols + param_cols].sort_values('value', ascending=False).head(20)

    print(df_show.to_string(index=False))

    print("\n--- Pruned Trials ---")
    pruned = df[df['state'] == 'PRUNED']
    if not pruned.empty:
        print(f"Found {len(pruned)} pruned trials.")
        print(pruned[existing_cols + param_cols].head(10).to_string(index=False))
    else:
        print("No pruned trials found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Optuna DB")
    parser.add_argument("db_path", help="Path to tuning.db")
    parser.add_argument("--study", help="Specific study name to inspect", default=None)

    args = parser.parse_args()
    inspect_study(args.db_path, args.study)
