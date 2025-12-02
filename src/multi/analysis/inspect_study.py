import argparse
import optuna
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("inspector")


def inspect_study(db_path, study_name=None, h=2):
    storage_url = f"sqlite:///{db_path}"

    # List all studies if name not provided
    if not study_name:
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        except Exception as e:
            print(f"Error reading DB at {db_path}: {e}")
            return

        skip = []
        ok = []
        now = datetime.now()

        for s in summaries:
            # Check for stale "running" studies
            # If a study has no trials but is older than h hours, it's likely abandoned/stale
            in_grace = s.datetime_start and (now - s.datetime_start) < timedelta(hours=h)
            if (not s.n_trials or not s.best_trial) and not in_grace:
                skip.append(s.study_name)
                continue

            if s.best_trial:
                val_str = f"{s.best_trial.value:.4f}"
            else:
                val_str = "Running/Pending"
            ok.append(f"  - {s.study_name:<30} (Trials: {s.n_trials:<3}, Best: {val_str})")

        print(f"Found {len(ok)} active/completed studies in {db_path}:")
        for x in ok:
            print(x)

        if skip:
            print(f"\n[Skipped {len(skip)} empty/stale studies older than {h} hours]")

        if not summaries:
            return

        # Pick the most recent one by default
        # Filter summaries to find the last valid one
        valid_summaries = [s for s in summaries if s.study_name not in skip]
        if valid_summaries:
            # Sort by start time just in case, though usually returned in order
            valid_summaries.sort(key=lambda z: z.datetime_start if z.datetime_start else datetime.min)
            study_name = valid_summaries[-1].study_name
            print(f"\nDefaulting to most recent valid study: {study_name}")
        else:
            print("\nNo valid recent studies found.")
            return

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
    parser.add_argument("--db_path", help="Path to tuning.db", default="checkpoints/multi/tuning.db")
    parser.add_argument("--study", help="Specific study name to inspect", default=None)
    parser.add_argument("--grace", type=float, help="Hours grace period", default=2)

    args = parser.parse_args()
    inspect_study(args.db_path, args.study, args.grace)