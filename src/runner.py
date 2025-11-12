import time
from typing import Self, Generator

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.utils.data import DataLoader

from classifier import HybridModel
from data import TransactionDataset, FeatureSet, TrainingSample
from feature_processor import HybridFeatureProcessor, check_unknown_rate, FeatProcParams
from config import *
from embedder import EmbeddingService
import logging
from config import EmbModel
from trainer import train_model, evaluate_model

logger = logging.getLogger(__name__)


def r(x: float):
    return round(x, 3)

@dataclass(frozen=True)
class ExpData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

class ExpRunner:

    @classmethod
    def copy(cls, other:Self) -> Self:
        return ExpRunner(
            exp_params=other.exp_params,
            full_df=other.full_df,
            emb_params=other.emb_params,
            feat_proc_params=other.feat_proc_params,
            field_config=other.field_config
        )

    @staticmethod
    def create(
        exp_params: ExperimentConfig,
        full_df:pd.DataFrame,
        emb_params:EmbeddingService.Params,
        feat_proc_params:FeatProcParams,
        field_config:FieldConfig = FieldConfig()
    ):
        return ExpRunner(exp_params, full_df, emb_params, feat_proc_params, field_config)


    def __init__(self,
        exp_params: ExperimentConfig,
        full_df: pd.DataFrame,
        emb_params: EmbeddingService.Params,
        feat_proc_params: FeatProcParams,
        field_config: FieldConfig = FieldConfig()
    ):
        self.exp_params = exp_params
        self.full_df = full_df
        self.emb_params = emb_params
        self.emb_proc_params = emb_params
        self.feat_proc_params = feat_proc_params
        self.field_config = field_config
        self.embedder_map: dict[EmbModel, EmbeddingService] = {}

    def get_embedder(self, model_params:EmbeddingService.Params) -> EmbeddingService:
        model_name = model_params.model_name
        if model_name not in self.embedder_map:
            logger.info(f"Creating new EmbeddingService(model_name={model_name})")
            self.embedder_map[model_name] = EmbeddingService.create(model_params)
        return self.embedder_map[model_name]

    def split_data_by_group(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into train and test sets, ensuring that all rows
        for a given accountId stay in the *same* set to prevent data leakage.

        This split is 100% deterministic and consistent if the
        input_df and random_state are the same.
        """

        full_df: pd.DataFrame = self.full_df.copy()
        field_config: FieldConfig = self.field_config
        test_size: float = self.exp_params.test_size
        random_state: int = self.exp_params.random_state

        logger.info(f"Splitting {len(full_df)} rows by group '{field_config.accountId}'...")

        # Define our grouper
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        # Get the indices for the split
        try:
            train_idx, test_idx = next(gss.split(
                full_df,
                y=full_df[field_config.label], # stratified
                groups=full_df[field_config.accountId]  # crucial
            ))
        except ValueError as e:
            logger.error(f"ERROR: Cannot split data. This often happens if 'n_splits' "
                         f"is greater than the number of unique groups. Error: {e}")
            raise

        train_df = full_df.iloc[train_idx]
        test_df = full_df.iloc[test_idx]

        # --- Critical Sanity Check for Data Leakage ---
        train_accounts = set(train_df[field_config.accountId].unique())
        test_accounts = set(test_df[field_config.accountId].unique())
        overlap = train_accounts.intersection(test_accounts)

        logger.info(f"Train accounts: {len(train_accounts)}, Test accounts: {len(test_accounts)}")
        if not overlap:
            logger.info("SUCCESS: No account overlap between train and test sets.")
        else:
            # This should theoretically never happen with GroupShuffleSplit
            logger.error(f"FATAL LEAKAGE: {len(overlap)} accounts are in BOTH sets.")
            raise AssertionError("Data leakage detected! Account overlap in train/test.")

        def df_stats(df:pd.DataFrame) -> str:
            return f"len={len(df)} accounts={df[self.field_config.accountId].nunique()}"

        logger.info(f"Split complete. Train: {df_stats(train_df)}, Test: {df_stats(test_df)}.")

        return train_df, test_df

    def create_learning_curve_splits(
        self,
        full_train_df: pd.DataFrame,
        fractions: list[float],
    ) -> Generator[tuple[float, pd.DataFrame], None, None]:
        """
        Takes the *full* training set and yields smaller, fractional subsets
        for creating a learning curve.

        This split is also done by 'accountId' to ensure a valid experiment,
        comparing (e.g.) 25% of *accounts* vs. 50% of *accounts*.
        """

        logger.info(f"Preparing to create {len(fractions)} training set fractions...")

        unique_accounts = full_train_df[self.field_config.accountId].unique()
        total_accounts = len(unique_accounts)

        # Shuffle the accounts once to ensure random, *nested* sampling
        # (i.e., the 50% set will contain the 25% set)
        rng = np.random.RandomState(self.exp_params.random_state)
        shuffled_accounts = rng.permutation(unique_accounts)

        for frac in sorted(fractions):
            if frac <= 0.0:
                continue
            if frac >= 1.0:
                logger.info(f"Yielding {frac * 100:.0f}% split: {total_accounts} accounts, {len(full_train_df)} rows")
                yield (1.0, full_train_df)
                continue

            n_accounts_to_take = int(total_accounts * frac)
            if n_accounts_to_take == 0:
                logger.warning(f"Fraction {frac} resulted in 0 accounts. Skipping.")
                continue

            # Take the *first n* accounts from the shuffled list
            accounts_sample = shuffled_accounts[:n_accounts_to_take]

            # Use .isin() to select all rows belonging to these accounts
            sub_df = full_train_df[
                full_train_df[self.field_config.accountId].isin(accounts_sample)
            ].copy()

            logger.info(f"Yielding {frac * 100:.0f}% split: {n_accounts_to_take} accounts, {len(sub_df)} rows")
            yield frac, sub_df

    # In ExpRunner class (in src/runner.py)

    def build_data_for_pytorch(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[
        FeatureSet, FeatureSet, HybridFeatureProcessor]:
        embedder = self.get_embedder(self.emb_params)

        y_train = df_train[self.field_config.label].values
        y_test = df_test[self.field_config.label].values

        # --- 1. Text Features ---
        X_text_train = embedder.embed(df_train[self.field_config.text].tolist())
        X_text_test = embedder.embed(df_test[self.field_config.text].tolist())

        # --- 2. Numerical/Date Features ---
        processor = HybridFeatureProcessor.create(self.feat_proc_params, self.field_config)
        processor.fit(df_train)

        train_features_df = processor.transform(df_train)
        test_features_df = processor.transform(df_test)

        # --- 3. Separate feature groups ---

        # Define which columns go where
        continuous_cols = [
            'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos',
            'day_of_14_cycle_sin', 'day_of_14_cycle_cos', 'is_positive', 'log_abs_amount'
        ]

        categorical_cols = [
            'day_of_week_id', 'day_of_month_id', 'amount_token_id'
        ]

        # Filter columns that were actually created (based on config)
        continuous_cols = [c for c in continuous_cols if c in train_features_df.columns]
        categorical_cols = [c for c in categorical_cols if c in train_features_df.columns]

        X_cont_train = train_features_df[continuous_cols].values
        X_cont_test = test_features_df[continuous_cols].values

        X_cat_train = train_features_df[categorical_cols].values
        X_cat_test = test_features_df[categorical_cols].values

        # --- 4. Package for output ---
        train_feature_set = FeatureSet(
            X_text=X_text_train,
            X_continuous=X_cont_train,
            X_categorical=X_cat_train,
            y=y_train
        )

        test_feature_set = FeatureSet(
            X_text=X_text_test,
            X_continuous=X_cont_test,
            X_categorical=X_cat_test,
            y=y_test
        )

        return train_feature_set, test_feature_set, processor

    def build_data(self, df_train:pd.DataFrame, df_test:pd.DataFrame) -> ExpData:
        embedder = self.get_embedder(self.emb_params)
        logger.info(f"{embedder.model_name = }")

        y_train = df_train[self.field_config.label]
        y_test = df_test[self.field_config.label]

        logger.info(f"Total data: {len(self.full_df)}, Train: {len(df_train)}, Test: {len(df_test)}")
        logger.info(f"positive class %: train={y_train.mean() * 100:.2f}%  test={y_test.mean() * 100:.2f}%")

        # --- Create Text Features (Cached) ---
        logger.info("\nProcessing text features (using EmbeddingService)...")
        logger.info(f"Embedding {len(df_train)} train texts...")
        train_text_features_np = embedder.embed(df_train[self.field_config.text].tolist())

        logger.info(f"Embedding {len(df_test)} test texts...")
        test_text_features_np = embedder.embed(df_test[self.field_config.text].tolist())

        # --- Create Numerical Features ---
        if self.feat_proc_params.is_nop():
            X_train = train_text_features_np
            X_test = test_text_features_np
        else:
            logger.info("\nProcessing numerical/date features...")
            processor = HybridFeatureProcessor.create(self.feat_proc_params)

            processor.fit(df_train)

            train_num_features_df = processor.transform(df_train)
            test_num_features_df = processor.transform(df_test)

            # --- Health Check (Go/No-Go) ---
            logger.info("\n--- Health Check on Processor ---")
            report = check_unknown_rate(processor, test_num_features_df, "Test Set")
            test_unknown_pct = report.get('percent', 100.0)

            if test_unknown_pct > self.exp_params.go_no_go_threshold_pct:
                raise Exception(f"**NO-GO!** Test [UNKNOWN] rate is {test_unknown_pct:.2f}%. Halting experiment.")
            else:
                logger.info(f"**GO!** Test [UNKNOWN] rate is {test_unknown_pct:.2f}%, which is acceptable.")

            # --- Concatenate ---
            logger.info("\nConcatenating all features...")
            X_train = np.concatenate([train_text_features_np, train_num_features_df.values], axis=1)
            X_test = np.concatenate([test_text_features_np, test_num_features_df.values], axis=1)

        logger.info(f"Total feature space size: {X_train.shape[1]} features")
        return ExpData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

    def run(self, fractions: list[float]) -> dict[int, dict]:
        df_train, df_test = self.split_data_by_group()
        results = {}
        for frac, sub_train_df in self.create_learning_curve_splits(df_train, fractions):
            exp_data = self.build_data(sub_train_df, df_test)
            res = self.run_experiment(exp_data)
            d = {
                **res,
                "train_frac": r(frac),
                "train_size": len(sub_train_df),
                "test_size": len(df_test),
                "train_accounts": sub_train_df[self.field_config.accountId].nunique(),
                "test_accounts": df_test[self.field_config.accountId].nunique()
            }
            logger.info(d)
            results[len(sub_train_df)] = d
        return results

    def run_experiment(self, data:ExpData):
        """
        Runs the full "Scenario 1" pipeline:
        1. Splits data
        2. Creates numerical features
        3. Creates "frozen" text embeddings
        4. Concatenates features
        5. Trains and scores a simple classifier
        """
        X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

        # --- 5. Train the "Learner" ---
        logger.info("Training the final, simple classifier (MLP)...")
        start_time = time.time()
        learner = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=self.exp_params.random_state,
            early_stopping=True
        )
        learner.fit(X_train, y_train)
        logger.info(f"Training complete in {time.time() - start_time:.2f} seconds.")

        # --- Score the Experiment ---
        y_pred = learner.predict(X_test)
        y_pred_proba = learner.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        res = {
            "f1": r(f1),
            "roc_auc": r(roc_auc),
            "accuracy": r(accuracy),
            "embedder.model_name": str(self.emb_params.model_name),
        }
        return res

    def run_experiment_pytorch(
        self,
       train_features: FeatureSet,
       test_features: FeatureSet,
       processor: HybridFeatureProcessor
    ):
        # --- Hyperparameters (tune) ---
        DEVICE = get_device()
        NUM_EPOCHS = 10
        BATCH_SIZE = 256
        LEARNING_RATE = 1e-3

        # --- Create DataLoaders ---
        train_dataset = TransactionDataset(train_features)
        test_dataset = TransactionDataset(test_features)

        def dataloader(dataset, shuffle:bool):
            return DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=TrainingSample.collate_fn
            )

        train_loader = dataloader(train_dataset, shuffle=True)
        test_loader = dataloader(test_dataset, shuffle=False)

        # --- Instantiate the Model ---

        # Get dimensions from the FeatureSet attributes
        model_config = processor.build_model_config(train_features)

        model = HybridModel(
            config=model_config,
            mlp_hidden_layers=[256, 128],
            dropout_rate=0.4
        ).to(DEVICE)

        # --- Setup Optimizer and Loss ---
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        logger.info(f"Starting PyTorch training on {DEVICE} for {NUM_EPOCHS} epochs...")

        # --- 5. Training Loop ---
        best_f1 = -1.0
        final_metrics = {}

        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = time.time()

            train_loss = train_model(model, train_loader, optimizer, criterion, DEVICE)
            metrics = evaluate_model(model, test_loader, criterion, DEVICE)

            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{NUM_EPOCHS} [{epoch_time:.2f}s] | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {metrics['loss']:.4f} | "
                f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                final_metrics = metrics

        logger.info("Training complete.")

        # --- 6. Return Metrics (unchanged) ---
        final_metrics_rounded = {k: r(v) for k, v in final_metrics.items()}
        return {
            **final_metrics_rounded,
            "embedder.model_name": str(self.emb_params.model_name)
        }

    def run_torch(self, fractions: list[float]) -> dict[int, dict]:
        df_train, df_test = self.split_data_by_group()

        results = {}
        for frac, sub_train_df in self.create_learning_curve_splits(df_train, fractions):
            train_feature_set, test_feature_set, processor = self.build_data_for_pytorch(sub_train_df, df_test)
            res = self.run_experiment_pytorch(train_feature_set, test_feature_set, processor)
            d = {
                **res,
                "train_frac": r(frac),
                "train_size": len(sub_train_df),
                "test_size": len(df_test),
                "train_accounts": sub_train_df[self.field_config.accountId].nunique(),
                "test_accounts": df_test[self.field_config.accountId].nunique()
            }
            logger.info(d)
            results[len(sub_train_df)] = d
        return results
