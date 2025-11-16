import logging
import time
from typing import Self, Generator, Any
import pandas as pd
from numpy import floating
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from classifier import HybridModel
from config import *
from config import EmbModel
from data import TransactionDataset, FeatureSet, TrainingSample
from embedder import EmbeddingService
from feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureMetadata, FeatureHyperParams
from trainer import train_model, evaluate_model

logger = logging.getLogger(__name__)


def r(x: float| floating[Any]) -> float:
    return round(float(x), 3)

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
            model_params=other.model_params,
            field_config=other.field_config
        )

    @staticmethod
    def create(
        exp_params: ExperimentConfig,
        full_df:pd.DataFrame,
        emb_params:EmbeddingService.Params,
        feat_proc_params:FeatProcParams,
        model_params:HybridModel.MlpHyperParams,
        field_config:FieldConfig = FieldConfig()
    ):
        return ExpRunner(exp_params, full_df, emb_params, feat_proc_params, model_params, field_config)


    def __init__(self,
        exp_params: ExperimentConfig,
        full_df: pd.DataFrame,
        emb_params: EmbeddingService.Params,
        feat_proc_params: FeatProcParams,
        model_params: HybridModel.MlpHyperParams,
        field_config: FieldConfig = FieldConfig()
    ):
        self.exp_params = exp_params
        self.full_df = full_df
        self.emb_params = emb_params
        self.emb_proc_params = emb_params
        self.feat_proc_params = feat_proc_params
        self.model_params = model_params
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
                yield 1.0, full_train_df
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

    def build_data(
        self,
        df_train_frac: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> tuple[FeatureSet, FeatureSet, HybridFeatureProcessor, FeatureMetadata]:
        """
        Builds all feature sets (text, continuous, categorical) for the model.
        This method is now driven entirely by the FeatureMetadata returned
        from the processor.
        """

        # --- 1. Get Text Embeddings and Labels ---
        embedder = self.get_embedder(self.emb_params)
        y_train = df_train_frac[self.field_config.label].values
        y_test = df_test[self.field_config.label].values

        logger.info(f"Embedding {len(df_train_frac)} train texts...")
        X_text_train = embedder.embed(df_train_frac[self.field_config.text].tolist())

        logger.info(f"Embedding {len(df_test)} test texts...")
        X_text_test = embedder.embed(df_test[self.field_config.text].tolist())

        # --- 2. Fit and Transform Date/Amount Features ---
        processor = HybridFeatureProcessor.create(self.feat_proc_params, self.field_config)

        # Fit on train data and get the metadata
        metadata = processor.fit(df_train_frac)

        # Transform both train and test
        train_features_df = processor.transform(df_train_frac)
        test_features_df = processor.transform(df_test)

        # --- 3. Get Feature Lists from Metadata ---
        cyclical_cols = metadata.cyclical_cols
        continuous_scalable_cols = metadata.continuous_scalable_cols
        categorical_cols = list(metadata.categorical_features.keys())

        # --- 4. Build Continuous Features Array (X_continuous) ---
        # This list will hold the arrays to be concatenated
        train_cont_arrays_to_stack = []
        test_cont_arrays_to_stack = []

        # Add cyclical (unscaled) features if they exist
        if cyclical_cols:
            train_cont_arrays_to_stack.append(train_features_df[cyclical_cols].values)
            test_cont_arrays_to_stack.append(test_features_df[cyclical_cols].values)

        # Add scaled continuous features if they exist
        if continuous_scalable_cols:
            scaler = StandardScaler()
            X_cont_scaled_train = scaler.fit_transform(train_features_df[continuous_scalable_cols])
            X_cont_scaled_test = scaler.transform(test_features_df[continuous_scalable_cols])

            train_cont_arrays_to_stack.append(X_cont_scaled_train)
            test_cont_arrays_to_stack.append(X_cont_scaled_test)

        # Concatenate all continuous features
        if train_cont_arrays_to_stack:
            X_cont_train = np.concatenate(train_cont_arrays_to_stack, axis=1)
            X_cont_test = np.concatenate(test_cont_arrays_to_stack, axis=1)
        else:
            # Handle edge case where no continuous/cyclical features are made
            X_cont_train = np.empty((len(df_train_frac), 0))
            X_cont_test = np.empty((len(df_test), 0))

        # --- 5. Build Categorical Features Array (X_categorical) ---
        # This works even if categorical_cols is empty (creates an [N, 0] array)
        X_cat_train = train_features_df[categorical_cols].values
        X_cat_test = test_features_df[categorical_cols].values

        # --- 6. Package into FeatureSet Dataclasses ---
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

        return train_feature_set, test_feature_set, processor, metadata

    def run_experiment(
        self,
       train_features: FeatureSet,
       test_features: FeatureSet,
       metadata: FeatureMetadata
    ):
        # --- Hyperparameters (tune) ---
        DEVICE = get_device()
        NUM_EPOCHS = self.exp_params.epochs
        BATCH_SIZE = self.exp_params.batch_size
        LEARNING_RATE = self.exp_params.learning_rate
        patience = self.exp_params.early_stopping_patience
        patience_counter = 0

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
        model_config = FeatureHyperParams.build(train_features, metadata)

        model = HybridModel(
            feature_config=model_config,
            mlp_config=self.model_params
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
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best F1: {best_f1:.4f}")
                break

        logger.info("Training complete.")

        # --- 6. Return Metrics (unchanged) ---
        final_metrics_rounded = {k: r(v) for k, v in final_metrics.items()}
        return {
            **final_metrics_rounded,
            "embedder.model_name": str(self.emb_params.model_name)
        }

    def run_training_set_size(self, fractions: list[float]) -> dict[int, dict]:
        df_train, df_test = self.split_data_by_group()

        results = {}
        for frac, sub_train_df in self.create_learning_curve_splits(df_train, fractions):
            train_feature_set, test_feature_set, processor, meta = self.build_data(sub_train_df, df_test)
            res = self.run_experiment(train_feature_set, test_feature_set, meta)
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

    def create_train_val_test_split(
            self, test_size: float = 0.2, val_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data into three sets: train, validation, and a holdout test set.
        Ensures group integrity (accountId) across all splits.
        """
        full_df: pd.DataFrame = self.full_df.copy()
        field_config: FieldConfig = self.field_config
        random_state: int = self.exp_params.random_state

        logger.info(f"Splitting {len(full_df)} rows into Train/Val/Test...")

        # --- 1. Split off the Holdout Test set (e.g., 20%) ---
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_val_idx, test_idx = next(gss_test.split(
            full_df, y=full_df[field_config.label], groups=full_df[field_config.accountId]
        ))

        df_train_val = full_df.iloc[train_val_idx]
        df_test_holdout = full_df.iloc[test_idx]

        # --- 2. Split the Train/Val set into Train and Validation ---
        # We need to adjust the val_size relative to the remaining data
        # e.g., if val_size=0.2 and test_size=0.2, we want 20% of original,
        # which is 0.2 / (1.0 - 0.2) = 0.25 of the remaining df_train_val.

        relative_val_size = val_size / (1.0 - test_size)

        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_size, random_state=random_state
        )
        train_idx, val_idx = next(gss_val.split(
            df_train_val, y=df_train_val[field_config.label], groups=df_train_val[field_config.accountId]
        ))

        df_train = df_train_val.iloc[train_idx]
        df_val = df_train_val.iloc[val_idx]

        # --- 3. Sanity Checks ---
        train_acc = set(df_train[field_config.accountId].unique())
        val_acc = set(df_val[field_config.accountId].unique())
        test_acc = set(df_test_holdout[field_config.accountId].unique())

        assert train_acc.isdisjoint(val_acc), "Leakage: Train/Val account overlap"
        assert train_acc.isdisjoint(test_acc), "Leakage: Train/Test account overlap"
        assert val_acc.isdisjoint(test_acc), "Leakage: Val/Test account overlap"

        logger.info("Split complete. No account overlap.")
        logger.info(f"  Train:   {len(df_train)} rows, {len(train_acc)} accounts")
        logger.info(f"  Val:     {len(df_val)} rows, {len(val_acc)} accounts")
        logger.info(f"  Test:    {len(df_test_holdout)} rows, {len(test_acc)} accounts")

        return df_train, df_val, df_test_holdout

    def run_cross_validation(
            self, df_train_val: pd.DataFrame, n_splits: int = 5
    ) -> dict[str, float]:
        """
        Performs k-fold cross-validation on the provided df_train_val.
        Returns the averaged metrics.
        """
        field_config: FieldConfig = self.field_config
        random_state: int = self.exp_params.random_state

        gss_cv = GroupShuffleSplit(
            n_splits=n_splits, test_size=1.0 / n_splits, random_state=random_state
        )

        all_metrics = []

        logger.info(f"Starting {n_splits}-Fold Cross-Validation...")

        for i, (train_idx, val_idx) in enumerate(gss_cv.split(
                df_train_val, y=df_train_val[field_config.label], groups=df_train_val[field_config.accountId]
        )):
            logger.info(f"--- Fold {i + 1}/{n_splits} ---")
            df_train_fold = df_train_val.iloc[train_idx]
            df_val_fold = df_train_val.iloc[val_idx]

            # --- Run the standard pipeline ---
            train_fs, val_fs, processor, meta = self.build_data(
                df_train_fold, df_val_fold
            )

            # Use run_experiment_pytorch, which trains and evaluates
            metrics = self.run_experiment(train_fs, val_fs, meta)

            logger.info(f"Fold {i + 1} Metrics: f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}")
            all_metrics.append(metrics)

        # --- Average the metrics ---
        avg_metrics = {
            'cv_f1': r(np.mean([m['f1'] for m in all_metrics])),
            'cv_roc_auc': r(np.mean([m['roc_auc'] for m in all_metrics])),
            'cv_loss': r(np.mean([m['loss'] for m in all_metrics])),
            'cv_f1_std': r(np.std([m['f1'] for m in all_metrics])),
        }

        logger.info("Cross-Validation complete.")
        logger.info(f"Average F1: {avg_metrics['cv_f1']:.4f} +/- {avg_metrics['cv_f1_std']:.4f}")
        logger.info(f"Average ROC-AUC: {avg_metrics['cv_roc_auc']:.4f}")

        return avg_metrics
