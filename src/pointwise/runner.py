import logging
from pathlib import Path
from typing import Self, Generator, Any
import pandas as pd
from numpy import floating
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from common.config import *
from common.config import EmbModel
from common.data import TransactionDataset, FeatureSet, TrainingSample, create_train_val_test_split
from common.embedder import EmbeddingService
from common.exp_utils import set_global_seed
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureMetadata, FeatureHyperParams
from pointwise.classifier_transformer import TransformerHyperParams, TabularTransformerModel
from pointwise.classifier import HybridModel
from pointwise.trainer import PyTorchTrainer

logger = logging.getLogger(__name__)

ModelParams = HybridModel.MlpHyperParams | TransformerHyperParams


def r(x: float | floating[Any]) -> float:
    return round(float(x), 3)


@dataclass(frozen=True)
class ExpData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class ExpRunner:

    @classmethod
    def copy(cls, other: Self) -> Self:
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
            full_df: pd.DataFrame,
            emb_params: EmbeddingService.Params,
            feat_proc_params: FeatProcParams,
            model_params: ModelParams,
            field_config: FieldConfig = FieldConfig()
    ):
        set_global_seed(exp_params.random_state)
        return ExpRunner(exp_params, full_df, emb_params, feat_proc_params, model_params, field_config)

    def __init__(self,
                 exp_params: ExperimentConfig,
                 full_df: pd.DataFrame,
                 emb_params: EmbeddingService.Params,
                 feat_proc_params: FeatProcParams,
                 model_params: ModelParams,
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

    def get_embedder(self, model_params: EmbeddingService.Params) -> EmbeddingService:
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
                y=full_df[field_config.label],  # stratified
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

        def df_stats(df: pd.DataFrame) -> str:
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
        Standard 2-way build (Train/Test). Fits on Train.
        """
        return self._build_data_internal(df_train_frac, df_test, None)

    def build_data_three_way(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            df_test: pd.DataFrame
    ) -> tuple[FeatureSet, FeatureSet, FeatureSet, HybridFeatureProcessor, FeatureMetadata]:
        """
        3-way build (Train/Val/Test). Fits on Train, transforms all three.
        """
        train_fs, val_fs, processor, meta, test_fs = self._build_data_internal(df_train, df_val, df_test)
        return train_fs, val_fs, test_fs, processor, meta

    def _build_data_internal(self, df_train, df_val, df_test=None):
        """Shared logic for building feature sets."""

        # 1. Embeddings
        embedder = self.get_embedder(self.emb_params)

        def get_text_emb(df):
            return embedder.embed(df[self.field_config.text].tolist())

        X_text_train = get_text_emb(df_train)
        X_text_val = get_text_emb(df_val)
        X_text_test = get_text_emb(df_test) if df_test is not None else None

        # 2. Feature Processor
        processor = HybridFeatureProcessor.create(self.feat_proc_params, self.field_config)
        metadata = processor.fit(df_train)  # Fit ONLY on Train

        df_train_feats = processor.transform(df_train)
        df_val_feats = processor.transform(df_val)
        df_test_feats = processor.transform(df_test) if df_test is not None else None

        # 3. Scaling & Stacking
        cyclical_cols = metadata.cyclical_cols
        continuous_scalable_cols = metadata.continuous_scalable_cols
        categorical_cols = list(metadata.categorical_features.keys())

        scaler = StandardScaler()

        # Fit scaler ONLY on Train
        if continuous_scalable_cols:
            scaler.fit(df_train_feats[continuous_scalable_cols])

        def build_fs(df_original, df_feats, x_text):
            y = df_original[self.field_config.label].values

            # Continuous
            cont_parts = []
            if cyclical_cols:
                cont_parts.append(df_feats[cyclical_cols].values)
            if continuous_scalable_cols:
                cont_parts.append(scaler.transform(df_feats[continuous_scalable_cols]))

            if cont_parts:
                X_cont = np.concatenate(cont_parts, axis=1)
            else:
                X_cont = np.empty((len(df_original), 0))

            # Categorical
            X_cat = df_feats[categorical_cols].values

            return FeatureSet(X_text=x_text, X_continuous=X_cont, X_categorical=X_cat, y=y)

        train_fs = build_fs(df_train, df_train_feats, X_text_train)
        val_fs = build_fs(df_val, df_val_feats, X_text_val)

        if df_test is not None:
            test_fs = build_fs(df_test, df_test_feats, X_text_test)
            return train_fs, val_fs, processor, metadata, test_fs

        return train_fs, val_fs, processor, metadata

    @classmethod
    def _calculate_optimal_threshold_metrics(
            cls,
            model: nn.Module,
            data_loader: DataLoader,
            device: torch.device
    ) -> dict[str, Any]:
        """
        Evaluates the model on a dataset, calculates the Precision-Recall curve,
        and finds the threshold that maximizes the F1 score.

        Notice: If used on the test set, use ONLY to detect the NEED for threshold calibration
        """
        model.eval()
        all_logits = []
        all_y_true = []

        with torch.no_grad():
            for batch in data_loader:
                x_text = batch.x_text.to(device)
                x_continuous = batch.x_continuous.to(device)
                x_categorical = batch.x_categorical.to(device)
                y_true = batch.y.to(device)

                logits = model(x_text, x_continuous, x_categorical)
                all_logits.append(logits)
                all_y_true.append(y_true)

        np_logits = torch.cat(all_logits).cpu().numpy()
        np_y_true = torch.cat(all_y_true).cpu().numpy()
        np_probs = 1 / (1 + np.exp(-np_logits))  # Sigmoid

        # Calculate the raw PR curve
        precisions, recalls, thresholds = precision_recall_curve(np_y_true, np_probs)

        # Calculate F1 for every single possible threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        f1_scores = np.nan_to_num(f1_scores)
        best_idx = np.argmax(f1_scores)

        best_f1 = f1_scores[best_idx]
        # thresholds array is length N, prec/rec arrays are N+1.
        # If best_idx is the last element (F1 at max recall/prec), use standard 0.5 or last thresh
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        return {
            "val_best_f1": r(best_f1),
            "val_best_threshold": r(best_threshold),
            "pr_curve": {
                "precisions": precisions,
                "recalls": recalls,
                "thresholds": thresholds
            }
        }

    def _setup_training(self, train_features: FeatureSet, test_features: FeatureSet, metadata: FeatureMetadata):
        DEVICE = get_device()
        NUM_EPOCHS = self.exp_params.epochs
        BATCH_SIZE = self.exp_params.batch_size
        LEARNING_RATE = self.exp_params.learning_rate

        train_dataset = TransactionDataset(train_features)
        test_dataset = TransactionDataset(test_features)

        def dataloader(dataset, is_train: bool):
            use_gpu = DEVICE.type != "cpu"
            return DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=is_train,
                collate_fn=TrainingSample.collate_fn,
                num_workers=4,
                pin_memory=use_gpu,
                drop_last=is_train
            )

        train_loader = dataloader(train_dataset, is_train=True)
        test_loader = dataloader(test_dataset, is_train=False)

        feature_config = FeatureHyperParams.build(train_features, metadata)
        model = self.build_model(feature_config).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        trainer = PyTorchTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            patience=self.exp_params.early_stopping_patience
        )

        return trainer, model, train_loader, test_loader, DEVICE, NUM_EPOCHS

    def run_experiment(
            self,
            train_features: FeatureSet,
            test_features: FeatureSet,
            metadata: FeatureMetadata
    ):
        return self.do_run_experiment(train_features, test_features, metadata)[0]

    def do_run_experiment(
            self,
            train_features: FeatureSet,
            test_features: FeatureSet,
            metadata: FeatureMetadata
    ):
        trainer, model, train_loader, test_loader, DEVICE, NUM_EPOCHS = self._setup_training(train_features,
                                                                                             test_features, metadata)
        final_metrics = trainer.fit(train_loader, test_loader, NUM_EPOCHS)
        diagnostics = self._calculate_optimal_threshold_metrics(model, test_loader, DEVICE)
        final_metrics_rounded = {k: r(v) for k, v in final_metrics.items()}
        return {**final_metrics_rounded, **diagnostics, "embedder.model_name": str(self.emb_params.model_name)}, model

    def run_experiment_and_return_model(
            self,
            train_features: FeatureSet,
            test_features: FeatureSet,
            metadata: FeatureMetadata
    ):
        """
        Runs training and returns the trained model (restored to best state)
        along with metrics.
        """
        # Re-use logic via do_run_experiment
        metrics, model = self.do_run_experiment(train_features, test_features, metadata)
        return metrics, model

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
        return create_train_val_test_split(
            test_size=test_size,
            val_size=val_size,
            random_state=self.exp_params.random_state,
            full_df=self.full_df.copy(),
            field_config=self.field_config
        )

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
            'cv_best_epoch': r(np.mean([m.get('best_epoch', 0) for m in all_metrics])),
            'cv_epoch_1_stop': r(np.mean([1 if m.get('best_epoch', 0) == 1 else 0 for m in all_metrics]))
        }

        logger.info("Cross-Validation complete.")
        logger.info(f"Average F1: {avg_metrics['cv_f1']:.4f} +/- {avg_metrics['cv_f1_std']:.4f}")
        logger.info(f"Average ROC-AUC: {avg_metrics['cv_roc_auc']:.4f}")

        return avg_metrics

    def build_model(self, feature_config):
        if isinstance(self.model_params, HybridModel.MlpHyperParams):
            return HybridModel(
                feature_config=feature_config,
                mlp_config=self.model_params
            )

        elif isinstance(self.model_params, TransformerHyperParams):
            return TabularTransformerModel(
                feature_config=feature_config,
                transformer_config=self.model_params
            )

        else:
            raise TypeError(f"Unknown model_params type: {type(self.model_params)}")

    def save_checkpoint(self, model: HybridModel, path: str | Path):
        """Saves model weights AND configuration in one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create payload
        payload = {
            "state_dict": model.state_dict(),
            'emb_params': self.emb_params,
            'feat_proc_params': self.feat_proc_params,
            'model_params': self.model_params,
            'exp_config': self.exp_params,
            'field_config': self.field_config,
            'feature_config': model.feature_config
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path):
        """Saves model weights AND configuration in one file."""
        payload = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        # Create payload
        # TODO update self
        # runner = cls.create(
        #     emb_params=payload['emb_params'],
        #     feat_proc_params=payload['feat_proc_params'],
        #     model_params=payload['model_params'],
        #     full_df=df_full,
        #     field_config=payload['field_config'],
        #     exp_params=payload['exp_params']
        # )
        model = self.build_model(feature_config=payload['feature_config'])
        model.load_state_dict(payload['state_dict'])
        return model