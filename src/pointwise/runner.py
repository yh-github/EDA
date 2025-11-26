import logging
from typing import Self, Any

import numpy as np
import pandas as pd
import torch
from numpy import floating
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from common.config import ExperimentConfig, FieldConfig, EmbModel, get_device
from common.data import TransactionDataset, FeatureSet, TrainingSample, create_train_val_test_split
from common.embedder import EmbeddingService
from common.exp_utils import set_global_seed
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureMetadata, FeatureHyperParams
from pointwise.classifier import HybridModel
from pointwise.classifier_transformer import TransformerHyperParams, TabularTransformerModel
from pointwise.trainer import PyTorchTrainer

logger = logging.getLogger(__name__)

ModelParams = HybridModel.MlpHyperParams | TransformerHyperParams


def r(x: float | floating[Any]) -> float:
    return round(float(x), 3)


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
        full_df = self.full_df.copy()
        gss = GroupShuffleSplit(n_splits=1, test_size=self.exp_params.test_size,
                                random_state=self.exp_params.random_state)
        train_idx, test_idx = next(
            gss.split(full_df, y=full_df[self.field_config.label], groups=full_df[self.field_config.accountId]))
        return full_df.iloc[train_idx], full_df.iloc[test_idx]

    def create_train_val_test_split(self, test_size: float = 0.2, val_size: float = 0.2) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return create_train_val_test_split(
            test_size=test_size, val_size=val_size,
            random_state=self.exp_params.random_state,
            full_df=self.full_df.copy(), field_config=self.field_config
        )

    def create_learning_curve_splits(self, full_train_df: pd.DataFrame, fractions: list[float]):
        unique_accounts = full_train_df[self.field_config.accountId].unique()
        rng = np.random.RandomState(self.exp_params.random_state)
        shuffled_accounts = rng.permutation(unique_accounts)
        total_accounts = len(unique_accounts)

        for frac in sorted(fractions):
            if frac <= 0.0: continue
            if frac >= 1.0:
                yield 1.0, full_train_df
                continue
            n_take = int(total_accounts * frac)
            if n_take == 0: continue
            subset = full_train_df[full_train_df[self.field_config.accountId].isin(shuffled_accounts[:n_take])].copy()
            yield frac, subset

    def build_data_three_way(self, df_train, df_val, df_test) -> tuple[
        FeatureSet, FeatureSet, FeatureSet, HybridFeatureProcessor, FeatureMetadata]:
        return self._build_data_internal(df_train, df_val, df_test)

    def build_data(self, df_train_frac, df_test) -> tuple[
        FeatureSet, FeatureSet, HybridFeatureProcessor, FeatureMetadata]:
        res = self._build_data_internal(df_train_frac, df_test, None)
        return res[0], res[1], res[3], res[4]

    def _build_data_internal(self, df_train, df_val, df_test=None):
        embedder = self.get_embedder(self.emb_params)

        def get_text(df):
            return embedder.embed(df[self.field_config.text].tolist())

        X_txt_tr = get_text(df_train)
        X_txt_val = get_text(df_val)
        X_txt_te = get_text(df_test) if df_test is not None else None

        processor = HybridFeatureProcessor.create(self.feat_proc_params, self.field_config)
        meta = processor.fit(df_train)

        df_tr_f = processor.transform(df_train)
        df_val_f = processor.transform(df_val)
        df_te_f = processor.transform(df_test) if df_test is not None else None

        scaler = StandardScaler()
        if meta.continuous_scalable_cols:
            scaler.fit(df_tr_f[meta.continuous_scalable_cols])

        def build_fs(df_orig, df_feat, x_txt):
            y = df_orig[self.field_config.label].values
            parts = []
            if meta.cyclical_cols: parts.append(df_feat[meta.cyclical_cols].values)
            if meta.continuous_scalable_cols: parts.append(scaler.transform(df_feat[meta.continuous_scalable_cols]))

            X_cont = np.concatenate(parts, axis=1) if parts else np.empty((len(df_orig), 0))
            X_cat = df_feat[list(meta.categorical_features.keys())].values
            return FeatureSet(X_text=x_txt, X_continuous=X_cont, X_categorical=X_cat, y=y)

        fs_tr = build_fs(df_train, df_tr_f, X_txt_tr)
        fs_val = build_fs(df_val, df_val_f, X_txt_val)

        if df_test is not None:
            fs_te = build_fs(df_test, df_te_f, X_txt_te)
            return fs_tr, fs_val, fs_te, processor, meta

        return fs_tr, fs_val, None, processor, meta

    def _setup_training(self, train_features, test_features, metadata):
        DEVICE = get_device()
        train_loader = DataLoader(TransactionDataset(train_features), batch_size=self.exp_params.batch_size,
                                  shuffle=True, collate_fn=TrainingSample.collate_fn)
        test_loader = DataLoader(TransactionDataset(test_features), batch_size=self.exp_params.batch_size,
                                 shuffle=False, collate_fn=TrainingSample.collate_fn)

        feature_config = FeatureHyperParams.build(train_features, metadata)
        model = self.build_model(feature_config).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.exp_params.learning_rate)

        trainer = PyTorchTrainer(model, optimizer, nn.BCEWithLogitsLoss(), DEVICE,
                                 patience=self.exp_params.early_stopping_patience)
        return trainer, model, train_loader, test_loader, DEVICE

    def run_experiment(self, train_features, test_features, metadata):
        return self.run_experiment_and_return_model(train_features, test_features, metadata)[0]

    def run_experiment_and_return_model(self, train_features, test_features, metadata):
        trainer, model, train_loader, test_loader, DEVICE = self._setup_training(train_features, test_features,
                                                                                 metadata)
        metrics = trainer.fit(train_loader, test_loader, self.exp_params.epochs)

        if trainer.best_model_state:
            model.load_state_dict(trainer.best_model_state)

        return metrics, model

    # --- NEW: Evaluate existing model on new set ---
    def evaluate_model_on_set(self, model, feature_set):
        DEVICE = get_device()
        dataset = TransactionDataset(feature_set)
        loader = DataLoader(dataset, batch_size=self.exp_params.batch_size, shuffle=False,
                            collate_fn=TrainingSample.collate_fn)

        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                x_text = batch.x_text.to(DEVICE)
                x_cont = batch.x_continuous.to(DEVICE)
                x_cat = batch.x_categorical.to(DEVICE)
                y = batch.y.to(DEVICE)

                logits = model(x_text, x_cont, x_cat)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        y_pred = (y_prob > 0.5).astype(int)

        p, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_prob)

        return {
            "precision": r(p), "recall": r(recall), "f1": r(f1), "roc_auc": r(auc),
            "y_true": y_true, "y_pred": y_pred  # Return raw for detailed reports if needed
        }

    def build_model(self, feature_config):
        if isinstance(self.model_params, HybridModel.MlpHyperParams):
            return HybridModel(feature_config, self.model_params)
        elif isinstance(self.model_params, TransformerHyperParams):
            return TabularTransformerModel(feature_config, self.model_params)
        raise TypeError(f"Unknown model type: {type(self.model_params)}")