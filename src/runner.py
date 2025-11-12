import time
from typing import Self

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from feature_processor import HybridFeatureProcessor, check_unknown_rate, FeatProcParams
from config import *
from embedder import EmbeddingService
import logging

logger = logging.getLogger(__name__)


from config import BaseModel

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
        self.embedder_map: dict[BaseModel, EmbeddingService] = {}

    def get_embedder(self, model_params:EmbeddingService.Params) -> EmbeddingService:
        model_name = model_params.model_name
        if model_name not in self.embedder_map:
            logger.info(f"Creating new EmbeddingService(model_name={model_name})")
            self.embedder_map[model_name] = EmbeddingService.create(model_params)
        return self.embedder_map[model_name]

    def build_data(self, emb_params : EmbeddingService.Params|None=None) -> ExpData:
        embedder = self.get_embedder(emb_params or self.emb_params)

        logger.info("\n" + "=" * 50)
        logger.info(f"RUNNING EXPERIMENT WITH BASE MODEL: {embedder.model_name}")
        logger.info("=" * 50)

        # --- Split Data ---
        train_df, test_df = train_test_split(
            self.full_df,
            test_size=self.exp_params.test_size,
            random_state=self.exp_params.random_state,
            stratify=self.full_df[self.field_config.label]
        )
        y_train = train_df[self.field_config.label]
        y_test = test_df[self.field_config.label]

        logger.info(f"Total data: {len(self.full_df)}, Train: {len(train_df)}, Test: {len(test_df)}")
        logger.info(f"Train set positive class %: {y_train.mean() * 100:.2f}%")

        # --- Create Text Features (Cached) ---
        logger.info("\nProcessing text features (using EmbeddingService)...")
        logger.info(f"Embedding {len(train_df)} train texts...")
        train_text_features_np = embedder.embed(train_df[self.field_config.text].tolist())

        logger.info(f"Embedding {len(test_df)} test texts...")
        test_text_features_np = embedder.embed(test_df[self.field_config.text].tolist())

        # --- Create Numerical Features ---
        if self.feat_proc_params.is_nop():
            X_train = train_text_features_np
            X_test = test_text_features_np
        else:
            logger.info("\nProcessing numerical/date features...")
            processor = HybridFeatureProcessor.create(self.feat_proc_params)

            processor.fit(train_df)

            train_num_features_df = processor.transform(train_df)
            test_num_features_df = processor.transform(test_df)

            # --- Health Check (Go/No-Go) ---
            logger.info("\n--- Health Check on Processor ---")
            report = check_unknown_rate(processor, test_num_features_df, "Test Set")
            test_unknown_pct = report.get('percent', 100.0)

            if test_unknown_pct > self.exp_params.go_no_go_threshold_pct:
                logger.info(f"**NO-GO!** Test [UNKNOWN] rate is {test_unknown_pct:.2f}%. Halting experiment.")
                return {}
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

        def r(x: float):
            return round(x, 3)

        res = {
            "f1": r(f1),
            "roc_auc": r(roc_auc),
            "accuracy": r(accuracy),
            "embedder.model_name": str(self.emb_params.model_name),
        }

        logger.info(res)

        return res

