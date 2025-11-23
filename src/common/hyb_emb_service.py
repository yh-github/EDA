import logging
from enum import StrEnum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from classifier import HybridModel
from common.config import FieldConfig, EmbModel, get_device
from embedder import EmbeddingService
from feature_processor import FeatureHyperParams

logger = logging.getLogger(__name__)
DEVICE = get_device()


class HybridEmbeddingService:
    """
    Loads a fine-tuned HybridModel and generates 'Smart Embeddings'
    from raw transaction DataFrames.
    """

    class HyEmbModel(StrEnum):
        """Using StrEnum for our model name constants."""
        MLP1 = "ft_models/mlp_hybrid"

    def __init__(self, emb_model:HyEmbModel):
        self.emb_model = emb_model
        self.model_dir = Path(emb_model.value)
        self.field_config = FieldConfig()
        self.device = DEVICE

        self._load_artifacts()

    def _load_artifacts(self):
        logger.info(f"Loading {self.emb_model} artifacts from {self.model_dir}...")

        # 1. Load Processor & Scaler
        self.processor = joblib.load(self.model_dir / "processor.joblib")
        if (self.model_dir / "scaler.joblib").exists():
            self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        else:
            self.scaler = None

        # 2. Initialize Base Text Embedder (MPNET)
        # We need this to generate the input for our Hybrid Model
        self.text_embedder = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=256)

        # 3. Reconstruct Model Architecture
        # We need to rebuild the config to match the saved weights exactly.
        # We derive dimensions by "dry running" the processor on a dummy row.
        dummy_row = {
            self.field_config.date: "01/01/2023",
            self.field_config.amount: 10.00,
            self.field_config.text: "DUMMY"
        }
        dummy_df = pd.DataFrame([dummy_row])

        # Get metadata from the loaded processor
        # Note: We assume the processor kept its internal state (vocab_map, etc.)
        # We run transform to get the shape of continuous features
        processed_dummy = self.processor.transform(dummy_df)

        # Infer Input Dimensions
        text_dim = 768  # MPNET default

        # Continuous Dim
        cyc_cols = [c for c in processed_dummy.columns if 'sin' in c or 'cos' in c]
        cont_cols = [c for c in processed_dummy.columns if
                     c == 'log_abs_amount']  # simplistic check based on your FeatParams
        # Better way: look at what the processor produced

        # We need to match the exact concatenation logic from fine_tune_hybrid.py
        # Recalculating dimensions based on the processor's known state is safer.
        meta = self.processor._build_meta()  # Re-generate metadata from internal state

        # Build FeatureHyperParams manually or via helper
        # Logic mirrored from fine_tune_hybrid.py main()
        processed_cols = processed_dummy.shape[1]
        # We need to know exactly how many cols are categorical vs continuous
        num_cats = len(meta.categorical_features)
        cont_dim = processed_cols - num_cats

        # Reconstruct FeatureConfig
        # NOTE: You must match the vocab sizes used during training
        cat_config = {k: v.vocab_size for k, v in meta.categorical_features.items()}
        emb_dims = {k: v.embedding_dim for k, v in meta.categorical_features.items()}

        feature_config = FeatureHyperParams(
            text_embed_dim=text_dim,
            continuous_feat_dim=cont_dim,
            categorical_vocab_sizes=cat_config,
            embedding_dims=emb_dims
        )

        # Reconstruct MLP Config (Must match training script!)
        mlp_config = HybridModel.MlpHyperParams(
            text_projection_dim=128,
            mlp_hidden_layers=[128, 64],
            dropout_rate=0.0  # No dropout for inference
        )

        # 4. Load Model Weights
        self.model = HybridModel(feature_config, mlp_config)
        #TODO file name shouldn't be hardcoded here
        state_dict = torch.load(self.model_dir / "mlp_model_state.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Hybrid Model loaded successfully.")

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        """
        The main entry point. Converts a DataFrame of transactions into
        fine-tuned embeddings.
        """
        if df.empty:
            return np.array([])

        # A. Text Embeddings (Base)
        texts = df[self.field_config.text].tolist()
        # Uses caching internally
        x_text_np = self.text_embedder.embed(texts)

        # B. Process Features
        processed_df = self.processor.transform(df)
        meta = self.processor._build_meta()  # Get metadata to know which cols are which

        # C. Prepare Continuous
        # Extract Cyclical
        x_cyc = processed_df[meta.cyclical_cols].values if meta.cyclical_cols else np.zeros((len(df), 0))

        # Extract and Scale Continuous
        x_cont_raw = processed_df[meta.continuous_scalable_cols].values if meta.continuous_scalable_cols else np.zeros(
            (len(df), 0))

        if self.scaler and x_cont_raw.shape[1] > 0:
            x_cont_scaled = self.scaler.transform(x_cont_raw)
        else:
            x_cont_scaled = x_cont_raw

        x_continuous_np = np.concatenate([x_cyc, x_cont_scaled], axis=1)

        # D. Prepare Categorical
        cat_cols = list(meta.categorical_features.keys())
        if cat_cols:
            x_cat_np = processed_df[cat_cols].values
        else:
            x_cat_np = np.zeros((len(df), 0))

        # E. Convert to Tensor
        x_text = torch.from_numpy(x_text_np).float().to(self.device)
        x_cont = torch.from_numpy(x_continuous_np).float().to(self.device)
        x_cat = torch.from_numpy(x_cat_np).long().to(self.device)

        # F. Forward Pass (Inference)
        with torch.no_grad():
            # Use .embed() to get the vector, NOT .forward() which gives logits
            embeddings = self.model.embed(x_text, x_cont, x_cat)

        return embeddings.cpu().numpy()