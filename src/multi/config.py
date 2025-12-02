from dataclasses import dataclass, field
import torch
from common.config import EmbModel, get_device


@dataclass(frozen=True)
class MultiFieldConfig:
    date: str = 'date'
    amount: str = 'amount'
    text: str = 'bankRawDescription'
    label: str = 'isRecurring'
    accountId: str = 'accountId'
    trId: str = 'trId'
    patternId: str = 'patternId'
    patternCycle: str = 'patternCycle'
    counter_party: str = 'counter_party'
    bank_name: str = 'bank_name'


@dataclass
class MultiExpConfig:
    """
    Central configuration for the Multi experiment.
    """
    # Paths
    random_state: int = 0x5EED

    data_path: str = "data/all_data.csv"
    output_dir: str = "checkpoints/multi"
    downsample: float = 0.3  # 1.0 means no downsampling

    # --- Task Configuration ---
    # Options: 'multiclass' (Original), 'binary' (New: Adjacency + isRecurring)
    task_type: str = "binary"

    # Model Hyperparameters
    emb_model:EmbModel = EmbModel.MPNET.value

    # If 0, we can optionally use use_cached_embeddings=True for massive speedup
    unfreeze_last_n_layers: int = 0

    # NEW: If True, uses EmbeddingService to pre-compute vectors.
    # Must be False if unfreeze_last_n_layers > 0.
    use_cached_embeddings: bool = False

    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_counter_party: bool = True

    # Options: "no", "edge_informed_max" (Legacy/Multiclass only)
    edge_informed_type: str = "no"

    # Normalization
    normalization_type: str = "layer_norm"  # 'layer_norm', 'rms_norm', 'none'

    # Encoder internals
    chunk_size: int = 2048  # For memory efficient encoding
    time_encoding_max_len: int = 1000  # using relative date

    # Training Hyperparameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 15
    max_seq_len: int = 200
    metric_to_track: str = 'pr_auc'
    early_stopping_patience: int = 4
    scheduler_patience: int = 2

    # Optimization Flags
    gradient_accumulation_steps: int = 1
    use_contrastive_loss: bool = True
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    hard_negative_weight: float = 1.0
    pos_weight: float = 2.5

    # Focal Loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Scheduler
    scheduler_type: str = 'cosine'  # 'plateau' or 'cosine'
    scheduler_t0: int = 5
    scheduler_t_mult: int = 2
    max_grad_norm: float = 1.0

    # Tokenizer Limits
    max_text_length: int = 44
    max_cp_length: int = 20

    # Cycle Labels (Only used if task_type='multiclass')
    cycle_map: dict = field(default_factory=lambda: {
        'None': 0,
        'monthly': 1,
        'onceAWeek': 2,
        'onceEvery2Weeks': 3,
        'twiceAMonth': 4,
        'onceEvery4Weeks': 5,
        'weekBasedOther': 6,
        'every4_5Weeks': 7
    })

    @property
    def num_classes(self) -> int:
        return len(self.cycle_map)

    @property
    def device(self) -> torch.device:
        return get_device()

    @property
    def text_encoder_model(self) -> str:
        return self.emb_model.value()