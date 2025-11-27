from dataclasses import dataclass, field
import torch
from common.config import EmbModel

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
    counter_party:str = 'counterParty'
    bank_name: str = 'bank_name'


@dataclass
class MultiExpConfig:
    """
    Central configuration for the Multi experiment.
    """
    # Paths
    random_state: int = 0x5EED
    data_path: str = "./data/transactions.csv"
    output_dir: str = "./checkpoints"

    # Model Hyperparameters
    text_encoder_model: str = EmbModel.MPNET.value
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Training Hyperparameters
    batch_size: int = 16  # Accounts per batch
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_seq_len: int = 200  # Cap N to avoid OOM on extreme outliers

    # Cycle Labels
    # We ignore 'Annual'/'SemiAnnual'.
    # 0 is reserved for 'None' (non-recurring).
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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")