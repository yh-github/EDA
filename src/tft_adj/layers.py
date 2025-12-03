import torch
import torch.nn as nn
import math


class GatedResidualNetwork(nn.Module):
    """
    The core building block of TFT.
    Applies non-linear processing with a Skip Connection and Gating mechanism.
    Good for learning complex functions while allowing simple signals to pass through.
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = None

        self.fc1 = nn.Linear(self.input_size + (context_size or 0), self.hidden_size)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.gate = nn.Linear(self.hidden_size, self.output_size)

        self.layernorm = nn.LayerNorm(self.output_size)

    def forward(self, x, context=None):
        residual = self.skip_layer(x) if self.skip_layer else x

        if context is not None:
            x_c = torch.cat((x, context), dim=-1)
        else:
            x_c = x

        x = self.fc1(x_c)
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Gating mechanism (GLU-like)
        gate = torch.sigmoid(self.gate(x))
        x = x * gate

        return self.layernorm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Allows the model to weight features dynamically.
    For example, for 'Netflix', 'Amount' might be important.
    For 'Check', 'Amount' might vary, but 'Text' is important.
    """

    def __init__(self, input_sizes, hidden_size, dropout=0.1, context_size=None):
        """
        input_sizes: dict mapping feature_name -> feature_dim
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.feature_names = sorted(list(input_sizes.keys()))

        # Individual GRNs for each feature
        self.single_variable_grns = nn.ModuleDict({
            name: GatedResidualNetwork(size, hidden_size, hidden_size, dropout, context_size)
            for name, size in input_sizes.items()
        })

        # Weighting GRN to decide importance of each feature
        # Input is flattened concatenation of all features (or a summary state)
        # Simplified: We use a learnable vector or simple projection for weights
        # TFT uses a GRN on the flattened state vector to produce weights.

        # We assume pre-projected inputs for the weight calculator for simplicity
        # or just use a GRN on the context if available.
        # Here we implement a simplified VSN that transforms inputs first.

        self.weight_grn = GatedResidualNetwork(
            hidden_size * len(input_sizes),
            hidden_size,
            len(input_sizes),  # Output one weight per feature
            dropout,
            context_size
        )

    def forward(self, x_dict, context=None):
        # 1. Transform each feature individually
        transformed_features = []
        for name in self.feature_names:
            feat_input = x_dict[name]
            transformed = self.single_variable_grns[name](feat_input, context)
            transformed_features.append(transformed)

        # [Batch, Seq, NumFeatures, Hidden]
        stacked_features = torch.stack(transformed_features, dim=2)

        # 2. Calculate Weights
        # We flatten the features to determine weights
        flattened = stacked_features.view(stacked_features.size(0), stacked_features.size(1), -1)
        weights = self.weight_grn(flattened, context)  # [Batch, Seq, NumFeatures]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # [Batch, Seq, NumFeatures, 1]

        # 3. Weighted Sum
        weighted_sum = (stacked_features * weights).sum(dim=2)

        return weighted_sum, weights