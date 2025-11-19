import torch
import torch.nn as nn
from classifier import HybridModel


class SiameseHybridModel(nn.Module):
    """
    Wraps your HybridModel so it can process pairs of inputs
    for Contrastive Learning.
    """

    def __init__(self, hybrid_model: HybridModel):
        super().__init__()
        self.model = hybrid_model

    def forward(self, features):
        """
        sentence-transformers passes a dictionary of features.
        We unpack them for your HybridModel.
        """
        # Check if this is a single input or already batched by the loss function
        # Usually features is {'x_text': ..., 'x_cont': ..., 'x_cat': ...}

        output = self.model.get_representation(
            x_text=features['x_text'],
            x_continuous=features['x_continuous'],
            x_categorical=features['x_categorical']
        )
        return output

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}/hybrid_model.pt")