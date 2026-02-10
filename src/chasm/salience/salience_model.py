import torch
import torch.nn as nn

class SalienceFusion(nn.Module):
    """
    Learns how to combine multiple salience signals
    """

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))  # α, β, γ

    def forward(self, textrank, bertsum, position):
        """
        All inputs: tensors of shape [N]
        """
        scores = (
            self.weights[0] * textrank +
            self.weights[1] * bertsum +
            self.weights[2] * position
        )
        return scores