"""
Autoencoder model for anomaly detection.
"""
import torch
import torch.nn as nn
from typing import Tuple


class Autoencoder(nn.Module):
    """
    Autoencoder used to spot unusual earthquake patterns.

    It compresses features into a latent space and reconstructs them.
    Larger reconstruction error usually means a more unusual sample.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Build the encoder/decoder layers.
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: squeeze features into a smaller space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder: rebuild the original feature space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Keep outputs in a 0-1 range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return the reconstruction.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the latent representation for the input.
        """
        return self.encoder(x)
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection.
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error
