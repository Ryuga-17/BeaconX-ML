"""
Autoencoder model definition for anomaly detection.
"""
import torch
import torch.nn as nn
from typing import Tuple


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection in earthquake data.
    
    This model compresses input features into a latent representation and then
    reconstructs them, allowing for anomaly detection based on reconstruction error.
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden layers
        latent_dim (int): Size of latent representation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize the Autoencoder model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            latent_dim: Size of latent representation
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: Compress data to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder: Reconstruct data from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid for normalized data (0 to 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error for anomaly detection.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstruction error tensor of shape (batch_size,)
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error
