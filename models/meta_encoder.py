"""Encoder for scalar metadata for translation disambiguation. Receives a [10,] tensor"""

import torch
import torch.nn as nn


class MetaEncoder(nn.Module):
    """
    Encodes scalar metadata (bbox location, size, camera intrinsics) for translation disambiguation.
    
    Input: (batch_size, 10) tensor containing:
        - uc, vc: normalized bbox center coordinates
        - wn, hn: normalized bbox width/height
        - area_n: normalized bbox area
        - ar: aspect ratio
        - fx_n, fy_n, cx_n, cy_n: normalized camera intrinsics
    
    Output: (batch_size, output_dim) encoded features
    """
    
    def __init__(self, input_dim: int = 10, output_dim: int = 32, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input metadata (default: 10)
            output_dim: Dimension of output encoded features (default: 32)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-layer perceptron to encode metadata
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            meta: (batch_size, 10) metadata tensor
        
        Returns:
            (batch_size, output_dim) encoded features
        """
        return self.encoder(meta)

