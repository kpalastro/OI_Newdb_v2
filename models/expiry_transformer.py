import logging
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

class ExpiryDayTransformer(nn.Module if torch else object):
    """
    Specialized transformer for expiry day dynamics (0-DTE).
    
    Architecture:
    - Input: Feature vector (including Max Pain, Gamma Flip, Greeks).
    - Embedding: Projects features to higher dim.
    - Encoder: Transformer Encoder to capture non-linear interactions.
    - Heads:
        1. Pin Risk: Binary classification (Will price pin to a strike?).
        2. Gamma Flip: Regression (Distance to nearest flip zone).
        3. Direction: 3-class classification (Buy/Sell/Hold).
    """
    def __init__(self, feature_dim=32, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        if torch is None:
            raise ImportError("PyTorch is required for ExpiryDayTransformer")
            
        super().__init__()
        
        self.embedding = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Heads
        self.pin_risk_head = nn.Linear(d_model, 1)
        self.gamma_flip_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 3) # [SELL, HOLD, BUY]
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
               If (batch, feature_dim), we unsqueeze to (batch, 1, feature_dim) to treat as seq len 1.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, feature_dim)
            
        # Embedding
        emb = self.embedding(x) # (batch, seq, d_model)
        
        # Transformer
        encoded = self.transformer_encoder(emb)
        
        # Global Pooling (Mean over sequence if seq > 1)
        pooled = encoded.mean(dim=1)
        
        # Predictions
        pin_prob = torch.sigmoid(self.pin_risk_head(pooled))
        gamma_dist = self.gamma_flip_head(pooled)
        direction_logits = self.direction_head(pooled)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        return {
            'pin_risk': pin_prob,
            'gamma_dist': gamma_dist,
            'direction_probs': direction_probs
        }

    def predict(self, feature_vector):
        """
        Inference helper.
        Args:
            feature_vector: Numpy array (feature_dim,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(feature_vector).unsqueeze(0) # (1, feature_dim)
            out = self(x)
            
        return {
            'pin_risk': float(out['pin_risk'].item()),
            'gamma_dist': float(out['gamma_dist'].item()),
            'direction': int(out['direction_probs'].argmax().item()),
            'probabilities': out['direction_probs'].squeeze(0).tolist()
        }

