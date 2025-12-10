import logging
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

class IntradayLSTMModel(nn.Module if torch else object):
    """
    LSTM with attention for 5-second to 1-minute predictions.
    
    Architecture:
    - Input: Sequence of tick features (e.g., past 60 5-sec ticks)
    - Layers: 2-layer LSTM + Multi-Head Attention
    - Output: 3-class probability (Buy, Sell, Hold)
    """
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        if torch is None:
            raise ImportError("PyTorch is required for IntradayLSTMModel")
            
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer: Captures temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention Layer: Focuses on critical time steps
        # Hidden dim * 2 because of bidirectional LSTM
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
        # Output Heads
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # [BUY, SELL, HOLD]
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # LSTM Forward
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Self-Attention
        # query, key, value all from lstm_out
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global Average Pooling (over time dimension)
        # pool_out: (batch, hidden_dim * 2)
        pool_out = torch.mean(attn_out, dim=1)
        
        # Classification
        logits = self.fc_layers(pool_out)
        return F.softmax(logits, dim=-1)

    def predict_single(self, feature_sequence):
        """
        Inference helper for a single sequence.
        Args:
            feature_sequence: Numpy array (seq_len, input_dim)
        Returns:
            dict: {class_index: int, probabilities: list}
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(feature_sequence).unsqueeze(0)
            probs = self(x).squeeze(0).numpy()
            
        return {
            'class': int(probs.argmax()),
            'probabilities': probs.tolist()
        }

