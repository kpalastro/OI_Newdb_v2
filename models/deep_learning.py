"""
Deep learning models for OI Gemini (Phase 2).

Implements PyTorch-based LSTM and Temporal Convolutional Network models
for time-series option chain prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

LOGGER = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM-based time-series model for option chain prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time-series prediction."""
    
    def __init__(
        self,
        input_size: int,
        num_channels: list[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        tcn_out = self.tcn(x)
        # Take last time step
        last_output = tcn_out[:, :, -1]
        output = self.fc(last_output)
        return output


class DeepLearningPredictor:
    """Wrapper for deep learning model inference."""
    
    def __init__(
        self,
        exchange: str,
        model_type: str = "lstm",
        model_path: Optional[Path] = None,
    ):
        self.exchange = exchange
        self.model_type = model_type.lower()
        self.model: Optional[nn.Module] = None
        self.model_loaded = False
        self.input_size = 0
        self.sequence_length = 20  # Default sequence length
        
        if not TORCH_AVAILABLE:
            LOGGER.warning(f"[{exchange}] PyTorch not available, deep learning disabled")
            return
        
        if model_path and model_path.exists():
            self._load_model(model_path)
        else:
            model_dir = Path("models") / exchange / "nn"
            if model_dir.exists():
                model_files = list(model_dir.glob(f"{model_type}_*.pth"))
                if model_files:
                    self._load_model(model_files[0])
    
    def _load_model(self, model_path: Path) -> None:
        """Load a trained PyTorch model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            self.input_size = config.get('input_size', 50)
            self.sequence_length = config.get('sequence_length', 20)
            
            if self.model_type == "lstm":
                self.model = LSTMModel(
                    input_size=self.input_size,
                    hidden_size=config.get('hidden_size', 64),
                    num_layers=config.get('num_layers', 2),
                    dropout=config.get('dropout', 0.2),
                )
            elif self.model_type == "tcn":
                self.model = TemporalConvNet(
                    input_size=self.input_size,
                    num_channels=config.get('num_channels', [64, 64, 64]),
                    kernel_size=config.get('kernel_size', 3),
                    dropout=config.get('dropout', 0.2),
                )
            else:
                LOGGER.error(f"[{self.exchange}] Unknown model type: {self.model_type}")
                return
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            LOGGER.info(f"[{self.exchange}] Deep learning model loaded: {model_path}")
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] Failed to load deep learning model: {e}")
    
    def predict(self, feature_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Generate prediction from feature sequence.
        
        Args:
            feature_sequence: Array of shape (sequence_length, num_features)
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.model_loaded or self.model is None:
            return 0, 0.0
        
        try:
            if feature_sequence.shape[0] < self.sequence_length:
                # Pad with zeros if sequence is too short
                padding = np.zeros((self.sequence_length - feature_sequence.shape[0], feature_sequence.shape[1]))
                feature_sequence = np.vstack([padding, feature_sequence])
            elif feature_sequence.shape[0] > self.sequence_length:
                # Take last N timesteps
                feature_sequence = feature_sequence[-self.sequence_length:]
            
            # Convert to tensor and add batch dimension
            tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, predicted_class].item())
            
            # Map to signal: 0=SELL, 1=HOLD, 2=BUY -> -1, 0, 1
            signal_class = predicted_class - 1
            
            return signal_class, confidence
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] Deep learning prediction error: {e}")
            return 0, 0.0

