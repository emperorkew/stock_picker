import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
from stock_picker.db import get_client

class StockLSTM(nn.Module):
    """LSTM network for sequence prediction returning log-returns."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, feature_count)
        out, _ = self.lstm(x)

        # Take the output of the last time step
        last_out = out[:, -1, :]

        # Output log-return
        return self.fc(last_out)


def get_historical_data(asset_id: str, limit: int = 30) -> pd.DataFrame:
    """Fetch the last `limit` historical snapshots for an asset from Supabase."""
    client = get_client()
    response = client.table("market_snapshots") \
        .select("*") \
        .eq("asset_id", asset_id) \
        .order("timestamp", desc=True) \
        .limit(limit) \
        .execute()

    df = pd.DataFrame(response.data)
    if not df.empty:
        # Sort chronologically
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df

def preprocess_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[torch.Tensor, MinMaxScaler]:
    """Scale features and format them into the 3D tensor shape (1, 30, F)."""
    # Handle missing data (e.g. forward fill, then backward fill)
    df_features = df[feature_cols].copy()
    df_features = df_features.ffill().bfill()

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    # Reshape to (1, sequence_length, feature_count)
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

    return tensor_data, scaler

def log_return_to_percentage(log_return: float) -> float:
    """Convert predicted log-return back to nominal percentage change."""
    return (np.exp(log_return) - 1.0) * 100.0

def percentage_to_log_return(percentage: float) -> float:
    """Convert percentage change to log-return."""
    return np.log(1.0 + (percentage / 100.0))
