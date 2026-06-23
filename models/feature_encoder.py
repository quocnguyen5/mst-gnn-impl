"""
Stock Feature Encoder — Module A
=================================
Attentive LSTM for encoding sequential stock features.

Reference: Section IV-B of the paper (Eqs. 4-7)

"For an individual stock i, its initial features are a multivariable
time series from a historical observation with T time steps."

"The dynamics of stock market time series have complex temporal patterns.
Therefore, our method exploits an attentive LSTM to encode the sequential
features."

Architecture:
    x_{i,t} ∈ R^{T×d}  →  LSTM  →  Temporal Attention  →  h_i ∈ R^{d1}

The attentive LSTM shares parameters among different stock networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for weighting LSTM hidden states.

    "The temporal attention network is used to adaptively devote more
    focus to the important parts of the sequential features by
    context-dependent learning."

    Eqs. (5)-(7):
        e_{i,t} = v^T tanh(W_h h̃_{i,t} + b)           [Eq. 5]
        α_{i,t} = softmax(e_{i,t})                       [Eq. 6]
        h_i = Σ_t α_{i,t} h̃_{i,t}                      [Eq. 7]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_outputs: LSTM hidden states, shape (batch, T, d1)

        Returns:
            Attended representation, shape (batch, d1)
        """
        # Eq. 5: e_{i,t} = v^T tanh(W_h h̃_{i,t} + b)
        energy = self.v(torch.tanh(self.W_h(lstm_outputs)))  # (batch, T, 1)
        energy = energy.squeeze(-1)  # (batch, T)

        # Eq. 6: α_{i,t} = softmax(e_{i,t})
        attention_weights = F.softmax(energy, dim=-1)  # (batch, T)

        # Eq. 7: h_i = Σ_t α_{i,t} h̃_{i,t}
        attended = torch.bmm(
            attention_weights.unsqueeze(1), lstm_outputs
        )  # (batch, 1, d1)
        attended = attended.squeeze(1)  # (batch, d1)

        return attended


class AttentiveLSTMEncoder(nn.Module):
    """
    Module A: Stock Feature Encoding with Attentive LSTM.

    "The attentive LSTM shares parameters among different stock networks."

    Input:  x_{i,t} ∈ R^{T×d}  (T timesteps, d features per timestep)
    Output: h_i ∈ R^{d1}       (encoded stock representation)

    Process:
        1. LSTM encodes the T-step sequence → hidden states h̃_{i,t}  [Eq. 4]
        2. Temporal attention weights and aggregates hidden states    [Eqs. 5-7]
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: d — number of input features (13 in paper)
            hidden_dim: d1 — LSTM hidden dimension (64 in paper)
            num_layers: number of LSTM layers
            dropout: dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Eq. 4: h̃_{i,t}, c̃_{i,t} = LSTM(x_{i,t})
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention (Eqs. 5-7)
        self.attention = TemporalAttention(hidden_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequential stock features.

        Args:
            x: Stock features, shape (num_stocks, T, d)
               T = lookback window, d = number of features

        Returns:
            Encoded representations, shape (num_stocks, d1)
        """
        # Eq. 4: LSTM encoding
        # lstm_out: (num_stocks, T, d1) — all hidden states
        lstm_out, _ = self.lstm(x)

        # Eqs. 5-7: Temporal attention
        attended = self.attention(lstm_out)  # (num_stocks, d1)

        # Normalize and dropout
        encoded = self.layer_norm(attended)
        encoded = self.dropout(encoded)

        return encoded
