"""
MST-GNN: Full Model Assembly
==============================
Ties together all four modules into the complete MST-GNN framework.

Architecture (Fig. 1 in paper):
    Raw Features → [Module A: Attentive LSTM Encoder]
                → [Module B: Multi-Layer STNA]
                → [Module C: HOFF (DCN)]
                → [Module D: Multitask Predictor]
                → Movement Predictions + Ranking Scores

"The stock feature encoding module uses an attentive LSTM to encode
the initial stock features. Then the encoded hidden states are fed
into the spatial-temporal cross-layer high-order fusion module that
includes spatial-temporal neighborhood aggregation and cross-layer
high-order feature fusion."
"""

import torch
import torch.nn as nn

from models.feature_encoder import AttentiveLSTMEncoder
from models.stna import MultiLayerSTNA
from models.hoff import HOFF
from models.predictor import MultitaskPredictor


class MSTGNN(nn.Module):
    """
    MST-GNN: Graph Representation Learning of Multilayer
    Spatial-Temporal Networks for Stock Predictions.

    Full end-to-end model combining:
        A. Attentive LSTM Encoder
        B. Multi-Layer Spatial-Temporal Neighborhood Aggregation
        C. Cross-Layer High-Order Feature Fusion
        D. Multitask Prediction (Movement + Ranking)
    """

    NETWORK_NAMES = ["shareholding", "industry", "topicality", "comovement"]

    def __init__(
        self,
        # Module A params
        input_dim: int = 13,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        # Module B params
        num_networks: int = 4,
        stna_depth: int = 2,
        stna_aggregator: str = "mean",
        stna_hidden_dim: int = 64,
        # Module C params
        cross_layers: int = 3,
        deep_layers: int = 2,
        deep_hidden_dim: int = 128,
        hoff_output_dim: int = 64,
        # Module D params
        prediction_hidden_dim: int = 64,
        num_classes: int = 2,
        # General
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_networks = num_networks

        # --- Module A: Stock Feature Encoding ---
        self.encoder = AttentiveLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )

        # --- Module B: Spatial-Temporal Neighborhood Aggregation ---
        self.stna = MultiLayerSTNA(
            num_layers=num_networks,
            input_dim=lstm_hidden_dim,
            hidden_dim=stna_hidden_dim,
            depth=stna_depth,
            aggregator_type=stna_aggregator,
            dropout=dropout,
        )

        # --- Module C: Cross-Layer High-Order Feature Fusion ---
        self.hoff = HOFF(
            num_networks=num_networks,
            per_network_dim=stna_hidden_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            deep_hidden_dim=deep_hidden_dim,
            output_dim=hoff_output_dim,
            dropout=dropout,
        )

        # --- Module D: Multitask Prediction ---
        self.predictor = MultitaskPredictor(
            input_dim=hoff_output_dim,
            hidden_dim=prediction_hidden_dim,
            num_classes=num_classes,
        )

        # Store previous hidden states for temporal connections
        self._prev_hiddens = None

    def reset_temporal_state(self):
        """Reset stored temporal hidden states (call at start of sequence)."""
        self._prev_hiddens = None

    def forward(
        self,
        node_features: torch.Tensor,
        networks: dict,
        return_intermediate: bool = False,
    ) -> dict:
        """
        Forward pass through the full MST-GNN pipeline.

        Args:
            node_features: (num_stocks, T, d) — sequential stock features
            networks: Dict mapping network_name -> (edge_index, edge_weight)
                     Keys: "shareholding", "industry", "topicality", "comovement"
            return_intermediate: If True, also return intermediate representations

        Returns:
            Dict with keys:
            - "movement_logits": (num_stocks, num_classes)
            - "ranking_scores": (num_stocks, 1)
            - "fused_repr": (num_stocks, output_dim) [if return_intermediate]
            - "encoded": (num_stocks, d1) [if return_intermediate]
            - "stna_outputs": Dict [if return_intermediate]
        """
        network_names = self.NETWORK_NAMES[: self.num_networks]

        # --- Module A: Encode sequential features ---
        # x_{i,t} ∈ R^{T×d} → h_i ∈ R^{d1}
        encoded = self.encoder(node_features)  # (num_stocks, d1)

        # --- Module B: STNA per network layer ---
        stna_outputs = self.stna(
            encoded,
            networks,
            prev_hiddens=self._prev_hiddens,
            network_names=network_names,
        )

        # Store current hidden states for next timestep's temporal connection
        self._prev_hiddens = {
            name: h.detach() for name, h in stna_outputs.items()
        }

        # --- Module C: Cross-layer high-order feature fusion ---
        fused = self.hoff(stna_outputs, network_names)  # (num_stocks, output_dim)

        # --- Module D: Multitask prediction ---
        movement_logits, ranking_scores = self.predictor(fused)

        result = {
            "movement_logits": movement_logits,
            "ranking_scores": ranking_scores,
        }

        if return_intermediate:
            result["encoded"] = encoded
            result["stna_outputs"] = stna_outputs
            result["fused_repr"] = fused

        return result

    @classmethod
    def from_config(cls, config) -> "MSTGNN":
        """Create MST-GNN from a Config object."""
        return cls(
            input_dim=config.model.input_dim,
            lstm_hidden_dim=config.model.lstm_hidden_dim,
            lstm_num_layers=config.model.lstm_num_layers,
            num_networks=config.model.num_network_layers,
            stna_depth=config.model.stna_depth,
            stna_aggregator=config.model.stna_aggregator,
            stna_hidden_dim=config.model.stna_hidden_dim,
            cross_layers=config.model.cross_network_layers,
            deep_layers=config.model.deep_network_layers,
            deep_hidden_dim=config.model.deep_network_dim,
            hoff_output_dim=config.model.stna_hidden_dim,
            prediction_hidden_dim=config.model.prediction_hidden_dim,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_module_parameters(self) -> dict:
        """Get parameter count per module."""
        return {
            "encoder": sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            ),
            "stna": sum(
                p.numel() for p in self.stna.parameters() if p.requires_grad
            ),
            "hoff": sum(
                p.numel() for p in self.hoff.parameters() if p.requires_grad
            ),
            "predictor": sum(
                p.numel() for p in self.predictor.parameters() if p.requires_grad
            ),
        }
