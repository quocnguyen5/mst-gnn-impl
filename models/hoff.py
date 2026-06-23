"""
Cross-Layer High-Order Feature Fusion (HOFF) — Module C
=======================================================
Fuses representations from multiple network layers using
Deep & Cross Network (DCN) architecture.

Reference: Section IV-D of the paper (Eqs. 13-15)

"To leverage these complementary stock networks to produce more
comprehensive features, we introduce a deep cross network (DCN),
a specialized neural network architecture designed to achieve
high-order feature interactions, originally used in click-through
rate (CTR) estimation tasks."

Architecture:
    1. Concatenate representations from M networks → p⁰ᵢ    [Eq. 13]
    2. Cross Network: explicit high-order interactions        [Eq. 14]
    3. Deep Network: implicit nonlinear features              [MLP]
    4. Combination layer: concat cross + deep → final repr    [Eq. 15]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    """
    Single layer of the Cross Network.

    Eq. (14):
        p^{c+1} = p⁰ · (pᶜ)ᵀ · wᶜ + bᶜ + pᶜ

    "Each CL function uses the outer product of the input vector p⁰ᵢ
    and the crossed feature vector to produce higher-order interaction."

    Note: The outer product p⁰ · (pᶜ)ᵀ · wᶜ can be computed efficiently
    as p⁰ * (pᶜ · wᶜ) to avoid materializing the n×n outer product matrix.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(
        self, x0: torch.Tensor, x_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x0: Original input p⁰, shape (batch, dim)
            x_prev: Previous cross layer output pᶜ, shape (batch, dim)

        Returns:
            Next cross layer output p^{c+1}, shape (batch, dim)
        """
        # Efficient computation: p⁰ * (pᶜ · wᶜ) + bᶜ + pᶜ
        # (pᶜ · wᶜ) is a scalar per sample → (batch, 1)
        cross_term = (x_prev * self.w).sum(dim=-1, keepdim=True)  # (batch, 1)
        # x0 * scalar + bias + residual
        out = x0 * cross_term + self.b + x_prev  # (batch, dim)
        return out


class CrossNetwork(nn.Module):
    """
    C-layer Cross Network for explicit high-order feature interactions.

    "The cross network applies feature crossing at each layer,
    the highest polynomial degree increases with layer depth."
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Input features p⁰, shape (batch, dim)

        Returns:
            Cross network output, shape (batch, dim)
        """
        x = x0
        for layer in self.layers:
            x = layer(x0, x)
        return x


class DeepNetwork(nn.Module):
    """
    Deep Network (MLP) for implicit nonlinear feature learning.

    "We feed p⁰ᵢ into a multilayer perceptron (MLP) to obtain
    the deep nonlinear features."
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features p⁰, shape (batch, dim)

        Returns:
            Deep network output, shape (batch, hidden_dim)
        """
        return self.network(x)


class HOFF(nn.Module):
    """
    Cross-Layer High-Order Feature Fusion module.

    Combines representations from all M network layers through
    a Deep & Cross Network architecture.

    Process (Eqs. 13-15):
        1. Concatenate: p⁰ᵢ = [h¹ᵢ ∥ h²ᵢ ∥ ... ∥ hᴹᵢ]        [Eq. 13]
        2. Cross Network: explicit high-order interactions       [Eq. 14]
        3. Deep Network: implicit nonlinear features
        4. Combine: zᵢ = W_comb · [p^C_i ∥ p^deep_i] + b_comb  [Eq. 15]

    "Cross-layer high-order feature fusion combines the strengths of
    both deep neural networks and cross feature interactions, making
    it highly effective for modeling complex relationships between
    features from different stock networks."
    """

    def __init__(
        self,
        num_networks: int = 4,
        per_network_dim: int = 64,
        cross_layers: int = 3,
        deep_layers: int = 2,
        deep_hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_networks: M — number of network types
            per_network_dim: Dimension of each network's representation
            cross_layers: C — number of cross network layers
            deep_layers: Number of deep network layers
            deep_hidden_dim: Hidden dimension of deep network
            output_dim: Final output dimension
            dropout: Dropout rate
        """
        super().__init__()

        concat_dim = num_networks * per_network_dim  # Eq. 13

        # Cross Network (Eq. 14)
        self.cross_network = CrossNetwork(concat_dim, cross_layers)

        # Deep Network
        self.deep_network = DeepNetwork(
            concat_dim, deep_hidden_dim, deep_layers, dropout
        )

        # Combination Layer (Eq. 15)
        # Concatenates cross output and deep output
        combination_input_dim = concat_dim + self.deep_network.output_dim
        self.combination = nn.Sequential(
            nn.Linear(combination_input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(
        self,
        network_representations: dict,
        network_names: list = None,
    ) -> torch.Tensor:
        """
        Fuse representations from multiple network layers.

        Args:
            network_representations: Dict mapping network_name ->
                stock representations (num_stocks, per_network_dim)
            network_names: Ordered list of network names

        Returns:
            Fused representation (num_stocks, output_dim)
        """
        if network_names is None:
            network_names = ["shareholding", "industry", "topicality", "comovement"]

        # Eq. 13: Concatenate all M hidden features
        # p⁰ᵢ = [h¹ᵢ ∥ h²ᵢ ∥ ... ∥ hᴹᵢ]
        representations = []
        for name in network_names:
            if name in network_representations:
                representations.append(network_representations[name])
            else:
                # If a network is missing, use zeros
                sample_repr = list(network_representations.values())[0]
                representations.append(
                    torch.zeros_like(sample_repr)
                )

        p0 = torch.cat(representations, dim=-1)  # (num_stocks, M * d)

        # Cross Network: explicit high-order interactions (Eq. 14)
        p_cross = self.cross_network(p0)  # (num_stocks, M * d)

        # Deep Network: implicit nonlinear features
        p_deep = self.deep_network(p0)  # (num_stocks, deep_hidden_dim)

        # Eq. 15: Combination layer
        combined = torch.cat([p_cross, p_deep], dim=-1)
        z = self.combination(combined)  # (num_stocks, output_dim)

        return z
