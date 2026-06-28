"""
Spatial-Temporal Neighborhood Aggregation (STNA) — Module B
===========================================================
Aggregates information from spatial neighbors (same-time graph connections)
and temporal neighbors (self-connection from previous timestep).

Reference: Section IV-C of the paper (Eqs. 8-12)

Key insight from paper:
    "No longer needs RNN but can directly use graph convolution to model
    the spatial and temporal relations. Therefore, it avoids the problems
    of high complexity and gradient vanishing/explosion."

1-Hop Spatial-Temporal Neighborhood (Eq. 8):
    N_q(s_i, t) = {s_j, e_{j,i} ∈ E_{t,q}} ∪ {s_i, s_i ∈ S_{t-1}}

Three aggregator types:
    - Mean (Eqs. 9-10)
    - LSTM (Eq. 11)
    - Max-pooling (Eq. 12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


class MeanAggregator(nn.Module):
    """
    Mean aggregator for STNA.

    Eqs. (9)-(10):
        h̄_{N_q(s_i)} = MEAN({h_j, s_j ∈ N_q(s_i, t)})     [Eq. 9]
        h_i^k = σ(W · CONCAT(h_i^{k-1}, h̄_{N_q(s_i)}))     [Eq. 10]
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # W in Eq. 10: transforms concatenated self + neighbor features
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional

        Returns:
            Aggregated features (num_nodes, output_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Weighted neighbor messages
        messages = x[src]  # (num_edges, input_dim)
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)

        # Mean aggregation: scatter and normalize
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Count neighbors for normalization
        deg = degree(dst, num_nodes=num_nodes).clamp(min=1)
        agg = agg / deg.unsqueeze(-1)

        # Eq. 10: concat self and neighbor, then transform
        combined = torch.cat([x, agg], dim=-1)  # (num_nodes, 2*input_dim)
        out = F.relu(self.linear(combined))

        return out


class LSTMAggregator(nn.Module):
    """
    LSTM aggregator for STNA.

    Eq. (11):
        "we randomly permute the node's spatial-temporal neighbors
        and apply the LSTM network to the unordered set"

    Has larger model capacity than mean aggregator.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim,
            batch_first=True,
        )
        self.linear = nn.Linear(input_dim + output_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional

        Returns:
            Aggregated features (num_nodes, output_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        output_dim = self.lstm.hidden_size

        agg_out = torch.zeros(num_nodes, output_dim, device=x.device)

        for node_idx in range(num_nodes):
            # Get neighbors of this node
            mask = dst == node_idx
            neighbor_indices = src[mask]

            if neighbor_indices.numel() == 0:
                continue

            # Random permutation of neighbors
            perm = torch.randperm(neighbor_indices.numel(), device=x.device)
            neighbor_features = x[neighbor_indices[perm]]  # (num_neighbors, dim)

            if edge_weight is not None:
                neighbor_weights = edge_weight[mask][perm].unsqueeze(-1)
                neighbor_features = neighbor_features * neighbor_weights

            # LSTM over randomly permuted neighbor sequence
            neighbor_seq = neighbor_features.unsqueeze(0)  # (1, num_neighbors, dim)
            _, (h_n, _) = self.lstm(neighbor_seq)
            agg_out[node_idx] = h_n.squeeze(0).squeeze(0)

        # Combine self and aggregated neighbor features
        combined = torch.cat([x, agg_out], dim=-1)
        out = F.relu(self.linear(combined))

        return out


class MaxPoolAggregator(nn.Module):
    """
    Max-pooling aggregator for STNA.

    Eq. (12):
        "a tradeoff between trainability and symmetry"
        "trainable by feeding each neighbor's hidden vector into a
        fully-connected network. Following the network, an elementwise
        max-pooling is exploited"
    """

    def __init__(self, input_dim: int, output_dim: int, pool_dim: int = None):
        super().__init__()
        pool_dim = pool_dim or output_dim

        # FC applied to each neighbor before pooling
        self.fc_neighbor = nn.Sequential(
            nn.Linear(input_dim, pool_dim),
            nn.ReLU(),
        )
        # Transform concatenated self + pooled features
        self.linear = nn.Linear(input_dim + pool_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional

        Returns:
            Aggregated features (num_nodes, output_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        pool_dim = self.fc_neighbor[0].out_features

        # Transform all source node features through FC
        src_features = x[src]  # (num_edges, input_dim)
        if edge_weight is not None:
            src_features = src_features * edge_weight.unsqueeze(-1)
        transformed = self.fc_neighbor(src_features)  # (num_edges, pool_dim)

        # Element-wise max pooling per destination node
        pooled = torch.full(
            (num_nodes, pool_dim), float("-inf"), device=x.device
        )
        pooled.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand_as(transformed),
            transformed,
            reduce="amax",
            include_self=False,
        )
        # Replace -inf with 0 for nodes with no neighbors
        pooled = pooled.clamp(min=0)

        # Combine self and pooled features
        combined = torch.cat([x, pooled], dim=-1)
        out = F.relu(self.linear(combined))

        return out


class STNA(nn.Module):
    """
    Spatial-Temporal Neighborhood Aggregation module.

    For each network layer q, performs K-depth aggregation that
    simultaneously captures:
    1. Spatial neighbors from the current graph G_{t,q}
    2. Temporal self-connection from the previous timestep

    "This mechanism aggregates spatial neighborhood information from
    the current network and temporal information from the previous
    stock network."

    The STNA operates independently per network layer to
    "retain the corresponding financial implications."
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 2,
        aggregator_type: str = "mean",
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Input feature dimension (d1 from feature encoder)
            hidden_dim: Hidden dimension in aggregation
            depth: K — aggregation depth
            aggregator_type: "mean", "lstm", or "maxpool"
            dropout: Dropout rate
        """
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        # Gated temporal connection (GRU-like)
        self.temporal_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )
        self.temporal_transform = nn.Linear(input_dim, input_dim)

        # Build K aggregation layers
        self.aggregators = nn.ModuleList()
        for k in range(depth):
            in_dim = input_dim if k == 0 else hidden_dim
            out_dim = hidden_dim

            if aggregator_type == "mean":
                agg = MeanAggregator(in_dim, out_dim)
            elif aggregator_type == "lstm":
                agg = LSTMAggregator(in_dim, out_dim)
            elif aggregator_type == "maxpool":
                agg = MaxPoolAggregator(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown aggregator type: {aggregator_type}")

            self.aggregators.append(agg)

        # Layer norms for each depth
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        )

    def _build_st_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple:
        """
        Build spatial-temporal neighborhood edges (Eq. 8).

        N_q(s_i, t) = {s_j, e_{j,i} ∈ E_{t,q}} ∪ {s_i, s_i ∈ S_{t-1}}

        The temporal connection is handled via gated hidden state fusion.
        Self-loops model within-timestep self-connection.
        """
        if edge_index.numel() == 0:
            # No edges — create only self-loops
            self_loops = torch.arange(num_nodes, device=edge_index.device)
            edge_index = torch.stack([self_loops, self_loops])
            edge_weight = torch.ones(num_nodes, device=edge_index.device)
        else:
            # Add self-loops for self-connection
            edge_index, edge_weight = add_self_loops(
                edge_index,
                edge_weight,
                fill_value=1.0,
                num_nodes=num_nodes,
            )

        return edge_index, edge_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        prev_hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        K-depth spatial-temporal neighborhood aggregation.

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Current graph edges (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            prev_hidden: Previous timestep hidden states (num_nodes, input_dim)
                        Already aligned by stock code (handled in MSTGNN).

        Returns:
            Aggregated features (num_nodes, hidden_dim)
        """
        num_nodes = x.size(0)

        # Build spatial edge index with self-loops
        st_edge_index, st_edge_weight = self._build_st_edge_index(
            edge_index, edge_weight, num_nodes
        )

        # Gated temporal connection (Eq. 8 — temporal part)
        h = x
        if prev_hidden is not None:
            # GRU-like gate: learn how much temporal info to keep
            gate_input = torch.cat([h, prev_hidden], dim=-1)  # (n, 2*dim)
            gate = self.temporal_gate(gate_input)  # (n, dim), values in [0,1]
            temporal_info = self.temporal_transform(prev_hidden)
            h = gate * temporal_info + (1 - gate) * h

        # K-depth spatial aggregation with residual connections
        h_input = h  # save for skip connection
        for k in range(self.depth):
            h_prev = h
            h = self.aggregators[k](h, st_edge_index, st_edge_weight)
            h = self.layer_norms[k](h)
            # Residual connection: preserve individual stock info
            if h.size(-1) == h_prev.size(-1):
                h = h + h_prev
            h = self.dropout(h)

        # Final skip from input (before aggregation) to preserve stock identity
        if h.size(-1) == h_input.size(-1):
            h = h + h_input

        return h


class MultiLayerSTNA(nn.Module):
    """
    STNA applied independently to each of the M network layers.

    "Following the spatial-temporal neighborhood aggregation, we can
    obtain the corresponding stock representations for each stock
    network to retain the corresponding financial implications."
    """

    def __init__(
        self,
        num_layers: int = 4,
        input_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 2,
        aggregator_type: str = "mean",
        dropout: float = 0.3,
    ):
        """
        Args:
            num_layers: M — number of network types (4 in paper)
            input_dim: Feature dimension from encoder
            hidden_dim: STNA hidden dimension
            depth: K — aggregation depth
            aggregator_type: Aggregator type
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers

        # Independent STNA for each network layer
        self.stna_layers = nn.ModuleList(
            [
                STNA(input_dim, hidden_dim, depth, aggregator_type, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        networks: dict,
        prev_hiddens: dict = None,
        network_names: list = None,
    ) -> dict:
        """
        Apply STNA to each network layer independently.

        Args:
            x: Encoded stock features (num_stocks, input_dim)
            networks: Dict mapping network_name -> (edge_index, edge_weight)
            prev_hiddens: Dict mapping network_name -> previous hidden states
            network_names: Ordered list of network names

        Returns:
            Dict mapping network_name -> aggregated features (num_stocks, hidden_dim)
        """
        if network_names is None:
            network_names = ["shareholding", "industry", "topicality", "comovement"]

        results = {}
        for idx, name in enumerate(network_names):
            if idx >= self.num_layers:
                break

            edge_index, edge_weight = networks.get(
                name,
                (
                    torch.zeros(2, 0, dtype=torch.long, device=x.device),
                    torch.zeros(0, device=x.device),
                ),
            )

            prev_h = None
            if prev_hiddens is not None and name in prev_hiddens:
                prev_h = prev_hiddens[name]

            results[name] = self.stna_layers[idx](
                x, edge_index, edge_weight, prev_h
            )

        return results
