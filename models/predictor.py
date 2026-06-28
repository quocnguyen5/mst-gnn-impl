"""
Multitask Prediction Head — Module D
=====================================
Joint prediction of stock movement direction and return ranking.

Reference: Section IV-E of the paper (Eqs. 17-19)

"Predicting stocks for profitable investment requires considering
the stock price movement direction and identifying stocks with
high expected returns."

"We propose a two-task prediction module that includes two-class
stock movement classification and stock return ranking."

Loss function (Eq. 17):
    L = δ·L_move + (1-δ)·L_rank + c·||Θ||²
where:
    L_move: binary cross-entropy loss (Eq. 18)
    L_rank: pairwise ranking loss (Eq. 19)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovementPredictor(nn.Module):
    """
    Binary movement classification head.

    Predicts whether each stock's price will go up (1) or down (0).

    L_move = -Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]  [Eq. 18]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Fused stock representations (num_stocks, input_dim)

        Returns:
            Movement logits (num_stocks, num_classes)
        """
        return self.classifier(z)


class RankingPredictor(nn.Module):
    """
    Stock return ranking head.

    Produces a ranking score for each stock to identify
    high-return stocks.

    L_rank uses pairwise comparison (Eq. 19).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Fused stock representations (num_stocks, input_dim)

        Returns:
            Ranking scores (num_stocks, 1)
        """
        return self.scorer(z)


class MultitaskLoss(nn.Module):
    """
    Combined multitask loss function.

    Eq. (17):
        L = δ·L_move + (1-δ)·L_rank + c·||Θ||²

    L_move: Binary cross-entropy for movement classification (Eq. 18)
    L_rank: Pairwise ranking loss (Eq. 19)

    "Multitask prediction is an approach to inductive transfer that
    enhances generalization by utilizing the domain information
    contained in the training signals of related tasks as an
    inductive bias."
    """

    def __init__(self, delta: float = 0.5, margin: float = 0.1, class_weight: list = None):
        """
        Args:
            delta: Weight for movement loss (1-delta for ranking)
            margin: Margin for pairwise ranking loss
            class_weight: Optional [w_down, w_up] weights for CrossEntropy.
                         Use to counteract class imbalance and prevent
                         model collapse to majority class.
        """
        super().__init__()
        self.delta = delta
        self.margin = margin
        if class_weight is not None:
            self.register_buffer(
                "class_weight", torch.tensor(class_weight, dtype=torch.float32)
            )
            self.ce_loss = None  # will use F.cross_entropy with weights
        else:
            self.class_weight = None
            self.ce_loss = nn.CrossEntropyLoss()

    def movement_loss(
        self,
        movement_logits: torch.Tensor,
        movement_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. (18): Binary cross-entropy loss for movement prediction.

        Args:
            movement_logits: (num_stocks, 2)
            movement_labels: (num_stocks,) with values {0, 1}

        Returns:
            Scalar loss
        """
        if self.class_weight is not None:
            weight = self.class_weight.to(movement_logits.device)
            return F.cross_entropy(movement_logits, movement_labels, weight=weight)
        return self.ce_loss(movement_logits, movement_labels)

    def ranking_loss(
        self,
        ranking_scores: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. (19): Pairwise ranking loss.

        For each pair (i, j) where return_i > return_j, we want
        score_i > score_j + margin.

        L_rank = Σ_{(i,j): r_i > r_j} max(0, -(score_i - score_j) + margin)

        Args:
            ranking_scores: (num_stocks, 1)
            returns: (num_stocks,) actual return values

        Returns:
            Scalar loss
        """
        scores = ranking_scores.squeeze(-1)  # (num_stocks,)
        n = scores.size(0)

        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Create all pairs (i, j) where return_i > return_j
        # Score differences: score_i - score_j
        score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (n, n)
        return_diff = returns.unsqueeze(0) - returns.unsqueeze(1)  # (n, n)

        # Mask for pairs where return_i > return_j
        valid_pairs = (return_diff > 0).float()

        # Hinge loss: max(0, -(score_i - score_j) + margin)
        pair_loss = F.relu(-score_diff + self.margin)

        # Apply mask and average
        loss = (pair_loss * valid_pairs).sum()
        num_valid = valid_pairs.sum().clamp(min=1)
        loss = loss / num_valid

        return loss

    def forward(
        self,
        movement_logits: torch.Tensor,
        movement_labels: torch.Tensor,
        ranking_scores: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple:
        """
        Compute combined multitask loss.

        Args:
            movement_logits: (num_stocks, 2) classification logits
            movement_labels: (num_stocks,) binary labels
            ranking_scores: (num_stocks, 1) ranking scores
            returns: (num_stocks,) actual return values

        Returns:
            Tuple of (total_loss, move_loss, rank_loss)
        """
        l_move = self.movement_loss(movement_logits, movement_labels)
        l_rank = self.ranking_loss(ranking_scores, returns)

        # Eq. 17: L = δ·L_move + (1-δ)·L_rank
        # Note: L2 regularization (c·||Θ||²) is handled by optimizer weight_decay
        total = self.delta * l_move + (1 - self.delta) * l_rank

        return total, l_move, l_rank


class MultitaskPredictor(nn.Module):
    """
    Module D: Combined multitask prediction head.

    Contains both movement classification and ranking prediction heads.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.movement_head = MovementPredictor(input_dim, hidden_dim, num_classes)
        self.ranking_head = RankingPredictor(input_dim, hidden_dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple:
        """
        Args:
            z: Fused stock representations (num_stocks, input_dim)

        Returns:
            Tuple of:
            - movement_logits: (num_stocks, num_classes)
            - ranking_scores: (num_stocks, 1)
        """
        movement_logits = self.movement_head(z)
        ranking_scores = self.ranking_head(z)
        return movement_logits, ranking_scores
