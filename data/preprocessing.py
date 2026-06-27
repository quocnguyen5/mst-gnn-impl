"""
Data Preprocessing
==================
Computes the 13 stock features described in Section IV-A of the paper:
  - 5 raw OHLCV features
  - 8 derivative features (price change, pct change, moving averages)

Also handles:
  - Missing data imputation
  - Feature normalization (per-stock z-score)
  - Label generation (movement direction + return)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StockPreprocessor:
    """Preprocess raw OHLCV data into the 13-feature representation."""

    def __init__(
        self,
        lookback_window: int = 5,
        prediction_horizon: int = 1,
        ma_windows: List[int] = None,
    ):
        """
        Args:
            lookback_window: T in paper — number of historical days for input
            prediction_horizon: s in paper — predict s days ahead
            ma_windows: moving average windows (default: [5, 10, 20])
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.ma_windows = ma_windows or [5, 10, 20]

    def compute_features(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 13 features for each stock on each trading day.

        Input columns: [date, stock_code, open, high, low, close, volume]
        Output adds: [price_change, pct_change, close_ma5, close_ma10,
                       close_ma20, vol_ma5, vol_ma10, vol_ma20]

        Reference: Section IV-A
            "five daily volume-price indicators: open, high, low, and close
            prices, and volume. They also include eight derivative indicators:
            daily price change, percentage of daily price change, and
            5-day/10-day/20-day moving averages of both the close price
            and volume"
        """
        logger.info("Computing 13 stock features...")
        df = daily_prices.copy()
        df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

        feature_dfs = []
        for code, group in df.groupby("stock_code"):
            group = group.sort_values("date").copy()

            # --- Derivative Feature 1: Daily price change ---
            group["price_change"] = group["close"].diff()

            # --- Derivative Feature 2: Percentage of daily price change ---
            group["pct_change"] = group["close"].pct_change() * 100

            # --- Derivative Features 3-5: Moving averages of close price ---
            for w in self.ma_windows:
                group[f"close_ma{w}"] = (
                    group["close"].rolling(window=w, min_periods=1).mean()
                )

            # --- Derivative Features 6-8: Moving averages of volume ---
            for w in self.ma_windows:
                group[f"vol_ma{w}"] = (
                    group["volume"].rolling(window=w, min_periods=1).mean()
                )

            feature_dfs.append(group)

        result = pd.concat(feature_dfs, ignore_index=True)

        # Drop rows where derivative features can't be computed
        result = result.dropna(
            subset=["price_change", "pct_change"]
        ).reset_index(drop=True)

        logger.info(
            f"Feature computation complete: {result.shape[0]} records, "
            f"{len(result['stock_code'].unique())} stocks."
        )
        return result

    def compute_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute prediction labels:
        1. Movement direction (Eq. 1): binary, 1 if price goes up, 0 otherwise
        2. Return ratio (Eq. 2): (close_{t+s} - close_t) / close_t

        Reference: Eqs. (1)-(3) in the paper.
        """
        logger.info("Computing prediction labels...")
        df = feature_df.copy()
        label_dfs = []

        for code, group in df.groupby("stock_code"):
            group = group.sort_values("date").copy()
            s = self.prediction_horizon

            # Future close price (s days ahead)
            group["future_close"] = group["close"].shift(-s)

            # Return: r_i,t = (close_{t+s} - close_t) / close_t  [Eq. 2]
            group["return"] = (
                (group["future_close"] - group["close"]) / group["close"]
            )

            # Movement: y_i,t = 1 if close_{t+s} > close_t else 0  [Eq. 1]
            group["movement"] = (group["future_close"] > group["close"]).astype(
                int
            )

            label_dfs.append(group)

        result = pd.concat(label_dfs, ignore_index=True)

        # Drop rows without labels (last s days of each stock)
        result = result.dropna(subset=["return", "movement"]).reset_index(
            drop=True
        )

        logger.info(f"Label computation complete: {result.shape[0]} labeled records.")
        return result

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        method: str = "zscore",
    ) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
        """
        Normalize features per-stock using z-score normalization.

        We normalize within the training period and apply the same
        transformation to validation and test sets (done in dataset.py).

        Args:
            df: DataFrame with features
            feature_cols: columns to normalize
            method: "zscore" (default) or "minmax"

        Returns:
            Tuple of (normalized DataFrame, dict of fitted scalers per stock)
        """
        if feature_cols is None:
            feature_cols = [
                "open", "high", "low", "close", "volume",
                "price_change", "pct_change",
                "close_ma5", "close_ma10", "close_ma20",
                "vol_ma5", "vol_ma10", "vol_ma20",
            ]

        logger.info(f"Normalizing {len(feature_cols)} features with {method}...")
        df_norm = df.copy()
        scalers = {}

        for code, group in df_norm.groupby("stock_code"):
            scaler = StandardScaler()
            normalized = scaler.fit_transform(group[feature_cols].values)
            # Handle NaN/inf from zero-std columns (stock price didn't change)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            df_norm.loc[group.index, feature_cols] = normalized
            scalers[code] = scaler

        return df_norm, scalers

    def create_sliding_windows(
        self, df: pd.DataFrame, feature_cols: List[str] = None
    ) -> Tuple[Dict, List[str]]:
        """
        Create sliding window samples for each stock on each date.

        For stock i at time t, the input is x_{i,t} ∈ R^{T×d}
        (T historical days, d features).

        Returns:
            Dict mapping (stock_code, date) -> {
                "features": np.array of shape (T, d),
                "movement": int,
                "return": float,
            }
            List of sorted trading dates
        """
        if feature_cols is None:
            feature_cols = [
                "open", "high", "low", "close", "volume",
                "price_change", "pct_change",
                "close_ma5", "close_ma10", "close_ma20",
                "vol_ma5", "vol_ma10", "vol_ma20",
            ]

        T = self.lookback_window
        samples = {}
        all_dates = sorted(df["date"].unique())

        for code, group in df.groupby("stock_code"):
            group = group.sort_values("date").reset_index(drop=True)
            dates = group["date"].values
            features = group[feature_cols].values
            movements = group["movement"].values
            returns = group["return"].values

            for idx in range(T - 1, len(group)):
                date = dates[idx]
                window = features[idx - T + 1 : idx + 1]  # (T, d)

                if window.shape[0] == T:
                    samples[(code, date)] = {
                        "features": window.astype(np.float32),
                        "movement": int(movements[idx]),
                        "return": float(returns[idx]),
                    }

        logger.info(f"Created {len(samples)} sliding window samples.")
        return samples, all_dates

    def get_trading_dates(self, df: pd.DataFrame) -> List:
        """Get sorted list of unique trading dates."""
        return sorted(df["date"].unique())

    def get_active_stocks(
        self, df: pd.DataFrame, date
    ) -> List[str]:
        """
        Get list of stocks actively trading on a given date.
        Handles the dynamic stock set described in the paper:
        "We do not assume that the set of investigated stocks in a
        financial market is fixed, i.e., the number of stocks changes
        over time."
        """
        mask = df["date"] == date
        return sorted(df.loc[mask, "stock_code"].unique().tolist())

    def process_pipeline(
        self, daily_prices: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict, List]:
        """
        Full preprocessing pipeline.

        Args:
            daily_prices: Raw OHLCV DataFrame

        Returns:
            - processed_df: DataFrame with features, labels, normalization
            - samples: Dict mapping (stock, date) -> sample data
            - trading_dates: Sorted list of trading dates
        """
        # Step 1: Compute 13 features
        featured = self.compute_features(daily_prices)

        # Step 2: Compute labels
        labeled = self.compute_labels(featured)

        # Step 3: Normalize features
        normalized, scalers = self.normalize_features(labeled)

        # Step 4: Create sliding windows
        samples, trading_dates = self.create_sliding_windows(normalized)

        return normalized, samples, trading_dates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage with dummy data
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    dummy = pd.DataFrame(
        {
            "date": np.tile(dates, 3),
            "stock_code": np.repeat(["000001", "000002", "000003"], len(dates)),
            "open": np.random.randn(3 * len(dates)) * 10 + 100,
            "high": np.random.randn(3 * len(dates)) * 10 + 105,
            "low": np.random.randn(3 * len(dates)) * 10 + 95,
            "close": np.random.randn(3 * len(dates)) * 10 + 100,
            "volume": np.random.randint(1000, 100000, 3 * len(dates)),
        }
    )
    preprocessor = StockPreprocessor()
    processed, samples, dates_list = preprocessor.process_pipeline(dummy)
    print(f"Processed: {processed.shape}")
    print(f"Samples: {len(samples)}")
    print(f"Trading dates: {len(dates_list)}")
