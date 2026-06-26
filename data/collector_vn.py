"""
Vietnam Stock Data Collector
============================
Fetches data for the top 100 Vietnamese stocks on HOSE exchange using
the `vnstock` library (free, no API key required, works from anywhere).

Mirrors the interface of StockDataCollector (data/collector.py) so it
plugs directly into the same preprocessing / graph-building pipeline.

Reference datasets:
    - vn100 : top ~100 liquid stocks on HOSE (default)
    - vn30  : official VN30 index (30 largest stocks)

Usage:
    from data.collector_vn import VietnamStockCollector
    collector = VietnamStockCollector(cache_dir="data/raw_vn")
    raw_data  = collector.collect_all(
        universe="vn100",
        start_date="2020-01-02",
        end_date="2024-06-30",
    )
"""

import os
import time
import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

try:
    from vnstock import Vnstock
    _VNSTOCK_AVAILABLE = True
except ImportError:
    _VNSTOCK_AVAILABLE = False
    Vnstock = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static constituent lists
# ---------------------------------------------------------------------------

# VN30 — official index (30 largest HOSE stocks by market cap & liquidity)
VN30_CODES: List[str] = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

# VN Top 100 — top 100 liquid / large-cap stocks on HOSE (as of 2024)
VN100_CODES: List[str] = [
    # Banking (20)
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "SHB",
    "VIB", "TPB", "LPB", "OCB", "SSB", "EIB", "ABB", "BVB", "KLB", "NAB",
    # Real estate (20)
    "VHM", "VIC", "VRE", "NVL", "PDR", "KDH", "DXG", "NLG", "DIG", "CII",
    "IDC", "KBC", "HDG", "SZC", "TDH", "AGG", "IJC", "BCM", "TCH", "PHR",
    # Consumer / Retail (20)
    "VNM", "MWG", "SAB", "MSN", "PNJ", "QNS", "VHC", "ANV", "FMC", "PAN",
    "DBC", "BAF", "HAG", "GTN", "TNG", "MSH", "VFC", "TRA", "RAL", "VCF",
    # Energy / Industrial (20)
    "GAS", "PLX", "BSR", "POW", "GEG", "VSH", "REE", "PC1", "PVT", "PVD",
    "DPM", "DCM", "GEX", "VGC", "HT1", "KSB", "HPG", "DPR", "TV2", "BWE",
    # Technology / Securities / Insurance (10)
    "FPT", "SSI", "HCM", "VCI", "VND", "BVH", "MIG", "CMG", "ELC", "VDS",
    # Transport / Logistics / Misc (10)
    "VJC", "GMD", "VOS", "SCS", "VPI", "NCT", "SKG", "NKG", "GVR", "HSG",
]
# De-duplicate while preserving order
_seen: set = set()
VN100_CODES = [c for c in VN100_CODES if not (c in _seen or _seen.add(c))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Collector class
# ---------------------------------------------------------------------------

class VietnamStockCollector:
    """
    Collects raw stock data for Vietnamese markets (HOSE).

    Supports:
        - vn30  (30 stocks)
        - vn100 (top ~100 stocks)  <- default

    All data is cached locally as CSV / Parquet files to avoid re-fetching.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw_vn",
        source: str = "VCI",   # vnstock data source: "VCI" (default) or "TCBS"
    ):
        if not _VNSTOCK_AVAILABLE:
            raise ImportError(
                "vnstock is required. Install with: pip install vnstock"
            )
        self.cache_dir = cache_dir
        self.source = source
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Constituent List
    # ------------------------------------------------------------------

    def get_constituents(self, universe: str = "vn100") -> pd.DataFrame:
        """
        Return the list of stocks to include.

        Args:
            universe: "vn30" or "vn100"

        Returns:
            DataFrame with columns: [code]
        """
        cache_path = os.path.join(self.cache_dir, f"{universe}_constituents.csv")
        if os.path.exists(cache_path):
            logger.info(f"Loading {universe} constituents from cache.")
            return pd.read_csv(cache_path, dtype={"code": str})

        codes = VN30_CODES if universe == "vn30" else VN100_CODES

        # Try to enrich with names via vnstock listing
        try:
            listing = Vnstock().stock(source=self.source).listing.all_symbols()
            listing.columns = [c.lower() for c in listing.columns]
            sym_col = next(
                (c for c in listing.columns if c in ("symbol", "ticker")), listing.columns[0]
            )
            name_col = next(
                (c for c in listing.columns if "name" in c.lower() or "organ" in c.lower()),
                None,
            )
            listing = listing.rename(columns={sym_col: "code"})
            listing["code"] = listing["code"].str.upper()
            filtered = listing[listing["code"].isin(codes)].copy()
            if name_col:
                filtered = filtered.rename(columns={name_col: "name"})
                result = filtered[["code", "name"]].reset_index(drop=True)
            else:
                result = pd.DataFrame({"code": codes})
        except Exception as e:
            logger.warning(f"Could not fetch live listing ({e}). Using static list.")
            result = pd.DataFrame({"code": codes})

        result.to_csv(cache_path, index=False)
        logger.info(f"{universe.upper()} constituent list: {len(result)} stocks.")
        return result

    # ------------------------------------------------------------------
    # 2. Daily OHLCV Price Data
    # ------------------------------------------------------------------

    def fetch_stock_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV for a single stock via vnstock.

        Args:
            symbol:     Stock ticker e.g. "VCB"
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            DataFrame with columns [date, stock_code, open, high, low, close, volume]
            or None on failure.
        """
        try:
            stock = Vnstock().stock(symbol=symbol, source=self.source)
            df = stock.quote.history(
                start=start_date,
                end=end_date,
                interval="1D",
            )
            if df is None or df.empty:
                logger.debug(f"No data returned for {symbol}.")
                return None

            df.columns = [c.lower() for c in df.columns]

            # Normalise date column name (varies across vnstock versions)
            time_col = next(
                (c for c in df.columns if c in ("time", "date", "tradingdate")),
                df.columns[0],
            )
            df = df.rename(columns={time_col: "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["stock_code"] = symbol

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' missing for {symbol}.")
                    return None
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "close"])
            df = df.sort_values("date").reset_index(drop=True)
            return df[["date", "stock_code", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return None

    def fetch_all_stocks_daily(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV for all stocks with local caching.

        Args:
            stock_codes: List of ticker symbols
            start_date:  "YYYY-MM-DD"
            end_date:    "YYYY-MM-DD"
            delay:       Seconds between requests (rate-limiting)

        Returns:
            Combined DataFrame of all stocks.
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"daily_prices_{start_date}_{end_date}.parquet",
        )
        if os.path.exists(cache_path):
            logger.info("Loading daily prices from cache.")
            return pd.read_parquet(cache_path)

        all_data = []
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"Fetching price data {i + 1}/{total}: {code}")

            df = self.fetch_stock_daily(code, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
            time.sleep(delay)

        if not all_data:
            raise RuntimeError(
                "No stock data could be fetched! "
                "Check your internet connection and vnstock version."
            )

        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        logger.info(
            f"Fetched data for {len(all_data)}/{total} stocks, "
            f"{len(combined):,} total records."
        )
        return combined

    # ------------------------------------------------------------------
    # 3. Industry Classification
    # ------------------------------------------------------------------

    def fetch_industry_classification(
        self,
        stock_codes: List[str],
    ) -> pd.DataFrame:
        """
        Fetch ICB industry classification for all stocks via vnstock.

        Tries the listing API first (fast), then falls back to per-stock
        company overview (slow but more detailed).

        Returns:
            DataFrame with columns: [stock_code, industry_name]
        """
        cache_path = os.path.join(self.cache_dir, "industry_classification.csv")
        if os.path.exists(cache_path):
            logger.info("Loading industry classification from cache.")
            return pd.read_csv(cache_path, dtype={"stock_code": str})

        logger.info("Fetching industry classification from vnstock...")

        # Method 1: listing API (has ICB codes in most vnstock versions)
        try:
            listing = Vnstock().stock(source=self.source).listing.all_symbols()
            listing.columns = [c.lower() for c in listing.columns]
            sym_col = next(
                (c for c in listing.columns if c in ("symbol", "ticker")),
                listing.columns[0],
            )
            ind_col = next(
                (
                    c for c in listing.columns
                    if any(kw in c for kw in ("industry", "sector", "icb", "nganh"))
                ),
                None,
            )
            if ind_col:
                listing = listing.rename(
                    columns={sym_col: "stock_code", ind_col: "industry_name"}
                )
                listing["stock_code"] = listing["stock_code"].str.upper()
                result = (
                    listing[listing["stock_code"].isin(stock_codes)][
                        ["stock_code", "industry_name"]
                    ]
                    .dropna()
                    .drop_duplicates("stock_code")
                    .reset_index(drop=True)
                )
                if len(result) > 0:
                    result.to_csv(cache_path, index=False)
                    logger.info(
                        f"Industry from listing API: {len(result)} stocks."
                    )
                    return result
        except Exception as e:
            logger.warning(f"Listing-based industry fetch failed: {e}")

        # Method 2: per-stock company overview
        logger.info("Falling back to per-stock company overview...")
        rows = []
        for i, code in enumerate(stock_codes):
            if (i + 1) % 20 == 0:
                logger.info(f"  Industry: {i + 1}/{len(stock_codes)}")
            industry = "Unknown"
            try:
                stock = Vnstock().stock(symbol=code, source=self.source)
                overview = stock.company.overview()
                if overview is not None and not overview.empty:
                    overview.columns = [c.lower() for c in overview.columns]
                    ind_col = next(
                        (
                            c for c in overview.columns
                            if any(kw in c for kw in ("industry", "sector", "icb"))
                        ),
                        None,
                    )
                    if ind_col:
                        industry = str(overview[ind_col].iloc[0])
                time.sleep(0.3)
            except Exception as e:
                logger.debug(f"Overview failed for {code}: {e}")
            rows.append({"stock_code": code, "industry_name": industry})

        result = pd.DataFrame(rows)
        result.to_csv(cache_path, index=False)
        logger.info(f"Fetched industry for {len(result)} stocks.")
        return result

    # ------------------------------------------------------------------
    # 4. Shareholding & News (placeholders — roadmap items)
    # ------------------------------------------------------------------

    def fetch_shareholding_data(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        Returns empty DataFrame.
        Shareholding data for Vietnam is not yet implemented.
        The Shareholding network layer will be disabled (empty graph).
        """
        cache_path = os.path.join(self.cache_dir, "shareholding.csv")
        if os.path.exists(cache_path):
            return pd.read_csv(
                cache_path, dtype={"held_code": str, "holder_name": str}
            )
        result = pd.DataFrame(columns=["held_code", "holder_name", "ratio"])
        result.to_csv(cache_path, index=False)
        logger.warning(
            "Vietnam shareholding data not implemented. "
            "Shareholding network will be empty."
        )
        return result

    def fetch_financial_news(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Returns empty DataFrame.
        News data for Vietnam is not yet implemented.
        The Topicality network layer will be disabled (empty graph).
        """
        cache_path = os.path.join(
            self.cache_dir, f"news_{start_date}_{end_date}.parquet"
        )
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        result = pd.DataFrame(columns=["date", "stock_code", "title", "content"])
        result.to_parquet(cache_path, index=False)
        logger.warning(
            "Vietnam news data not implemented. "
            "Topicality network will be empty."
        )
        return result

    # ------------------------------------------------------------------
    # 5. Master Collection Function
    # ------------------------------------------------------------------

    def collect_all(
        self,
        universe: str = "vn100",
        start_date: str = "2020-01-02",
        end_date: str = "2024-06-30",
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data for the Vietnam stock experiment.

        Args:
            universe:   "vn30" or "vn100"
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            Dict with keys: constituents, daily_prices, industry,
                            shareholding, news
        """
        logger.info(
            f"Collecting VN {universe.upper()} | {start_date} → {end_date}"
        )

        constituents = self.get_constituents(universe)
        stock_codes = constituents["code"].tolist()
        logger.info(f"Universe: {len(stock_codes)} stocks")

        daily_prices = self.fetch_all_stocks_daily(stock_codes, start_date, end_date)
        industry = self.fetch_industry_classification(stock_codes)
        shareholding = self.fetch_shareholding_data(stock_codes)
        news = self.fetch_financial_news(stock_codes, start_date, end_date)

        return {
            "constituents": constituents,
            "daily_prices": daily_prices,
            "industry": industry,
            "shareholding": shareholding,
            "news": news,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = VietnamStockCollector(cache_dir="data/raw_vn")
    data = collector.collect_all(
        universe="vn100",
        start_date="2022-01-02",
        end_date="2024-06-30",
    )
    for key, df in data.items():
        print(f"{key:15s}: {df.shape}")
