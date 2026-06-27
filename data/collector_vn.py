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
from tqdm import tqdm

# ── vnstock API: try new (4.x) first, fall back to old (3.x) ────────────────
try:
    from vnstock.api.quote import Quote as _VnQuote  # new API (vnstock >= 4.x)
    _VNSTOCK_NEW_API = True
    _VNSTOCK_AVAILABLE = True
except ImportError:
    _VnQuote = None
    _VNSTOCK_NEW_API = False

if not _VNSTOCK_NEW_API:
    try:
        from vnstock import Vnstock  # old API (vnstock 3.x)
        _VNSTOCK_AVAILABLE = True
    except ImportError:
        _VNSTOCK_AVAILABLE = False
        Vnstock = None
else:
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

# Static industry classification fallback (used when vnstock API returns None)
# Source: HOSE sector classification + ICB standard (2024)
VN_INDUSTRY_STATIC: dict = {
    # Banking
    "ACB": "Banking", "BID": "Banking", "CTG": "Banking", "EIB": "Banking",
    "HDB": "Banking", "KLB": "Banking", "LPB": "Banking", "MBB": "Banking",
    "NAB": "Banking", "OCB": "Banking", "SHB": "Banking", "SSB": "Banking",
    "STB": "Banking", "TCB": "Banking", "TPB": "Banking", "VCB": "Banking",
    "VIB": "Banking", "VPB": "Banking", "ABB": "Banking", "BVB": "Banking",
    # Real Estate
    "AGG": "Real Estate", "BCM": "Real Estate", "CII": "Real Estate",
    "DIG": "Real Estate", "DXG": "Real Estate", "HDG": "Real Estate",
    "IDC": "Real Estate", "IJC": "Real Estate", "KBC": "Real Estate",
    "KDH": "Real Estate", "NLG": "Real Estate", "NVL": "Real Estate",
    "PDR": "Real Estate", "PHR": "Real Estate", "SZC": "Real Estate",
    "TCH": "Real Estate", "TDH": "Real Estate", "VHM": "Real Estate",
    "VIC": "Real Estate", "VRE": "Real Estate",
    # Insurance & Financial Services
    "BVH": "Insurance", "MIG": "Insurance",
    "HCM": "Financial Services", "SSI": "Financial Services",
    "VCI": "Financial Services", "VND": "Financial Services",
    "VDS": "Financial Services",
    # Technology
    "CMG": "Technology", "ELC": "Technology", "FPT": "Technology",
    # Oil & Gas / Energy
    "BSR": "Oil & Gas", "DCM": "Chemicals",
    "DPM": "Chemicals", "GAS": "Oil & Gas", "PLX": "Oil & Gas",
    # Utilities / Power
    "BWE": "Utilities", "GEG": "Utilities", "PC1": "Utilities",
    "POW": "Utilities", "REE": "Utilities", "VSH": "Utilities",
    # Steel & Materials
    "GEX": "Steel & Materials", "HPG": "Steel & Materials",
    "HSG": "Steel & Materials", "KSB": "Steel & Materials",
    "NKG": "Steel & Materials", "VGC": "Steel & Materials",
    "HT1": "Steel & Materials",
    # Consumer Staples / F&B
    "ANV": "Food & Beverage", "BAF": "Agriculture",
    "DBC": "Agriculture", "FMC": "Food & Beverage",
    "GTN": "Agriculture", "HAG": "Agriculture",
    "MSN": "Consumer Staples", "PAN": "Agriculture",
    "QNS": "Food & Beverage", "RAL": "Consumer Staples",
    "SAB": "Food & Beverage", "VCF": "Food & Beverage",
    "VHC": "Food & Beverage", "VNM": "Food & Beverage",
    # Consumer Discretionary / Retail
    "MSH": "Consumer Discretionary", "MWG": "Consumer Discretionary",
    "PNJ": "Consumer Discretionary", "TNG": "Consumer Discretionary",
    "TRA": "Consumer Discretionary", "VFC": "Consumer Discretionary",
    # Transport / Logistics
    "GMD": "Transportation", "GVR": "Agriculture",
    "NCT": "Transportation", "NKG": "Steel & Materials",
    "PVD": "Oil & Gas", "PVT": "Transportation",
    "SCS": "Transportation", "SKG": "Transportation",
    "TV2": "Construction", "VJC": "Transportation",
    "VOS": "Transportation", "VPI": "Real Estate",
    "DPR": "Agriculture",
}


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
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV for a single stock via vnstock.

        Tries the new vnstock.api (4.x) first, then falls back to the old
        Vnstock() API (3.x).  Retries up to max_retries times on connection
        errors with exponential backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                df = self._fetch_one(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    return df
                return None
            except Exception as e:
                err_msg = str(e)
                if "Rate limit" in err_msg or "rate limit" in err_msg:
                    # Respect the rate limit window explicitly
                    wait = 60  # wait a full minute
                    print(f"  Rate limit hit — waiting {wait}s before retrying {symbol}...",
                          flush=True)
                    time.sleep(wait)
                elif attempt < max_retries:
                    wait = 5 * attempt  # 5s, 10s, 15s
                    logger.debug(f"Retry {attempt}/{max_retries} for {symbol} in {wait}s ({e})")
                    time.sleep(wait)
                else:
                    logger.warning(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    return None
        return None

    def _fetch_one(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Single fetch attempt using the best available vnstock API."""
        if _VNSTOCK_NEW_API and _VnQuote is not None:
            # ── new API (vnstock >= 4.x) ──────────────────────────────────
            q = _VnQuote(symbol=symbol, source=self.source)
            df = q.history(start=start_date, end=end_date, interval="1D")
        elif Vnstock is not None:
            # ── old API (vnstock 3.x) — still works on some versions ──────
            stock = Vnstock().stock(symbol=symbol, source=self.source)
            df = stock.quote.history(start=start_date, end=end_date, interval="1D")
        else:
            raise RuntimeError("vnstock is not installed")

        if df is None or df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
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

    def fetch_all_stocks_daily(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        delay: float = 4.0,   # vnstock Guest: 20 req/min → min 3s, use 4s to be safe
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV for all stocks with local caching and rate limiting.

        Args:
            stock_codes: List of ticker symbols
            start_date:  "YYYY-MM-DD"
            end_date:    "YYYY-MM-DD"
            delay:       Seconds between requests (default 4s respects the
                         free-tier 20 req/min rate limit)

        Returns:
            Combined DataFrame of all stocks.
        """
        n_stocks = len(stock_codes)
        cache_path = os.path.join(
            self.cache_dir,
            f"daily_prices_{n_stocks}stocks_{start_date}_{end_date}.parquet",
        )
        if os.path.exists(cache_path):
            logger.info("Loading daily prices from cache.")
            print("  [VN Cache] Loading daily prices from cache (instant).", flush=True)

            return pd.read_parquet(cache_path)

        total = len(stock_codes)
        eta_min = total * delay / 60
        print(
            f"  Fetching {total} stocks with {delay}s delay "
            f"(~{eta_min:.0f} min total, respecting 20 req/min rate limit)",
            flush=True,
        )

        all_data = []
        failed = []

        for code in tqdm(stock_codes, desc="  [VN fetch]", unit="stock", ncols=80):
            df = self.fetch_stock_daily(code, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
            else:
                failed.append(code)
            time.sleep(delay)  # respect rate limit

        if failed:
            print(f"  Warning: {len(failed)} stocks failed: {failed}", flush=True)

        if not all_data:
            raise RuntimeError(
                "No stock data could be fetched! "
                "Check your internet connection and vnstock version."
            )

        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        print(
            f"  Fetched {len(all_data)}/{total} stocks, "
            f"{len(combined):,} total records.",
            flush=True,
        )
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
