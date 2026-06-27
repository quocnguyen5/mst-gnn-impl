"""
Data Collector
==============
Fetches stock data using baostock (primary, works globally) and
AKShare (fallback, may be blocked outside mainland China).
Handles CSI 300 and CSI 500 constituent stocks including:
- Daily OHLCV price data
- Industry classification
- Shareholding relationships
- Financial news for topicality network

Reference: Section IV-A of the paper.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    raise ImportError(
        "AKShare is required. Install with: pip install akshare"
    )

try:
    import baostock as bs
    _BAOSTOCK_AVAILABLE = True
except ImportError:
    _BAOSTOCK_AVAILABLE = False
    bs = None

logger = logging.getLogger(__name__)


def _to_baostock_code(code: str) -> str:
    """Convert 6-digit A-share code to baostock format.
    
    Shanghai stocks start with 6: sh.6xxxxx
    Shenzhen stocks start with 0 or 3: sz.0xxxxx / sz.3xxxxx
    """
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


class StockDataCollector:
    """Collects raw stock data from AKShare for CSI 300/500 constituents."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Constituent Stock Lists
    # ------------------------------------------------------------------

    def get_csi300_constituents(self) -> pd.DataFrame:
        """
        Get current CSI 300 index constituent stocks.

        Returns:
            DataFrame with columns: [code, name, weight, ...]
        """
        cache_path = os.path.join(self.cache_dir, "csi300_constituents.csv")
        if os.path.exists(cache_path):
            logger.info("Loading CSI 300 constituents from cache.")
            return pd.read_csv(cache_path, dtype={"code": str})

        logger.info("Fetching CSI 300 constituents from AKShare...")
        try:
            # AKShare function for CSI 300 constituent list
            df = ak.index_stock_cons_csindex(symbol="000300")
            df.rename(
                columns={"成分券代码": "code", "成分券名称": "name"},
                inplace=True,
            )
            df["code"] = df["code"].astype(str).str.zfill(6)
            df.to_csv(cache_path, index=False)
            logger.info(f"Fetched {len(df)} CSI 300 constituents.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch CSI 300 constituents: {e}")
            raise

    def get_csi500_constituents(self) -> pd.DataFrame:
        """
        Get current CSI 500 index constituent stocks.

        Returns:
            DataFrame with columns: [code, name, weight, ...]
        """
        cache_path = os.path.join(self.cache_dir, "csi500_constituents.csv")
        if os.path.exists(cache_path):
            logger.info("Loading CSI 500 constituents from cache.")
            return pd.read_csv(cache_path, dtype={"code": str})

        logger.info("Fetching CSI 500 constituents from AKShare...")
        try:
            df = ak.index_stock_cons_csindex(symbol="000905")
            df.rename(
                columns={"成分券代码": "code", "成分券名称": "name"},
                inplace=True,
            )
            df["code"] = df["code"].astype(str).str.zfill(6)
            df.to_csv(cache_path, index=False)
            logger.info(f"Fetched {len(df)} CSI 500 constituents.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch CSI 500 constituents: {e}")
            raise

    def get_constituents(self, dataset: str = "csi300") -> pd.DataFrame:
        """Get constituents for the specified dataset."""
        if dataset == "csi300":
            return self.get_csi300_constituents()
        elif dataset == "csi500":
            return self.get_csi500_constituents()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    # ------------------------------------------------------------------
    # 2. Daily OHLCV Price Data
    # ------------------------------------------------------------------

    def _fetch_stock_daily_baostock(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from baostock (works from non-China IPs)."""
        if not _BAOSTOCK_AVAILABLE:
            return None
        try:
            bs_code = _to_baostock_code(stock_code)
            # adjustflag: "2"=forward-adjusted (qfq), "1"=backward, "3"=none
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                logger.debug(f"baostock error for {stock_code}: {rs.error_msg}")
                return None

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            # baostock returns strings — convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df["stock_code"] = stock_code
            df = df.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
            return df[["date", "stock_code", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"baostock fetch failed for {stock_code}: {e}")
            return None

    def _fetch_stock_daily_akshare(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from AKShare (may be blocked outside China)."""
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=sd,
                end_date=ed,
                adjust="qfq",
            )
            if df is None or df.empty:
                return None
            df.rename(
                columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                },
                inplace=True,
            )
            df["date"] = pd.to_datetime(df["date"])
            df["stock_code"] = stock_code
            df = df.sort_values("date").reset_index(drop=True)
            return df[["date", "stock_code", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"AKShare fetch failed for {stock_code}: {e}")
            return None

    def fetch_stock_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a single stock.
        Tries baostock first (works globally), then falls back to AKShare.

        Args:
            stock_code: 6-digit stock code (e.g., "000001")
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: [date, stock_code, open, high, low, close, volume]
        """
        # --- Primary: baostock (TCP protocol, works from Taiwan/Colab) ---
        df = self._fetch_stock_daily_baostock(stock_code, start_date, end_date)
        if df is not None and not df.empty:
            return df

        # --- Fallback: AKShare (HTTP, may be blocked outside mainland China) ---
        logger.debug(f"baostock failed for {stock_code}, trying AKShare...")
        df = self._fetch_stock_daily_akshare(stock_code, start_date, end_date)
        if df is not None and not df.empty:
            return df

        logger.warning(f"All sources failed for stock {stock_code}.")
        return None

    def fetch_all_stocks_daily(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        delay: float = 0.1,
    ) -> pd.DataFrame:
        """
        Fetch daily data for all stocks.
        Uses baostock (primary, no rate limit issues) with AKShare as fallback.

        Args:
            stock_codes: List of 6-digit stock codes
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            delay: Delay between requests (baostock needs very little)

        Returns:
            Combined DataFrame of all stocks
        """
        n_stocks = len(stock_codes)
        cache_path = os.path.join(
            self.cache_dir,
            f"daily_prices_{n_stocks}stocks_{start_date}_{end_date}.parquet",
        )
        if os.path.exists(cache_path):
            logger.info("Loading daily prices from cache.")
            return pd.read_parquet(cache_path)

        # Login to baostock once for the whole batch
        if _BAOSTOCK_AVAILABLE:
            login_result = bs.login()
            if login_result.error_code == "0":
                logger.info("baostock login successful — using as primary price source.")
            else:
                logger.warning(f"baostock login failed: {login_result.error_msg}. Falling back to AKShare.")

        all_data = []
        total = len(stock_codes)

        try:
            for i, code in enumerate(stock_codes):
                if (i + 1) % 50 == 0 or i == 0:
                    logger.info(f"Fetching stock {i+1}/{total}: {code}")

                df = self.fetch_stock_daily(code, start_date, end_date)
                if df is not None and not df.empty:
                    all_data.append(df)

                time.sleep(delay)
        finally:
            if _BAOSTOCK_AVAILABLE:
                bs.logout()

        if not all_data:
            raise RuntimeError("No stock data could be fetched!")

        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        logger.info(
            f"Fetched data for {len(all_data)}/{total} stocks, "
            f"{len(combined)} total records."
        )
        return combined

    # ------------------------------------------------------------------
    # 3. Industry Classification
    # ------------------------------------------------------------------

    def fetch_industry_classification(self) -> pd.DataFrame:
        """
        Fetch CSRC industry classification for all A-share stocks.
        Used for constructing the Industry network layer.

        Returns:
            DataFrame with columns: [stock_code, industry_code, industry_name]
        """
        cache_path = os.path.join(self.cache_dir, "industry_classification.csv")
        if os.path.exists(cache_path):
            logger.info("Loading industry classification from cache.")
            return pd.read_csv(cache_path, dtype={"stock_code": str})

        # --- Primary: baostock query_stock_industry (works globally) ---
        if _BAOSTOCK_AVAILABLE:
            logger.info("Fetching industry classification from baostock...")
            try:
                login_result = bs.login()
                if login_result.error_code == "0":
                    rs = bs.query_stock_industry()
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    bs.logout()

                    if rows:
                        df = pd.DataFrame(
                            rows,
                            columns=["updateDate", "code", "code_name",
                                     "industry", "industryClassification"]
                        )
                        # Convert baostock code (sh.600000) → 6-digit code
                        df["stock_code"] = df["code"].str.split(".").str[1].str.zfill(6)
                        df = df.rename(columns={"industry": "industry_name"})
                        result = df[["stock_code", "industry_name"]].dropna()
                        result = result.drop_duplicates(subset=["stock_code"], keep="first")
                        result.to_csv(cache_path, index=False)
                        logger.info(f"Fetched industry for {len(result)} stocks via baostock.")
                        return result
                else:
                    bs.logout()
                    logger.warning(f"baostock login failed: {login_result.error_msg}")
            except Exception as e:
                logger.warning(f"baostock industry fetch failed: {e}")

        # --- Fallback: AKShare (may be blocked outside China) ---
        logger.info("Trying AKShare for industry classification...")
        try:
            industry_list = ak.stock_board_industry_name_em()
            all_industry_data = []

            for _, row in industry_list.iterrows():
                industry_name = row["板块名称"]
                try:
                    members = ak.stock_board_industry_cons_em(symbol=industry_name)
                    if members is not None and not members.empty:
                        members_df = pd.DataFrame(
                            {
                                "stock_code": members["代码"].astype(str).str.zfill(6),
                                "industry_name": industry_name,
                            }
                        )
                        all_industry_data.append(members_df)
                    time.sleep(0.2)
                except Exception as e:
                    logger.debug(f"Skipping industry {industry_name}: {e}")
                    continue

            if all_industry_data:
                result = pd.concat(all_industry_data, ignore_index=True)
                result = result.drop_duplicates(subset=["stock_code"], keep="first")
                result.to_csv(cache_path, index=False)
                logger.info(f"Fetched industry for {len(result)} stocks via AKShare.")
                return result
        except Exception as e:
            logger.warning(f"AKShare industry fetch failed: {e}")

        # --- Last resort: return empty DataFrame (Industry network disabled) ---
        logger.warning(
            "Could not fetch industry classification from any source. "
            "Industry network layer will be empty (all stocks in one group)."
        )
        return pd.DataFrame(columns=["stock_code", "industry_name"])

    # ------------------------------------------------------------------
    # 4. Shareholding Data
    # ------------------------------------------------------------------

    def fetch_shareholding_data(
        self, stock_codes: List[str], delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Fetch top-10 shareholder data for cross-shareholding network.
        Constructs edges where company A holds shares in company B.

        Returns:
            DataFrame with columns: [holder_code, held_code, ratio]
        """
        cache_path = os.path.join(self.cache_dir, "shareholding.csv")
        if os.path.exists(cache_path):
            logger.info("Loading shareholding data from cache.")
            return pd.read_csv(
                cache_path, dtype={"holder_code": str, "held_code": str}
            )

        logger.info("Fetching shareholding data...")
        stock_set = set(stock_codes)
        edges = []

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"Shareholding: {i+1}/{len(stock_codes)}")
            try:
                # Top 10 shareholders (circulating shares)
                holders = ak.stock_gdfx_free_top_10_em(symbol=code)
                if holders is None or holders.empty:
                    continue

                for _, h_row in holders.iterrows():
                    holder_name = str(h_row.get("股东名称", ""))
                    ratio = h_row.get("持股比例", 0)
                    if isinstance(ratio, (int, float)) and ratio > 0:
                        edges.append(
                            {
                                "held_code": code,
                                "holder_name": holder_name,
                                "ratio": float(ratio),
                            }
                        )

                time.sleep(delay)
            except Exception as e:
                logger.debug(f"Shareholding error for {code}: {e}")
                continue

        result = pd.DataFrame(edges)
        if not result.empty:
            result.to_csv(cache_path, index=False)
            logger.info(f"Fetched {len(result)} shareholding records.")
        else:
            logger.warning(
                "No shareholding data fetched (API may be blocked). "
                "Shareholding network layer will be empty."
            )
            # Save empty file so we don't retry on next run
            result = pd.DataFrame(columns=["held_code", "holder_name", "ratio"])
            result.to_csv(cache_path, index=False)
        return result

    # ------------------------------------------------------------------
    # 5. Financial News Data
    # ------------------------------------------------------------------

    def fetch_financial_news(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch financial news related to stocks for topicality network.
        Uses AKShare's news functions.

        Returns:
            DataFrame with columns: [date, stock_code, title, content]
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"news_{start_date}_{end_date}.parquet",
        )
        if os.path.exists(cache_path):
            logger.info("Loading news data from cache.")
            return pd.read_parquet(cache_path)

        logger.info("Fetching financial news data...")
        all_news = []

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"News: {i+1}/{len(stock_codes)}")
            try:
                # Individual stock news from East Money
                news_df = ak.stock_news_em(symbol=code)
                if news_df is not None and not news_df.empty:
                    news_df = news_df.rename(
                        columns={
                            "发布时间": "datetime",
                            "新闻标题": "title",
                            "新闻内容": "content",
                        }
                    )
                    news_df["stock_code"] = code
                    news_df["date"] = pd.to_datetime(
                        news_df["datetime"]
                    ).dt.date
                    all_news.append(
                        news_df[["date", "stock_code", "title", "content"]]
                    )
                time.sleep(delay)
            except Exception as e:
                logger.debug(f"News error for {code}: {e}")
                continue

        if not all_news:
            logger.warning("No news data fetched. Topicality network will be empty.")
            return pd.DataFrame(
                columns=["date", "stock_code", "title", "content"]
            )

        result = pd.concat(all_news, ignore_index=True)
        result.to_parquet(cache_path, index=False)
        logger.info(f"Fetched {len(result)} news articles.")
        return result

    # ------------------------------------------------------------------
    # Master Collection Function
    # ------------------------------------------------------------------

    def collect_all(
        self,
        dataset: str = "csi300",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data for the specified dataset.

        Returns:
            Dictionary with keys:
            - "constituents": constituent stock list
            - "daily_prices": OHLCV data
            - "industry": industry classification
            - "shareholding": shareholding relationships
            - "news": financial news articles
        """
        if dataset == "csi300":
            start_date = start_date or "2018-01-02"
            end_date = end_date or "2026-06-26"
        elif dataset == "csi500":
            start_date = start_date or "2018-01-02"
            end_date = end_date or "2026-06-26"

        # 1. Get constituent list
        constituents = self.get_constituents(dataset)
        stock_codes = constituents["code"].tolist()
        logger.info(f"Dataset {dataset}: {len(stock_codes)} stocks")

        # 2. Fetch daily prices
        daily_prices = self.fetch_all_stocks_daily(
            stock_codes, start_date, end_date
        )

        # 3. Industry classification
        industry = self.fetch_industry_classification()

        # 4. Shareholding data
        shareholding = self.fetch_shareholding_data(stock_codes)

        # 5. Financial news
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
    collector = StockDataCollector()
    data = collector.collect_all(dataset="csi300")
    for key, df in data.items():
        print(f"{key}: {df.shape}")
