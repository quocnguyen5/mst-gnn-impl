"""
Graph Builder
=============
Constructs the 4 multilayer stock network layers described in Section III:

1. Shareholding Network (static): cross-shareholding relationships
2. Industry Network (static): same-sector connections
3. Topicality Network (semi-dynamic): news topic similarity via LDA
4. Comovement Network (dynamic): rolling price correlation

Each network Gₜ,q = (Sₜ, Eₜ,q, Xₜ) is represented as a PyG Data object.
The multilayer graph at time t is Gₜ = {Gₜ,q, q ∈ R}.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Constructs the 4 multilayer stock networks."""

    def __init__(
        self,
        comovement_window: int = 20,
        comovement_threshold: float = 0.3,
        num_topics: int = 50,
        topic_similarity_threshold: float = 0.2,
    ):
        """
        Args:
            comovement_window: Rolling window for price correlation
            comovement_threshold: Minimum correlation for edge creation
            num_topics: Number of LDA topics for topicality network
            topic_similarity_threshold: Minimum topic similarity for edges
        """
        self.comovement_window = comovement_window
        self.comovement_threshold = comovement_threshold
        self.num_topics = num_topics
        self.topic_similarity_threshold = topic_similarity_threshold

        # LDA model (trained once, used for all dates)
        self._lda_model = None
        self._dictionary = None

    # ------------------------------------------------------------------
    # 1. Shareholding Network (Static)
    # ------------------------------------------------------------------

    def build_shareholding_network(
        self,
        shareholding_df: pd.DataFrame,
        stock_codes: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the shareholding network.

        "describes the shareholder relations between listed companies,
        reflecting the financial influence between a company and its
        shareholder. The edge weight represents the shareholding ratio"

        Args:
            shareholding_df: DataFrame with [held_code, holder_name/holder_code, ratio]
            stock_codes: List of stock codes (defines node set)

        Returns:
            edge_index: (2, num_edges) array
            edge_weight: (num_edges,) array of shareholding ratios
        """
        code_to_idx = {code: i for i, code in enumerate(stock_codes)}
        edges = []
        weights = []

        if shareholding_df.empty:
            logger.warning("Empty shareholding data, returning empty network.")
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # If holder_code column exists, use it directly
        if "holder_code" in shareholding_df.columns:
            for _, row in shareholding_df.iterrows():
                holder = str(row["holder_code"]).zfill(6)
                held = str(row["held_code"]).zfill(6)
                ratio = float(row.get("ratio", 0.0))

                if holder in code_to_idx and held in code_to_idx:
                    edges.append([code_to_idx[holder], code_to_idx[held]])
                    weights.append(ratio / 100.0)  # normalize to [0, 1]
        else:
            # Fallback: create approximate network from holder names
            # by matching holder names to stock names
            logger.info(
                "No holder_code column; shareholding network will be sparse."
            )

        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
        edge_weight = np.array(weights, dtype=np.float32)
        logger.info(f"Shareholding network: {edge_index.shape[1]} edges")
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # 2. Industry Network (Static)
    # ------------------------------------------------------------------

    def build_industry_network(
        self,
        industry_df: pd.DataFrame,
        stock_codes: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the industry network.

        "denotes the sector affiliation of listed companies. It is based
        on the investment phenomenon where a leading stock in an industry
        influences other stocks' prices"

        Stocks in the same industry are connected (binary edges).

        Args:
            industry_df: DataFrame with [stock_code, industry_name]
            stock_codes: List of stock codes

        Returns:
            edge_index: (2, num_edges) array
            edge_weight: (num_edges,) array (all 1.0 for binary edges)
        """
        code_to_idx = {code: i for i, code in enumerate(stock_codes)}

        # Group stocks by industry
        industry_groups = {}
        for _, row in industry_df.iterrows():
            code = str(row["stock_code"]).zfill(6)
            industry = row["industry_name"]
            if code in code_to_idx:
                industry_groups.setdefault(industry, []).append(code)

        edges = []
        for industry, codes in industry_groups.items():
            # Fully connect stocks within the same industry
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    idx_i = code_to_idx[codes[i]]
                    idx_j = code_to_idx[codes[j]]
                    # Bidirectional edges
                    edges.append([idx_i, idx_j])
                    edges.append([idx_j, idx_i])

        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        logger.info(
            f"Industry network: {edge_index.shape[1]} edges, "
            f"{len(industry_groups)} industries"
        )
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # 3. Topicality Network (Semi-Dynamic)
    # ------------------------------------------------------------------

    def train_lda_model(
        self, news_df: pd.DataFrame
    ) -> None:
        """
        Train an LDA topic model on financial news corpus.

        "depicts the similarity of topics extracted from news. News on
        particular topics often impacts multiple related stocks, leading
        to similar price volatility"
        """
        try:
            import jieba
            from gensim import corpora
            from gensim.models import LdaModel
        except ImportError:
            logger.error("jieba and gensim are required for topicality network.")
            return

        if news_df.empty:
            logger.warning("No news data for LDA training.")
            return

        logger.info("Training LDA topic model...")

        # Tokenize Chinese text
        texts = []
        for _, row in news_df.iterrows():
            content = str(row.get("content", "")) + " " + str(row.get("title", ""))
            tokens = [
                w for w in jieba.cut(content) if len(w) > 1  # filter single chars
            ]
            if tokens:
                texts.append(tokens)

        if not texts:
            logger.warning("No valid text for LDA training.")
            return

        # Build dictionary and corpus
        self._dictionary = corpora.Dictionary(texts)
        # Filter extreme frequencies
        self._dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [self._dictionary.doc2bow(text) for text in texts]

        # Train LDA
        self._lda_model = LdaModel(
            corpus=corpus,
            id2word=self._dictionary,
            num_topics=self.num_topics,
            passes=10,
            random_state=42,
        )
        logger.info(f"LDA model trained with {self.num_topics} topics.")

    def _get_stock_topic_distribution(
        self,
        news_df: pd.DataFrame,
        stock_code: str,
        date: pd.Timestamp,
        window_days: int = 7,
    ) -> Optional[np.ndarray]:
        """Get topic distribution for a stock based on recent news."""
        try:
            import jieba
        except ImportError:
            return None

        if self._lda_model is None or self._dictionary is None:
            return None

        # Get news for this stock in the last window_days
        # Convert both sides to pd.Timestamp to ensure consistent comparison.
        # news_df["date"] may be datetime.date objects (from .dt.date), so
        # we normalise the column inline using pd.to_datetime.
        date_ts = pd.Timestamp(date)
        date_start = date_ts - pd.Timedelta(days=window_days)
        news_dates = pd.to_datetime(news_df["date"])
        mask = (
            (news_df["stock_code"] == stock_code)
            & (news_dates >= date_start)
            & (news_dates <= date_ts)
        )
        stock_news = news_df.loc[mask]

        if stock_news.empty:
            return None

        # Aggregate all news text
        all_text = " ".join(
            stock_news["title"].fillna("").astype(str)
            + " "
            + stock_news["content"].fillna("").astype(str)
        )
        tokens = [w for w in jieba.cut(all_text) if len(w) > 1]
        if not tokens:
            return None

        bow = self._dictionary.doc2bow(tokens)
        topic_dist = self._lda_model.get_document_topics(
            bow, minimum_probability=0.0
        )

        # Convert to dense vector
        dist = np.zeros(self.num_topics, dtype=np.float32)
        for topic_id, prob in topic_dist:
            dist[topic_id] = prob
        return dist

    def build_topicality_network(
        self,
        news_df: pd.DataFrame,
        stock_codes: List[str],
        date: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build topicality network for a given date.

        Computes cosine similarity of topic distributions between
        all pairs of stocks. Creates an edge if similarity exceeds threshold.

        Args:
            news_df: Financial news DataFrame
            stock_codes: Active stock codes
            date: Current date

        Returns:
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) cosine similarities
        """
        if news_df is None or news_df.empty:
            logger.debug("No news data available. Returning empty topicality network.")
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        if self._lda_model is None:
            logger.debug("LDA model not trained. Returning empty topicality network.")
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # Get topic distribution for each stock
        topic_dists = {}
        for code in stock_codes:
            dist = self._get_stock_topic_distribution(news_df, code, date)
            if dist is not None:
                topic_dists[code] = dist

        # Build edges based on cosine similarity
        code_to_idx = {code: i for i, code in enumerate(stock_codes)}
        codes_with_topics = list(topic_dists.keys())
        edges = []
        weights = []

        for i in range(len(codes_with_topics)):
            for j in range(i + 1, len(codes_with_topics)):
                ci, cj = codes_with_topics[i], codes_with_topics[j]
                # Cosine similarity (1 - cosine distance)
                sim = 1.0 - cosine(topic_dists[ci], topic_dists[cj])

                if sim >= self.topic_similarity_threshold:
                    idx_i = code_to_idx[ci]
                    idx_j = code_to_idx[cj]
                    edges.append([idx_i, idx_j])
                    edges.append([idx_j, idx_i])
                    weights.extend([sim, sim])

        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.array(weights, dtype=np.float32)
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # 4. Comovement Network (Dynamic)
    # ------------------------------------------------------------------

    def build_comovement_network(
        self,
        price_df: pd.DataFrame,
        stock_codes: List[str],
        date: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build comovement network for a given date.

        "models time-varying price correlation. An edge in the stock
        comovement represents the similarity of two stocks' price movement
        patterns exceeding a given level"

        Uses rolling Pearson correlation of daily returns.

        Args:
            price_df: DataFrame with [date, stock_code, close]
            stock_codes: Active stock codes on this date
            date: Current date

        Returns:
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) correlation values
        """
        code_to_idx = {code: i for i, code in enumerate(stock_codes)}

        # Get return series for the rolling window ending at `date`
        end_date = date
        mask = price_df["date"] <= end_date
        recent = price_df.loc[mask].copy()

        # Pivot to get returns matrix: (dates, stocks)
        pivot = recent.pivot_table(
            index="date", columns="stock_code", values="close"
        )
        # Keep only active stocks
        valid_cols = [c for c in stock_codes if c in pivot.columns]
        pivot = pivot[valid_cols]

        # Take last `window` trading days
        if len(pivot) < self.comovement_window:
            window = len(pivot)
        else:
            window = self.comovement_window
        pivot = pivot.tail(window)

        # Compute returns
        returns = pivot.pct_change().dropna()

        if returns.empty or returns.shape[1] < 2:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        # Correlation matrix
        corr_matrix = returns.corr()

        # Build edges from correlation exceeding threshold
        edges = []
        weights = []
        codes_in_corr = list(corr_matrix.columns)

        for i in range(len(codes_in_corr)):
            for j in range(i + 1, len(codes_in_corr)):
                ci, cj = codes_in_corr[i], codes_in_corr[j]
                corr_val = corr_matrix.loc[ci, cj]

                if (
                    not np.isnan(corr_val)
                    and abs(corr_val) >= self.comovement_threshold
                ):
                    if ci in code_to_idx and cj in code_to_idx:
                        idx_i = code_to_idx[ci]
                        idx_j = code_to_idx[cj]
                        edges.append([idx_i, idx_j])
                        edges.append([idx_j, idx_i])
                        weights.extend([abs(corr_val), abs(corr_val)])

        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.array(weights, dtype=np.float32)
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Build All Networks for a Given Date
    # ------------------------------------------------------------------

    def build_multilayer_graph(
        self,
        date: pd.Timestamp,
        stock_codes: List[str],
        price_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        shareholding_df: pd.DataFrame,
        news_df: pd.DataFrame,
        # Pre-computed static networks (avoid re-computing each date)
        static_shareholding: Optional[Tuple] = None,
        static_industry: Optional[Tuple] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Build all 4 network layers for a given date.

        Returns:
            Dict mapping network_name -> (edge_index, edge_weight)
            Keys: "shareholding", "industry", "topicality", "comovement"
        """
        networks = {}

        # Static networks (computed once, reused)
        if static_shareholding is not None:
            networks["shareholding"] = static_shareholding
        else:
            networks["shareholding"] = self.build_shareholding_network(
                shareholding_df, stock_codes
            )

        if static_industry is not None:
            networks["industry"] = static_industry
        else:
            networks["industry"] = self.build_industry_network(
                industry_df, stock_codes
            )

        # Semi-dynamic: topicality
        networks["topicality"] = self.build_topicality_network(
            news_df, stock_codes, date
        )

        # Dynamic: comovement
        networks["comovement"] = self.build_comovement_network(
            price_df, stock_codes, date
        )

        return networks

    def build_temporal_multilayer_graphs(
        self,
        trading_dates: List,
        price_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        shareholding_df: pd.DataFrame,
        news_df: pd.DataFrame,
        active_stocks_per_date: Dict,
    ) -> Dict:
        """
        Build multilayer graphs for all trading dates.

        Returns:
            Dict mapping date -> {
                "stock_codes": List[str],
                "networks": Dict[str, (edge_index, edge_weight)]
            }
        """
        total_dates = len(trading_dates)
        logger.info(
            f"Building temporal multilayer graphs for {total_dates} dates..."
        )
        print(
            f"[Phase 3] Building graphs for {total_dates} trading days "
            f"(this is the slowest step — ~5-15 min)",
            flush=True,
        )

        # Pre-compute static networks (using the full stock set)
        all_stocks = sorted(
            set(code for codes in active_stocks_per_date.values() for code in codes)
        )
        print(f"  Active stocks in universe: {len(all_stocks)}", flush=True)

        print("  Building shareholding network (static)...", flush=True)
        static_shareholding = self.build_shareholding_network(
            shareholding_df, all_stocks
        )
        print("  Building industry network (static)...", flush=True)
        static_industry = self.build_industry_network(
            industry_df, all_stocks
        )

        # Train LDA model once
        if not news_df.empty:
            print("  Training LDA topic model for topicality network...", flush=True)
            self.train_lda_model(news_df)

        all_graphs = {}
        t0 = time.time()

        # tqdm progress bar: visible in Colab and terminal
        for i, date in enumerate(
            tqdm(trading_dates, desc="  [Graph build]", unit="day", ncols=80)
        ):
            stock_codes = active_stocks_per_date.get(date, all_stocks)

            # Re-index static networks for the current stock subset
            # (needed because the active stock set may change)
            code_to_global = {code: idx for idx, code in enumerate(all_stocks)}
            code_to_local = {code: idx for idx, code in enumerate(stock_codes)}

            networks = {}

            # Remap static networks to local indices
            for net_name, (ei, ew) in [
                ("shareholding", static_shareholding),
                ("industry", static_industry),
            ]:
                if ei.shape[1] > 0:
                    global_to_local = {}
                    for code in stock_codes:
                        if code in code_to_global:
                            global_to_local[code_to_global[code]] = code_to_local[code]

                    new_edges = []
                    new_weights = []
                    for e in range(ei.shape[1]):
                        src, dst = ei[0, e], ei[1, e]
                        if src in global_to_local and dst in global_to_local:
                            new_edges.append(
                                [global_to_local[src], global_to_local[dst]]
                            )
                            new_weights.append(ew[e])

                    if new_edges:
                        networks[net_name] = (
                            np.array(new_edges, dtype=np.int64).T,
                            np.array(new_weights, dtype=np.float32),
                        )
                    else:
                        networks[net_name] = (
                            np.zeros((2, 0), dtype=np.int64),
                            np.zeros(0, dtype=np.float32),
                        )
                else:
                    networks[net_name] = (
                        np.zeros((2, 0), dtype=np.int64),
                        np.zeros(0, dtype=np.float32),
                    )

            # Dynamic networks
            networks["topicality"] = self.build_topicality_network(
                news_df, stock_codes, pd.Timestamp(date)
            )
            networks["comovement"] = self.build_comovement_network(
                price_df, stock_codes, pd.Timestamp(date)
            )

            all_graphs[date] = {
                "stock_codes": stock_codes,
                "networks": networks,
            }

        elapsed = time.time() - t0
        print(
            f"  [Phase 3 complete] {total_dates} graphs built in "
            f"{elapsed/60:.1f} min ({elapsed/total_dates:.2f}s/day)",
            flush=True,
        )
        logger.info("Temporal multilayer graph construction complete.")
        return all_graphs
