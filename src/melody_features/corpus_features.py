"""Corpus-dependent FANTASTIC feature definitions."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy

from .corpus import load_corpus_stats, make_corpus_stats
from .feature_decorators import both, corpus, fantastic
from .melody_tokenizer import FantasticTokenizer
from .ngram_counter import NGramCounter
from .representations import Melody


__all__ = [
    "get_ngram_document_frequency",
    "InverseEntropyWeighting",
    "tfdf_spearman",
    "tfdf_kendall",
    "mean_log_tfdf",
    "norm_log_dist",
    "max_log_df",
    "min_log_df",
    "mean_log_df",
    "mean_global_local_weight",
    "std_global_local_weight",
    "mean_global_weight",
    "std_global_weight",
    "get_corpus_features",
    "_fantastic_melody_tokens",
    "_fantastic_melody_tf_df",
    "_fantastic_log_normalized_tf_df",
    "_fantastic_melody_ngram_counts",
    "_fantastic_min_tie_ranks",
    "_compute_corpus_feature_bundle",
    "_setup_corpus_statistics",
]


@fantastic
def get_ngram_document_frequency(ngram: tuple, corpus_stats: dict) -> int:
    """Retrieve the document frequency for a given n-gram from the corpus statistics.

    Parameters
    ----------
    ngram : tuple
        The n-gram to look up
    corpus_stats : dict
        Dictionary containing corpus statistics

    Returns
    -------
    int
        Document frequency count for the n-gram
    """
    # Get document frequencies dictionary once
    doc_freqs = corpus_stats.get("document_frequencies", {})

    # Convert ngram to string only once
    ngram_str = str(ngram)

    # Look up the count directly
    return doc_freqs.get(ngram_str, {}).get("count", 0)

@fantastic
@both
@corpus
class InverseEntropyWeighting:
    """Calculate local weights for n-grams using an inverse-entropy measure.

    Inverse-entropy weighting is implemented following the specification in
    FANTASTIC and the Handbook of Latent Semantic Analysis (Landauer et al., 2007).
    It provides several quantifiers of the importance of an n-gram (here: m-type)
    based on its relative frequency in a given passage (here: melody)
    and its relative frequency in that passage as compared to the reference corpus.

    This class contains functions to compute the local weight of an m-type,
    the global weight of an m-type, and the combined weight of an m-type.
    """
    def __init__(self, ngram_counts: dict, corpus_stats: dict):
        self.ngram_counts = ngram_counts
        self.corpus_stats = corpus_stats

    @property
    def local_weights(self) -> list[float]:
        """Calculate local weights for n-grams using an inverse-entropy measure.
        The local weight of an m-type is defined as
        `loc.w(τ) = log2(f(τ) + 1)` where `f(τ)` is the frequency of a
        given m-type in the melody. As such, the local weight can take any real value
        greater than zero. High values mean that the m-type provides a lot of information
        about the melody, while low values mean that the m-type provides little information.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts

        Returns
        -------
        list[float]
            List of local weights, x >= 0 for all x in list
        """
        if not self.ngram_counts:
            return []

        local_weights = []
        for tf in self.ngram_counts.values():
            local_weight = np.log2(tf + 1)
            local_weights.append(local_weight)

        return local_weights

    @property
    def global_weights(self) -> list[float]:
        """Calculate global weights for n-grams using an inverse-entropy measure.
        First, a ratio between the frequency of an m-type in the melody and the frequency
        of the same m-type in the corpus is calculated:
        `Pc(τ) = fc(τ)/fC(τ)` where `fc(τ)` is the frequency of a given m-type in the melody,
        and `fC(τ)` is the frequency of the same m-type in the corpus.
        This ratio is then used to calculate the global weight of an m-type:
        `glob.w = 1 + Σ Pc(τ) · log2(Pc(τ)) / log2(|C|)` where `|C|` is the number of
        documents in the corpus.
        Global weights take a value from 0 to 1. A high value corresponds to a less informative m-type,
        while a low value corresponds to a more informative m-type, with regard to its position in the melody.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts
        corpus_stats : dict
            Dictionary containing corpus statistics

        Returns
        -------
        list[float]
            List of global weights, 0 <= x <= 1 for all x in list
        """
        if not self.ngram_counts or not self.corpus_stats:
            return []

        doc_freqs = self.corpus_stats.get("document_frequencies", {})
        total_docs = len(doc_freqs) if doc_freqs else 1

        global_weights = []
        for ngram, tf in self.ngram_counts.items():
            ngram_str = str(ngram)
            df = doc_freqs.get(ngram_str, {}).get("count", 0)

            if df > 0 and total_docs > 0:
                pc_ratio = tf / df if df > 0 else 0.0

                if pc_ratio > 0:
                    entropy_term = pc_ratio * np.log2(pc_ratio)
                    global_weight = 1 + entropy_term / np.log2(total_docs)
                else:
                    global_weight = 1.0
            else:
                global_weight = 1.0

            global_weights.append(global_weight)

        return global_weights

    @property
    def combined_weights(self) -> list[float]:
        """Calculate combined local-global weights for n-grams.
        The combined weight of an m-type is the product of the local and global weights.
        It summarises the relationship between distinctiveness of an m-type compared to the corpus
        and its frequency in the melody. A high combined weight indicates that the m-type is both
        distinctive and frequent in the melody, while a low combined weight indicates that the m-type
        is either not distinctive or not frequent in the melody.

        Parameters
        ----------
        ngram_counts : dict
            Dictionary containing n-gram counts
        corpus_stats : dict
            Dictionary containing corpus statistics

        Returns
        -------
        list[float]
            List of combined weights, x >= 0 for all x in list
        """
        if not self.ngram_counts or not self.corpus_stats:
            return []

        if len(self.local_weights) != len(self.global_weights):
            return []

        return [l * g for l, g in zip(self.local_weights, self.global_weights)]

def _fantastic_melody_tokens(melody: Melody, phrase_gap: float) -> list:
    """Tokenize a melody into FANTASTIC m-type tokens across phrases."""
    tokenizer = FantasticTokenizer()
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")
    all_tokens = []
    for segment in segments:
        all_tokens.extend(
            tokenizer.tokenize_melody(segment.pitches, segment.starts, segment.ends)
        )
    return all_tokens

def _fantastic_melody_tf_df(
    melody_tokens: list,
    doc_freqs: dict,
    max_ngram_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Melody TF and corpus DF for m-types with df > 0 (FANTASTIC ``TFDF.tab`` rows)."""
    tf_values: list[float] = []
    df_values: list[float] = []
    if max_ngram_order < 1:
        return np.array([], dtype=float), np.array([], dtype=float)

    for n in range(1, max_ngram_order + 1):
        ngram_counts: dict[tuple, int] = {}
        for i in range(len(melody_tokens) - n + 1):
            ngram = tuple(melody_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, tf in ngram_counts.items():
            df = doc_freqs.get(str(ngram), {}).get("count", 0)
            if df > 0:
                tf_values.append(float(tf))
                df_values.append(float(df))

    return np.array(tf_values, dtype=float), np.array(df_values, dtype=float)

def _fantastic_log_normalized_tf_df(
    tf: np.ndarray, df: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """log2-normalized TF and DF vectors as in FANTASTIC ``M-Type_Corpus_Features.R``."""
    log_tf = np.log2(tf)
    log_df = np.log2(df)
    return log_tf / log_tf.sum(), log_df / log_df.sum()

def _fantastic_melody_ngram_counts(
    melody_tokens: list,
    max_ngram_order: int,
) -> dict[tuple, int]:
    """Collapsed melody m-type counts for orders 1 through ``max_ngram_order`` (inclusive)."""
    all_ngram_counts: dict[tuple, int] = {}
    if max_ngram_order < 1:
        return all_ngram_counts

    for n in range(1, max_ngram_order + 1):
        ngram_counts: dict[tuple, int] = {}
        for i in range(len(melody_tokens) - n + 1):
            ngram = tuple(melody_tokens[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        for ngram, tf in ngram_counts.items():
            all_ngram_counts[ngram] = all_ngram_counts.get(ngram, 0) + tf
    return all_ngram_counts

def _fantastic_min_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks with FANTASTIC-compatible minimum tie policy."""
    if values.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        min_rank = float(i + 1)
        ranks[order[i:j]] = min_rank
        i = j
    return ranks

def _compute_corpus_feature_bundle(
    melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int
) -> Dict[str, float]:
    """Compute all corpus features from one shared TF/DF + weighting pipeline."""
    doc_freqs = corpus_stats.get("document_frequencies", {})
    all_tokens = _fantastic_melody_tokens(melody, phrase_gap)
    tf, df = _fantastic_melody_tf_df(all_tokens, doc_freqs, max_ngram_order)

    features: dict[str, float] = {
        "tfdf_spearman": 0.0,
        "tfdf_kendall": 0.0,
        "mean_log_tfdf": 0.0,
        "norm_log_dist": 0.0,
        "max_log_df": 0.0,
        "min_log_df": 0.0,
        "mean_log_df": 0.0,
        "mean_global_local_weight": 0.0,
        "std_global_local_weight": 0.0,
        "mean_global_weight": 0.0,
        "std_global_weight": 0.0,
    }

    if tf.size >= 2 and np.var(tf) > 0 and np.var(df) > 0:
        try:
            # FANTASTIC uses ties.method="min"; mirror that explicitly.
            tf_ranks = _fantastic_min_tie_ranks(tf)
            df_ranks = _fantastic_min_tie_ranks(df)
            spearman = scipy.stats.spearmanr(tf_ranks, df_ranks)[0]
            kendall = scipy.stats.kendalltau(tf_ranks, df_ranks)[0]
            features["tfdf_spearman"] = float(spearman if not np.isnan(spearman) else 0.0)
            features["tfdf_kendall"] = float(kendall if not np.isnan(kendall) else 0.0)
        except Exception:
            pass

    if tf.size > 0:
        norm_tf, norm_df = _fantastic_log_normalized_tf_df(tf, df)
        features["mean_log_tfdf"] = float(np.mean(norm_tf * norm_df))
        features["norm_log_dist"] = float(np.sum(np.abs(norm_tf - norm_df)) / tf.size)
        features["max_log_df"] = float(np.log2(df.max()))
        features["min_log_df"] = float(np.log2(df.min()))
        features["mean_log_df"] = float(np.mean(np.log2(df)))

    all_ngram_counts = _fantastic_melody_ngram_counts(all_tokens, max_ngram_order)
    if all_ngram_counts and doc_freqs:
        weights = InverseEntropyWeighting(all_ngram_counts, corpus_stats)
        all_combined_weights = weights.combined_weights
        all_global_weights = weights.global_weights
        if all_combined_weights:
            features["mean_global_local_weight"] = float(np.mean(all_combined_weights))
            features["std_global_local_weight"] = float(
                np.std(all_combined_weights, ddof=1)
                if len(all_combined_weights) > 1
                else 0.0
            )
        if all_global_weights:
            features["mean_global_weight"] = float(np.mean(all_global_weights))
            features["std_global_weight"] = float(
                np.std(all_global_weights, ddof=1)
                if len(all_global_weights) > 1
                else 0.0
            )

    return features

@fantastic
@corpus
@both
def tfdf_spearman(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Spearman rank correlation between TF and DF over m-types. Positive values mean
    higher within-melody usage tends to coincide with higher corpus-wide prevalence across m-types;
    negative values mean the opposite; near zero means little monotonic rank association.

    Notes
    -----
    Ties are ranked with the minimum-rank policy (`ties.method="min"`),
    then Spearman correlation is computed over those ranks.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Spearman correlation coefficient between TF and DF
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["tfdf_spearman"]

@fantastic
@corpus
@both
def tfdf_kendall(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Kendall's tau rank correlation between melody TF and corpus DF for each m-type.

    Similar to ``tfdf_spearman``, but ordinal association is measured
    with Kendall's tau instead of Spearman's rho. Positive values mean
    higher within-melody usage tends to coincide with higher corpus-wide prevalence across m-types;
    negative values mean the opposite; near zero means little monotonic rank association.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Kendall's tau correlation coefficient between TF and DF

    Notes
    -----
    Ties are first converted to minimum ranks, then Kendall's
    tau is computed over the resulting rank vectors.
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["tfdf_kendall"]

@fantastic
@corpus
@both
def mean_log_tfdf(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Mean of log2-normalized TF × DF products over m-types.

    Higher values mean stronger alignment between within-melody usage and corpus
    document-frequency on the same m-types.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Mean log2 TF-DF score
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["mean_log_tfdf"]

@fantastic
@corpus
@both
def norm_log_dist(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Mean absolute difference between log2-normalized TF and DF.

    Larger values mean the melody emphasizes different m-types than corpus prevalence;
    smaller values mean closer distributional match.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Mean absolute deviation between log2-normalized TF and DF vectors
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["norm_log_dist"]

@fantastic
@corpus
@both
def max_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """log2 of the largest corpus document frequency among melody m-types. Highlights how
    frequent the most common pattern in the melody is relative to the corpus. Large values indicate that the melody
    contains at least one pattern that is very frequent in the corpus.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Maximum log2 document frequency
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["max_log_df"]

@fantastic
@corpus
@both
def min_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """log2 of the smallest corpus DF among melody m-types. Highlights how
    frequent the least common pattern in the melody is relative to the corpus. Small values indicate that the melody
    contains at least one pattern that is very rare in the corpus.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Minimum log2 document frequency
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["min_log_df"]

@fantastic
@corpus
@both
def mean_log_df(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Mean log2 corpus DF over melody m-types.
    Highlights how frequent the average pattern in the melody is relative to the corpus.
    Large values indicate that the melody contains patterns that are relatively frequent in the corpus.
    Small values indicate that the melody contains patterns that are relatively rare in the corpus.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Mean log2 document frequency
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["mean_log_df"]

@fantastic
@corpus
@both
def mean_global_local_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Mean combined local-global weights for n-grams.
    The combined weight of an m-type is the product of local and global weights.
    In this implementation, unseen m-types (DF=0) receive a neutral global weight of 1.0.
    Higher values therefore indicate either high local frequency, higher global weight,
    or both. This relates to the percept of distinctiveness.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Mean global-local weight
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["mean_global_local_weight"]

@fantastic
@corpus
@both
def std_global_local_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Sample standard deviation of combined local-global weights for n-grams.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Standard deviation of global-local weight
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["std_global_local_weight"]

@fantastic
@corpus
@both
def mean_global_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Mean global weight across m-types.

    Higher values mean the m-types are less globally informative (more expected);
    lower values mean they are more globally informative (more distinctive).

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Mean global weight
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["mean_global_weight"]

@fantastic
@corpus
@both
def std_global_weight(melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int) -> float:
    """Sample standard deviation of global weights for m-types.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics
    phrase_gap : float
        Gap threshold for phrase segmentation
    max_ngram_order : int
        Maximum n-gram order to consider

    Returns
    -------
    float
        Standard deviation of global weight
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )["std_global_weight"]

def get_corpus_features(
    melody: Melody, corpus_stats: dict, phrase_gap: float, max_ngram_order: int
) -> Dict:
    """Compute all corpus-based features for a melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze
    corpus_stats : dict
        Dictionary containing corpus statistics

    Returns
    -------
    Dict
        Dictionary of corpus-based feature values
    """
    return _compute_corpus_feature_bundle(
        melody, corpus_stats, phrase_gap, max_ngram_order
    )

def _setup_corpus_statistics(config, output_file: str) -> Optional[dict]:
    """Set up corpus statistics for FANTASTIC features.

    Parameters
    ----------
    config : Config
        Configuration object containing corpus information
    output_file : str
        Path to output file for determining corpus stats location

    Returns
    -------
    Optional[dict]
        Corpus statistics dictionary or None if no corpus provided

    Raises
    ------
    FileNotFoundError
        If corpus path is not a valid directory
    """
    logger = logging.getLogger("melody_features")

    # Determine which corpus to use for FANTASTIC
    fantastic_corpus = (
        config.fantastic.corpus
        if config.fantastic.corpus is not None
        else config.corpus
    )

    if not fantastic_corpus:
        logger.info(
            "No corpus path provided, corpus-dependent features will not be computed."
        )
        return None

    if not Path(fantastic_corpus).is_dir():
        raise FileNotFoundError(
            f"Corpus path is not a valid directory: {fantastic_corpus}"
        )

    logger.info(f"Generating corpus statistics from: {fantastic_corpus}")

    # Include tokenizer-sensitive settings in cache key to avoid stale stats reuse.
    corpus_name = Path(fantastic_corpus).name
    phrase_gap_slug = str(config.fantastic.phrase_gap).replace(".", "p")
    max_order = config.fantastic.max_ngram_order
    corpus_stats_path = (
        Path(output_file).parent
        / f"{corpus_name}_corpus_stats_pg{phrase_gap_slug}_n{max_order}.json"
    )
    logger.info(f"Corpus statistics file will be at: {corpus_stats_path}")

    # Ensure the directory exists
    corpus_stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and load corpus stats.
    if not corpus_stats_path.exists():
        logger.info("Corpus statistics file not found. Generating a new one...")
        n_range = (1, config.fantastic.max_ngram_order)
        make_corpus_stats(
            fantastic_corpus,
            str(corpus_stats_path),
            n_range=n_range,
            phrase_gap=config.fantastic.phrase_gap,
        )
        logger.info("Corpus statistics generated.")
    else:
        logger.info("Existing corpus statistics file found.")

    corpus_stats = load_corpus_stats(str(corpus_stats_path))
    logger.info("Corpus statistics loaded successfully.")

    return corpus_stats
