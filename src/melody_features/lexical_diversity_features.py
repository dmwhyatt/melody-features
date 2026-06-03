"""Melody-only lexical diversity feature definitions."""

import warnings

from .feature_decorators import both, fantastic, lexical_diversity
from .melody_tokenizer import FantasticTokenizer
from .ngram_counter import NGramCounter
from .representations import Melody

DEFAULT_MAX_NGRAM_ORDER = 5


__all__ = [
    "get_mtype_features",
    "get_lexical_diversity_features",
]


@fantastic
@both
@lexical_diversity
def get_mtype_features(melody: Melody, phrase_gap: float, max_ngram_order: int) -> dict:
    """Various n-gram statistics for the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    dict
        Dictionary containing complexity measures averaged across n-gram lengths
    """
    # Initialize tokenizer and get M-type tokens
    tokenizer = FantasticTokenizer()

    # Segment the melody first, using quarters as the time unit
    segments = tokenizer.segment_melody(melody, phrase_gap=phrase_gap, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    # Create a fresh counter for this melody
    ngram_counter = NGramCounter()
    ngram_counter.ngram_counts = {}  # Explicitly reset the counter

    ngram_counter.count_ngrams(all_tokens, max_order=max_ngram_order)

    # Calculate complexity measures for each n-gram length
    mtype_features = {}

    # Initialize all features to NaN
    mtype_features["yules_k"] = float("nan")
    mtype_features["simpsons_d"] = float("nan")
    mtype_features["sichels_s"] = float("nan")
    mtype_features["honores_h"] = float("nan")
    mtype_features["mean_entropy"] = float("nan")
    mtype_features["mean_productivity"] = float("nan")

    # Try to calculate each feature individually
    if ngram_counter.ngram_counts:
        try:
            mtype_features["yules_k"] = ngram_counter.yules_k
        except Exception as e:
            warnings.warn(f"Error calculating Yule's K: {str(e)}")
        try:
            mtype_features["simpsons_d"] = ngram_counter.simpsons_d
        except Exception as e:
            warnings.warn(f"Error calculating Simpson's D: {str(e)}")

        try:
            mtype_features["sichels_s"] = ngram_counter.sichels_s
        except Exception as e:
            warnings.warn(f"Error calculating Sichel's S: {str(e)}")

        try:
            mtype_features["honores_h"] = ngram_counter.honores_h
        except Exception as e:
            warnings.warn(f"Error calculating Honoré's H: {str(e)}")

        try:
            mtype_features["mean_entropy"] = ngram_counter.mean_entropy
        except Exception as e:
            warnings.warn(f"Error calculating mean entropy: {str(e)}")

        try:
            mtype_features["mean_productivity"] = ngram_counter.mean_productivity
        except Exception as e:
            warnings.warn(f"Error calculating mean productivity: {str(e)}")

    return mtype_features

def get_lexical_diversity_features(
    melody: Melody,
    phrase_gap: float = 1.5,
    max_ngram_order: int = DEFAULT_MAX_NGRAM_ORDER,
) -> Dict:
    """Collect lexical-diversity (m-type) features for a melody.

    Corpus-dependent lexical-diversity features (e.g. TF/DF correlations) are
    computed in :func:`get_corpus_features` instead.
    """
    return get_mtype_features(
        melody, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
