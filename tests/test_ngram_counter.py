import math

import numpy as np
import pytest

from melody_features.ngram_counter import NGramCounter


def test_ngram_counter_total_tokens_uses_unigram_count():
    counter = NGramCounter()
    counter.count_ngrams(["a", "b", "a", "b"], max_order=2)
    assert counter.total_tokens == 4


def test_ngram_counter_mean_entropy_is_orderwise_mean():
    counter = NGramCounter()
    counter.count_ngrams(["a", "b", "a", "b"], max_order=2)

    unigram_entropy = 1.0
    bigram_entropy = -((2 / 3) * math.log2(2 / 3) + (1 / 3) * math.log2(1 / 3))
    expected = (unigram_entropy + bigram_entropy) / 2

    assert np.isclose(counter.mean_entropy, expected)


def test_ngram_counter_mean_productivity_uses_v1_over_n_per_order():
    counter = NGramCounter()
    counter.count_ngrams(["a", "b", "a", "b"], max_order=2)

    # Order 1: v1=0, n=4. Order 2: v1=1, n=3.
    expected = ((0 / 4) + (1 / 3)) / 2
    assert np.isclose(counter.mean_productivity, expected)


def test_ngram_counter_warning_mentions_distinct_ngram_types():
    counter = NGramCounter()
    counter.count_ngrams(["a"], max_order=3)
    with pytest.warns(UserWarning, match="distinct n-gram types <= 1"):
        _ = counter.yules_k
