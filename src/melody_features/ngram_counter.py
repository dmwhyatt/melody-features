import math
from collections import Counter
from typing import Dict, Optional


class NGramCounter:
    """A stateful n-gram counter that accumulates counts across multiple sequences."""

    def __init__(self):
        """Initialize an empty n-gram counter."""
        self.ngram_counts = {}
        self._total_tokens = None
        self._freq_spec = None
        self._count_values = None

    def count_ngrams(self, tokens: list, max_order: int = 5) -> None:
        """Count n-grams in the token sequence up to max_order length.

        Parameters
        ----------
        tokens : list
            List of tokens to count n-grams from
        max_order : int, optional
            Maximum n-gram length to count (default: 5)
        """
        # Clear previous counts and caches
        self.ngram_counts = {}
        self._total_tokens = None
        self._freq_spec = None
        self._count_values = None

        # Count n-grams for each possible length up to max_order
        max_length = min(max_order, len(tokens))
        for length in range(1, max_length + 1):
            for i in range(len(tokens) - length + 1):
                ngram = tuple(tokens[i : i + length])
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

    def reset(self) -> None:
        """Reset the n-gram counter to empty."""
        self.ngram_counts = {}
        self._total_tokens = None
        self._freq_spec = None
        self._count_values = None

    def get_counts(self, n: Optional[int] = None) -> Dict:
        """Get the current n-gram counts.

        Parameters
        ----------
        n : int, optional
            If provided, only return counts for n-grams of this length.
            If None, return counts for all n-gram lengths.

        Returns
        -------
        dict
            Dictionary mapping each n-gram to its count
        """
        if n is None:
            return self.ngram_counts.copy()
        return {k: v for k, v in self.ngram_counts.items() if len(k) == n}

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the sequence."""
        if self._total_tokens is None:
            self._total_tokens = sum(self.ngram_counts.values())
        return self._total_tokens

    @property
    def freq_spec(self) -> dict:
        """Frequency spectrum of n-gram counts."""
        if self._freq_spec is None:
            self._freq_spec = Counter(self.ngram_counts.values())
        return self._freq_spec

    @property
    def count_values(self) -> list:
        """List of all n-gram counts."""
        if self._count_values is None:
            self._count_values = list(self.ngram_counts.values())
        return self._count_values

    @property
    def yules_k(self) -> float:
        """Yule's K measure of lexical richness. This feature measures the rate
        at which m-types are repeated in a sequence. Higher values indicate more
        repetitive sequences.
        
        Citation
        --------
        Yule (1944)
        """
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn("Cannot calculate Yule's K for sequence of length <= 1")
                return float("nan")

            n = self.total_tokens
            if n == 0:
                return float("nan")

            s1 = sum(self.count_values)
            s2 = sum(x * x for x in self.count_values)

            if s1 == 0:
                return float("nan")

            return (10000 * (s2 - s1)) / (s1 * s1)
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating Yule's K: {str(e)}")
            return float("nan")

    @property
    def simpsons_d(self) -> float:
        """Simpson's D measure of diversity. This feature measures the rate of m-type
        repetition in a similar way to Yule's K.
        
        Citation
        --------
        Simpson (1949)
        """
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn(
                    "Cannot calculate Simpson's D for sequence of length <= 1"
                )
                return float("nan")

            n = self.total_tokens
            if n == 0:
                return float("nan")

            s2 = sum(x * x for x in self.count_values)
            return s2 / (n * n)
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating Simpson's D: {str(e)}")
            return float("nan")

    @property
    def sichels_s(self) -> float:
        """Sichel's S measure corresponds to the proportion of m-types that occur exactly twice in a sequence.
        Higher values indicate a greater amount of m-types that occur exactly twice.
        
        Citation
        --------
        Sichel (1975)
        """
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn("Cannot calculate Sichel's S for sequence of length <= 1")
                return float("nan")

            v = len(self.ngram_counts)
            if v == 0:
                return float("nan")

            v2 = self.freq_spec.get(2, 0)
            return v2 / v if v > 0 else float("nan")
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating Sichel's S: {str(e)}")
            return float("nan")

    @property
    def honores_h(self) -> float:
        """Honoré's H measure corresponds to the observation that the number of tokens occuring exactly once 
        in a sequence is logarithmically related to the total number of tokens in the sequence.
        
        Citation
        --------
        Honoré (1979)
        """
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn("Cannot calculate Honoré's H for sequence of length <= 1")
                return float("nan")

            n = self.total_tokens
            v = len(self.ngram_counts)
            v1 = self.freq_spec.get(1, 0)

            if n == 0 or v == 0:
                return float("nan")

            return 100 * math.log(n) / (1 - v1 / v) if v1 != v else float("nan")
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating Honoré's H: {str(e)}")
            return float("nan")

    @property
    def mean_entropy(self) -> float:
        """Calculate the zeroth-order base-2 entropy of m-types across all n-gram lengths."""
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn(
                    "Cannot calculate mean entropy for sequence of length <= 1"
                )
                return float("nan")

            n = self.total_tokens
            if n == 0:
                return float("nan")

            probs = [count / n for count in self.count_values]
            return -sum(p * math.log(p) for p in probs)
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating mean entropy: {str(e)}")
            return float("nan")

    @property
    def mean_productivity(self) -> float:
        """Mean productivity is defined as the mean of the number of types
        occurring only once divided by the total number of tokens. The types occurring
        only once in a sequence are known as hapax legomena."""
        try:
            if len(self.count_values) <= 1:
                import warnings

                warnings.warn(
                    "Cannot calculate mean productivity for sequence of length <= 1"
                )
                return float("nan")

            v = len(self.ngram_counts)
            v1 = self.freq_spec.get(1, 0)

            if v == 0:
                return float("nan")

            return v1 / v if v > 0 else float("nan")
        except Exception as e:
            import warnings

            warnings.warn(f"Error calculating mean productivity: {str(e)}")
            return float("nan")
