"""Invocation helpers for decorated melodic feature callables."""

import inspect
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("melody_features")


def invoke_feature(
    func: Callable,
    melody: Any,
    *,
    default_max_ngram_order: int,
    **extra: Any,
) -> Any:
    """Call a feature function, binding melody fields by parameter name."""
    sig = inspect.signature(func)
    if "melody" in sig.parameters:
        return func(melody)

    kwargs = {}
    if "pitches" in sig.parameters:
        kwargs["pitches"] = melody.pitches
    if "starts" in sig.parameters:
        kwargs["starts"] = melody.starts
    if "ends" in sig.parameters:
        kwargs["ends"] = melody.ends
    if "tempo" in sig.parameters:
        kwargs["tempo"] = melody.tempo
    if "tempo_changes" in sig.parameters:
        kwargs["tempo_changes"] = getattr(melody, "tempo_changes", [(0.0, melody.tempo)])
    if "channels" in sig.parameters:
        kwargs["channels"] = getattr(melody, "channels", [1] * len(melody.pitches))
    if "ppqn" in sig.parameters:
        kwargs["ppqn"] = extra.get("ppqn", 480)
    if "corpus_stats" in sig.parameters:
        kwargs["corpus_stats"] = extra.get("corpus_stats")
    if "phrase_gap" in sig.parameters:
        kwargs["phrase_gap"] = extra.get("phrase_gap", 1.5)
    if "max_ngram_order" in sig.parameters:
        kwargs["max_ngram_order"] = extra.get(
            "max_ngram_order",
            default_max_ngram_order,
        )
    return func(**kwargs)


def collect_feature_values(
    feature_functions: Dict[str, Callable],
    melody: Any,
    *,
    default_max_ngram_order: int,
    tuple_suffix: Optional[str] = None,
    numeric_tuple_only: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    """Compute feature functions with shared melody argument dispatch."""
    features: Dict[str, Any] = {}

    for name, func in feature_functions.items():
        try:
            result = invoke_feature(
                func,
                melody,
                default_max_ngram_order=default_max_ngram_order,
                **extra,
            )
            if tuple_suffix and isinstance(result, tuple) and len(result) == 2:
                if numeric_tuple_only and not all(
                    isinstance(x, (int, float)) for x in result
                ):
                    features[name] = result
                else:
                    features[f"{name}_mean"] = result[0]
                    features[f"{name}_{tuple_suffix}"] = result[1]
            else:
                features[name] = result
        except Exception as e:
            logger.warning("Could not compute %s: %s", name, e)
            features[name] = None

    return features
