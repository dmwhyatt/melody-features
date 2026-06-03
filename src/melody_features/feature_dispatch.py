"""Invocation helpers for decorated melodic feature callables."""

import inspect
from typing import Any, Callable, Dict, Optional


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
                features[f"{name}_mean"] = result[0]
                features[f"{name}_{tuple_suffix}"] = result[1]
            else:
                features[name] = result
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            features[name] = None

    return features
