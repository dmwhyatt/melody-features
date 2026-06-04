"""Per-melody and parallel processing helpers."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ..core.representations import Melody
from ..utils.warnings import suppress_common_melody_warnings
from .config import Config
from .timing import TIMING_STAT_CATEGORIES, _init_timing_stats


def _features_module():
    from melody_features import features as features_module

    return features_module


def _melody_id_to_num_map(melody_data_list: List[dict]) -> Dict[str, object]:
    """Map melody ID strings to melody_num for IDyOM and export lookups."""
    return {str(m["ID"]): m.get("melody_num") for m in melody_data_list}


def _build_feature_row(
    melody_num: object,
    melody_id: object,
    melody_features: Dict,
    headers: List[str],
    idyom_results_dict: Dict[str, dict],
) -> List:
    """Build one CSV row from computed features and IDyOM batch results."""
    row = [melody_num, melody_id]
    for header in headers[2:]:
        if header.startswith("idyom_"):
            prefix, feature_name = header.split(".", 1)
            idyom_name = prefix[len("idyom_") : -len("_features")]
            value = (
                idyom_results_dict.get(idyom_name, {})
                .get(str(melody_num), {})
                .get(feature_name, 0.0)
            )
            row.append(value)
        else:
            category, feature_name = header.split(".", 1)
            value = melody_features.get(category, {}).get(feature_name, 0.0)
            row.append(value)
    return row


def _record_timing_result(
    result: Tuple,
    headers: List[str],
    id_to_num: Dict[str, object],
    idyom_results_dict: Dict[str, dict],
    all_features: List[List],
    timing_stats: Dict[str, List[float]],
) -> None:
    """Append one melody result row and timing samples."""
    melody_id, melody_features, timings = result
    melody_num = id_to_num.get(str(melody_id))
    row = _build_feature_row(
        melody_num, melody_id, melody_features, headers, idyom_results_dict
    )
    all_features.append(row)
    for category, duration in timings.items():
        timing_stats[category].append(duration)


def process_melody(args):
    """Process a single melody and return its features.

    Parameters
    ----------
    args : tuple
        Tuple containing (melody_data, corpus_stats, idyom_features, phrase_gap, max_ngram_order, key_estimation)

    Returns
    -------
    tuple
        Tuple containing (melody_id, feature_dict, timings)
    """
    suppress_common_melody_warnings()

    features_module = _features_module()
    start_total = time.time()

    melody_data, corpus_stats, _idyom_results_dict, phrase_gap, max_ngram_order, key_estimation = args
    mel = Melody(melody_data)

    timings: Dict[str, float] = {category: 0.0 for category in TIMING_STAT_CATEGORIES}

    start = time.time()
    pitch_features = features_module.get_pitch_features(mel)
    timings["absolute_pitch"] = time.time() - start

    start = time.time()
    pitch_class_features = features_module.get_pitch_class_features(mel)
    timings["pitch_class"] = time.time() - start

    start = time.time()
    interval_features = features_module.get_interval_features(mel)
    timings["pitch_interval"] = time.time() - start

    start = time.time()
    contour_features = features_module.get_contour_features(mel)
    timings["contour"] = time.time() - start

    rhythm_features, rhythm_timings = features_module.collect_rhythm_for_pipeline(mel)
    timings["timing"] = rhythm_timings["timing"]
    timings["inter_onset_interval"] = rhythm_timings["inter_onset_interval"]

    start = time.time()
    tonality_features = features_module.get_tonality_features(mel, key_estimation=key_estimation)
    timings["tonality"] = time.time() - start

    start = time.time()
    metre_features = features_module.get_metre_features(mel)
    timings["metre"] = time.time() - start

    start = time.time()
    expectation_features = features_module.get_expectation_features(mel)
    timings["expectation"] = time.time() - start

    start = time.time()
    complexity_features = features_module.get_complexity_features(
        mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
    timings["complexity"] = time.time() - start

    start = time.time()
    lexical_diversity_features = features_module.get_lexical_diversity_features(
        mel, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
    )
    timings["lexical_diversity"] = time.time() - start

    melody_features = {
        "pitch_features": pitch_features,
        "pitch_class_features": pitch_class_features,
        "interval_features": interval_features,
        "contour_features": contour_features,
        "rhythm_features": rhythm_features,
        "tonality_features": tonality_features,
        "metre_features": metre_features,
        "expectation_features": expectation_features,
        "complexity_features": {**complexity_features, **lexical_diversity_features},
    }

    if corpus_stats:
        start = time.time()
        melody_features["corpus_features"] = features_module.get_corpus_features(
            mel, corpus_stats, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
        )
        timings["corpus"] = time.time() - start

    timings["total"] = time.time() - start_total

    return melody_data["ID"], melody_features, timings


def _setup_parallel_processing(
    melody_data_list: List[dict],
    corpus_stats: Optional[dict],
    idyom_results_dict: Dict[str, dict],
    config: Config,
) -> Tuple[List[str], List, Dict[str, List[float]]]:
    """Set up parallel processing arguments and headers."""
    suppress_common_melody_warnings()
    features_module = _features_module()

    from multiprocessing import cpu_count

    mel = Melody(melody_data_list[0])
    first_features = {
        "pitch_features": features_module.get_pitch_features(mel),
        "pitch_class_features": features_module.get_pitch_class_features(mel),
        "interval_features": features_module.get_interval_features(mel),
        "contour_features": features_module.get_contour_features(mel),
        "rhythm_features": features_module.get_rhythm_features(mel),
        "tonality_features": features_module.get_tonality_features(
            mel, key_estimation=config.key_estimation
        ),
        "metre_features": features_module.get_metre_features(mel),
        "expectation_features": features_module.get_expectation_features(mel),
        "complexity_features": features_module.get_complexity_feature_bundle(
            mel,
            phrase_gap=config.fantastic.phrase_gap,
            max_ngram_order=config.fantastic.max_ngram_order,
        ),
    }

    if corpus_stats:
        first_features["corpus_features"] = features_module.get_corpus_features(
            mel,
            corpus_stats,
            phrase_gap=config.fantastic.phrase_gap,
            max_ngram_order=config.fantastic.max_ngram_order,
        )

    for idyom_name, idyom_results in idyom_results_dict.items():
        if idyom_results:
            sample_id = next(iter(idyom_results))
            for feature in idyom_results[sample_id].keys():
                first_features[f"idyom_{idyom_name}_features.{feature}"] = None
        else:
            first_features[f"idyom_{idyom_name}_features.mean_information_content"] = None

    headers = ["melody_num", "melody_id"]
    for category, features in first_features.items():
        if isinstance(features, dict):
            headers.extend(f"{category}.{feature}" for feature in features.keys())
        elif features is None:
            headers.append(category)

    logger = logging.getLogger("melody_features")
    logger.info("Starting parallel processing...")
    n_cores = cpu_count()
    logger.info(f"Using {n_cores} CPU cores")

    melody_args = [
        (
            melody_data,
            corpus_stats,
            idyom_results_dict,
            config.fantastic.phrase_gap,
            config.fantastic.max_ngram_order,
            config.key_estimation,
        )
        for melody_data in melody_data_list
    ]

    timing_stats = _init_timing_stats()

    return headers, melody_args, timing_stats


def _process_melodies_parallel(
    melody_args: List,
    headers: List[str],
    melody_data_list: List[dict],
    idyom_results_dict: Dict[str, dict],
    timing_stats: Dict[str, List[float]],
) -> List[List]:
    """Process melodies in parallel and collect results."""
    all_features = []
    id_to_num = _melody_id_to_num_map(melody_data_list)
    n_melodies = len(melody_args)
    tqdm_kwargs = {
        "total": n_melodies,
        "unit": "melody",
        "dynamic_ncols": True,
        "mininterval": 0.15,
        "maxinterval": 1.0,
        "miniters": max(1, n_melodies // 200),
        "smoothing": 0.45,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    }
    logger = logging.getLogger("melody_features")

    try:
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp

        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

        logger.info("Parallel processing initiated")

        n_cores = cpu_count()
        chunk_size = max(1, n_melodies // (n_cores * 16))

        with Pool(n_cores) as pool:
            results_iter = pool.imap(process_melody, melody_args, chunksize=chunk_size)
            for result in tqdm(results_iter, desc="Processing melodies", **tqdm_kwargs):
                try:
                    _record_timing_result(
                        result,
                        headers,
                        id_to_num,
                        idyom_results_dict,
                        all_features,
                        timing_stats,
                    )
                except Exception as e:
                    logger.error("Error processing melody: %s", e)

    except Exception as e:
        logger.warning(
            "Parallel processing failed (%s), falling back to sequential processing",
            e,
        )

        for i, args in tqdm(
            enumerate(melody_args),
            desc="Processing melodies (sequential)",
            **tqdm_kwargs,
        ):
            try:
                result = process_melody(args)
                _record_timing_result(
                    result,
                    headers,
                    id_to_num,
                    idyom_results_dict,
                    all_features,
                    timing_stats,
                )
            except Exception as e:
                logger.error("Error processing melody %s: %s", i, e)

    return all_features
