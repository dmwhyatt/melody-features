"""Per-melody and parallel processing helpers."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ..representations import Melody
from .timing import TIMING_STAT_CATEGORIES, _init_timing_stats


def _features_module():
    from melody_features import features as features_module

    return features_module


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
    # Suppress warnings in worker processes
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

    features_module = _features_module()
    start_total = time.time()

    melody_data, corpus_stats, idyom_results_dict, phrase_gap, max_ngram_order, key_estimation = args
    mel = Melody(melody_data)

    # Time each taxonomy category separately for logging
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

    start = time.time()
    timing_features = features_module.get_timing_features(mel)
    timings["timing"] = time.time() - start

    start = time.time()
    ioi_features = features_module.get_inter_onset_interval_features(mel)
    timings["inter_onset_interval"] = time.time() - start

    rhythm_features = {
        **timing_features,
        **ioi_features,
        **features_module.get_metric_accent_features(mel),
    }

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

    # Add corpus features only if corpus stats are available
    if corpus_stats:
        start = time.time()
        melody_features["corpus_features"] = features_module.get_corpus_features(
            mel, corpus_stats, phrase_gap=phrase_gap, max_ngram_order=max_ngram_order
        )
        timings["corpus"] = time.time() - start

    # Add pre-computed IDyOM features if available for this melody's ID
    melody_id_str = str(melody_data["melody_num"])

    # Handle IDyOM results dictionary (multiple configurations)
    idyom_features = {}
    if idyom_results_dict:
        for idyom_name, idyom_results in idyom_results_dict.items():
            if idyom_results and melody_id_str in idyom_results:
                for feature_key, feature_value in idyom_results[melody_id_str].items():
                    # Match the header format: idyom_{idyom_name}_features.{feature_key}
                    idyom_features[f"idyom_{idyom_name}_features.{feature_key}"] = feature_value
            else:
                # Add fallback value for this config if results not found
                idyom_features[f"idyom_{idyom_name}_features.mean_information_content"] = -1

    if idyom_features:
        melody_features["idyom_features"] = idyom_features

    timings["total"] = time.time() - start_total

    return melody_data["ID"], melody_features, timings

def _setup_parallel_processing(
    melody_data_list: List[dict],
    corpus_stats: Optional[dict],
    idyom_results_dict: Dict[str, dict],
    config: Config,
) -> Tuple[List[str], List, Dict[str, List[float]]]:
    """Set up parallel processing arguments and headers.

    Parameters
    ----------
    melody_data_list : List[dict]
        List of melody data dictionaries
    corpus_stats : Optional[dict]
        Corpus statistics dictionary
    idyom_results_dict : Dict[str, dict]
        Dictionary of IDyOM results
    config : Config
        Configuration object

    Returns
    -------
    Tuple[List[str], List, Dict[str, List[float]]]
        Headers, melody arguments, and timing statistics dictionary
    """
    features_module = _features_module()

    # Suppress warnings at the system level
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

    from multiprocessing import cpu_count

    # Process first melody to get header structure
    mel = Melody(melody_data_list[0])
    first_features = {
        "pitch_features": features_module.get_pitch_features(mel),
        "pitch_class_features": features_module.get_pitch_class_features(mel),
        "interval_features": features_module.get_interval_features(mel),
        "contour_features": features_module.get_contour_features(mel),
        "rhythm_features": features_module.get_rhythm_features(mel),
        "tonality_features": features_module.get_tonality_features(mel, key_estimation=config.key_estimation),
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

    # Add IDyOM features for each config to the header
    for idyom_name, idyom_results in idyom_results_dict.items():
        if idyom_results:
            sample_id = next(iter(idyom_results))
            for feature in idyom_results[sample_id].keys():
                first_features[f"idyom_{idyom_name}_features.{feature}"] = None
        else:
            # Add header for fallback value even if no results
            first_features[f"idyom_{idyom_name}_features.mean_information_content"] = None

    # Create header by flattening feature names
    headers = ["melody_num", "melody_id"]
    for category, features in first_features.items():
        if isinstance(features, dict):
            headers.extend(f"{category}.{feature}" for feature in features.keys())
        elif features is None:
            # Already prefixed for IDyOM
            headers.append(category)

    logger = logging.getLogger("melody_features")
    logger.info("Starting parallel processing...")
    # Create pool of workers
    n_cores = cpu_count()
    logger.info(f"Using {n_cores} CPU cores")

    # Prepare arguments for parallel processing
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
    """Process melodies in parallel and collect results.

    Parameters
    ----------
    melody_args : List
        Arguments for parallel processing
    headers : List[str]
        CSV headers
    melody_data_list : List[dict]
        List of melody data dictionaries
    idyom_results_dict : Dict[str, dict]
        Dictionary of IDyOM results
    timing_stats : Dict[str, List[float]]
        Timing statistics dictionary

    Returns
    -------
    List[List]
        List of feature rows
    """
    all_features = []

    try:
        # Try to use multiprocessing
        from multiprocessing import Pool, cpu_count
        import multiprocessing as mp

        # Set start method to 'fork' for better compatibility
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass  # Start method already set

        logger = logging.getLogger("melody_features")
        logger.info("Parallel processing initiated")

        n_cores = cpu_count()
        chunk_size = max(1, len(melody_args) // (n_cores * 4))

        with Pool(n_cores) as pool:
            # Use tqdm to show progress as melodies are processed
            with tqdm(
                total=len(melody_args),
                desc="Processing melodies",
                unit="melody",
                ncols=80,
                mininterval=0.5,
                maxinterval=2.0,
                miniters=1,
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                for result in pool.imap(process_melody, melody_args, chunksize=chunk_size):
                    try:
                        melody_id, melody_features, timings = result
                        melody_num = None
                        for m in melody_data_list:
                            if str(m["ID"]) == str(melody_id):
                                melody_num = m.get("melody_num", None)
                                break
                        row = [melody_num, melody_id]
                        for header in headers[2:]:  # Skip melody_num and melody_id headers
                            if header.startswith("idyom_"):
                                prefix, feature_name = header.split(".", 1)
                                idyom_name = prefix[len("idyom_") : -len("_features")]
                                # Use melody_num for IDyOM lookup since IDyOM results are indexed by melody number
                                value = (
                                    idyom_results_dict.get(idyom_name, {})
                                    .get(str(melody_num), {})
                                    .get(feature_name, 0.0)
                                )
                                row.append(value)
                            else:
                                category, feature_name = header.split(".", 1)
                                value = melody_features.get(category, {}).get(
                                    feature_name, 0.0
                                )
                                row.append(value)
                        all_features.append(row)

                        for category, duration in timings.items():
                            timing_stats[category].append(duration)

                        # Update progress bar
                        pbar.update(1)

                    except Exception as e:
                        logger = logging.getLogger("melody_features")
                        logger.error(f"Error processing melody: {str(e)}")
                        pbar.update(1)  # Still update progress even on error
                        continue

    except Exception as e:
        # Fall back to sequential processing if multiprocessing fails
        logger = logging.getLogger("melody_features")
        logger.warning(f"Parallel processing failed ({str(e)}), falling back to sequential processing")

        with tqdm(
            total=len(melody_args),
            desc="Processing melodies (sequential)",
            unit="melody",
            ncols=80,
            mininterval=0.5,
            maxinterval=2.0,
            miniters=1,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for i, args in enumerate(melody_args):
                try:
                    result = process_melody(args)
                    melody_id, melody_features, timings = result

                    melody_num = None
                    for m in melody_data_list:
                        if str(m["ID"]) == str(melody_id):
                            melody_num = m.get("melody_num", None)
                            break
                    row = [melody_num, melody_id]

                    for header in headers[2:]:  # Skip melody_num and melody_id headers
                        if header.startswith("idyom_"):
                            prefix, feature_name = header.split(".", 1)
                            idyom_name = prefix[len("idyom_") : -len("_features")]
                            # Use melody_num for IDyOM lookup since IDyOM results are indexed by melody number
                            value = (
                                idyom_results_dict.get(idyom_name, {})
                                .get(str(melody_num), {})
                                .get(feature_name, 0.0)
                            )
                            row.append(value)
                        else:
                            category, feature_name = header.split(".", 1)
                            value = melody_features.get(category, {}).get(
                                feature_name, 0.0
                            )
                            row.append(value)
                    all_features.append(row)

                    # Update timing statistics
                    for category, duration in timings.items():
                        timing_stats[category].append(duration)

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing melody {i}: {str(e)}")
                    pbar.update(1)  # Still update progress even on error
                    continue

    return all_features
