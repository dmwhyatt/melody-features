"""IDyOM runner helpers used by the public feature facade."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from natsort import natsorted

from ..feature_definitions.tonality import infer_key_from_pitches
from ..io.midi import import_midi
from ..representations import Melody
from .config import _DEFAULT_IDYOM_CONFIGS, _resolve_idyom_corpus


def _melody_idyom_input_directory(melody: Melody) -> tuple[str, list[str]]:
    """Build a one-melody MIDI directory for IDyOM. Returns (directory, paths to remove)."""
    import pretty_midi

    cleanup_paths: list[str] = []
    melody_path = melody.id
    if (
        melody_path
        and str(melody_path).lower().endswith((".mid", ".midi"))
        and os.path.isfile(str(melody_path))
    ):
        temp_dir = tempfile.mkdtemp(prefix="idyom_melody_")
        cleanup_paths.append(temp_dir)
        dest = os.path.join(temp_dir, os.path.basename(str(melody_path)))
        shutil.copy2(str(melody_path), dest)
        return temp_dir, cleanup_paths

    temp_dir = tempfile.mkdtemp(prefix="idyom_melody_")
    cleanup_paths.append(temp_dir)
    temp_midi = os.path.join(temp_dir, "melody.mid")
    pm = pretty_midi.PrettyMIDI(initial_tempo=melody.tempo)
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in zip(melody.pitches, melody.starts, melody.ends):
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(instrument)
    pm.write(temp_midi)
    return temp_dir, cleanup_paths

def _idyom_mean_information_content(
    melody: Melody,
    config_key: str,
    corpus: Optional[os.PathLike] = None,
    key_estimation: str = "infer_if_necessary",
) -> float:
    """Run IDyOM for one melody and return mean information content."""
    if config_key not in _DEFAULT_IDYOM_CONFIGS:
        raise ValueError(
            f"Unknown IDyOM configuration {config_key!r}; "
            f"expected one of {sorted(_DEFAULT_IDYOM_CONFIGS)}"
        )

    idyom_config = _DEFAULT_IDYOM_CONFIGS[config_key]
    idyom_corpus = _resolve_idyom_corpus(idyom_config, override=corpus)

    from melody_features import features as features_module

    input_directory, cleanup_paths = _melody_idyom_input_directory(melody)
    try:
        idyom_results = features_module.get_idyom_results(
            input_directory,
            idyom_config.target_viewpoints,
            idyom_config.source_viewpoints,
            idyom_config.models,
            idyom_config.ppm_order,
            idyom_corpus,
            f"IDyOM_{config_key}_Results",
            key_estimation,
        )
    finally:
        for path in cleanup_paths:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    melody_features = idyom_results.get("1")
    if not melody_features or "mean_information_content" not in melody_features:
        raise RuntimeError(
            f"IDyOM did not return mean information content for configuration {config_key!r}"
        )
    return float(melody_features["mean_information_content"])

def get_idyom_results(
    input_directory,
    idyom_target_viewpoints,
    idyom_source_viewpoints,
    models,
    ppm_order,
    corpus_path,
    experiment_name="IDyOM_Feature_Set_Results",
    key_estimation="infer_if_necessary",
) -> dict:
    logger = logging.getLogger("melody_features")
    """Run IDyOM on the input MIDI directory and return mean information content for each melody.
    Uses the parameters supplied from Config dataclass to control IDyOM behaviour.

    Parameters
    ----------
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"

    Returns
    -------
    dict
        A dictionary mapping melody IDs to their mean information content.
    """
    logger = logging.getLogger("melody_features")
    from melody_features import features as features_module

    # Set default IDyOM viewpoints if not provided.
    if idyom_target_viewpoints is None:
        idyom_target_viewpoints = ["cpitch"]

    if idyom_source_viewpoints is None:
        idyom_source_viewpoints = [("cpint", "cpintfref")]

    logger.info(
        f"Creating temporary MIDI files with key_estimation='{key_estimation}' for IDyOM processing..."
    )
    temp_dir = tempfile.mkdtemp(prefix="idyom_key_")
    original_input_dir = input_directory
    input_directory = features_module.create_temp_midi_with_key_signature(input_directory, temp_dir, key_estimation)

    temp_files = glob.glob(os.path.join(input_directory, "*.mid"))
    temp_files.extend(glob.glob(os.path.join(input_directory, "*.midi")))

    try:
        # Try without pretraining first to see if that's the issue
        dat_file_path = features_module.run_idyom(
            input_directory,
            pretraining_path=corpus_path,  # Use the actual corpus path
            output_dir=".",
            experiment_name=experiment_name,
            target_viewpoints=idyom_target_viewpoints,
            source_viewpoints=idyom_source_viewpoints,
            models=models,
            detail=2,
            ppm_order=ppm_order,
        )

        if not dat_file_path:
            logger.warning(
                "run_idyom did not produce an output file. Skipping IDyOM features."
            )
            return {}

        # Get a naturally sorted list of MIDI files to match IDyOM's processing order.
        # Since we created temp files, IDyOM processed the temp directory files
        if temp_dir:
            # Use the temp directory files since that's what IDyOM actually processed
            midi_files = natsorted(glob.glob(os.path.join(input_directory, "*.mid")))
        else:
            # Use original input directory for file mapping if no temp files were created
            midi_files = natsorted(glob.glob(os.path.join(original_input_dir, "*.mid")))
            midi_files.extend(
                natsorted(glob.glob(os.path.join(original_input_dir, "*.midi")))
            )

        idyom_results = {}
        try:
            with open(dat_file_path, "r", encoding="utf-8") as f:
                # Read header to determine column names
                header_line = next(f).strip()
                header_parts = header_line.split()

                logger.debug(f"IDyOM header: {header_line}")
                logger.debug(f"IDyOM header parts: {header_parts}")

                # Find the column index for information content
                # The dat file typically has: melody.id melody.name information.content
                # We want to extract the last column (information content)
                if len(header_parts) < 2:
                    logger.error(f"Invalid header format: {header_line}")
                    return {}

                # The last column should be the information content value
                info_content_col_idx = len(header_parts) - 1

                logger.debug(f"Will extract information content from column index {info_content_col_idx} (header has {len(header_parts)} columns)")

                line_count = 0
                for line in f:
                    line_count += 1
                    parts = line.strip().split()

                    if len(parts) < 2:
                        logger.warning(f"Skipping malformed line: {line.strip()}")
                        continue  # Skip malformed lines

                    try:
                        # IDyOM's melody ID is a 1-based index.
                        melody_idx = int(parts[0]) - 1

                        if 0 <= melody_idx < len(midi_files):
                            # Map the index to the melody number (1-based index)
                            melody_id = str(melody_idx + 1)

                            logger.debug(f"Processing melody {melody_id}: parts={parts}, len={len(parts)}")

                            # Extract the information content value from the correct column
                            if len(parts) <= info_content_col_idx:
                                logger.warning(
                                    f"Not enough columns in line for melody {melody_id}. Expected at least {info_content_col_idx + 1}, got {len(parts)}. Parts: {parts}"
                                )
                                continue

                            try:
                                feature_value = float(parts[info_content_col_idx])
                                features = {"mean_information_content": feature_value}
                                idyom_results[melody_id] = features
                                logger.debug(f"Extracted mean_information_content={feature_value} for melody {melody_id}")
                            except (ValueError, IndexError) as e:
                                logger.warning(
                                    f"Could not parse information content at index {info_content_col_idx} for melody {melody_id}: {e}, parts={parts}"
                                )
                        else:
                            logger.warning(
                                f"IDyOM returned an out-of-bounds index: {parts[0]} (max: {len(midi_files)-1})"
                            )
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Could not parse line in IDyOM output: '{line.strip()}'. Error: {e}"
                        )

            os.remove(dat_file_path)

        except FileNotFoundError:
            logger.warning(
                f"IDyOM output file not found at {dat_file_path}. Skipping IDyOM features."
            )
            return {}
        except Exception as e:
            logger.error(
                f"Error parsing IDyOM output file: {e}. Skipping IDyOM features."
            )
            if os.path.exists(dat_file_path):
                os.remove(dat_file_path)
            return {}

        return idyom_results

    finally:
        # Clean up temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

def to_mido_key_string(key_name):
    """Convert key name to mido key string format."""
    key_name = key_name.strip().lower()

    # mido only allows certain key names, so we need to catch enharmonics
    # see https://mido.readthedocs.io/en/stable/meta_message_types.html
    enharmonic_map = {
        "a#": "bb",
        "a# major": "bb major",
        "d#": "eb",
        "d# major": "eb major",
        "g#": "ab",
        "g# major": "ab major",
        "c#": "db",
        "c# major": "db major",
        "f#": "gb",
        "f# major": "gb major",
    }

    # Apply enharmonic mapping
    if key_name in enharmonic_map:
        key_name = enharmonic_map[key_name]

    if "minor" in key_name:
        root = key_name.replace(" minor", "").replace("min", "").strip().capitalize()
        return f"{root}m"
    else:
        root = key_name.replace(" major", "").strip().capitalize()
        return root

def create_temp_midi_with_key_signature(
    input_directory: str,
    temp_dir: str,
    key_estimation: str = "infer_if_necessary",
    key_finding_algorithm: Literal["krumhansl_schmuckler"] = "krumhansl_schmuckler"
) -> str:
    """
    Create temporary MIDI files with key signatures for IDyOM processing.

    Parameters
    ----------
    input_directory : str
        Path to the input directory containing MIDI files
    temp_dir : str
        Path to the temporary directory to create the modified MIDI files
    key_estimation : str, optional
        Key estimation strategy: "always_read_from_file", "infer_if_necessary", or "always_infer"
    key_finding_algorithm : Literal["krumhansl_schmuckler"], optional
        Key-finding algorithm to use when inferring key. Default is "krumhansl_schmuckler".

    Returns
    -------
    str
        Path to the temporary directory containing MIDI files with key signatures
    """
    logger = logging.getLogger("melody_features")
    from mido import MetaMessage, MidiFile

    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # Get all MIDI files from input directory
    midi_files = glob.glob(os.path.join(input_directory, "*.mid"))
    midi_files.extend(glob.glob(os.path.join(input_directory, "*.midi")))

    logger.info(
        f"Processing {len(midi_files)} MIDI files with key_estimation='{key_estimation}'..."
    )

    successful_copies = 0
    for midi_file in midi_files:
        try:
            # First, try to copy the original file as a fallback
            # Always save with .mid extension for IDyOM compatibility
            base_filename = os.path.splitext(os.path.basename(midi_file))[0]
            output_filename = base_filename + ".mid"
            output_path = os.path.join(temp_dir, output_filename)
            shutil.copy2(midi_file, output_path)
            successful_copies += 1

            # Handle key signature based on key_estimation strategy
            try:
                mid = MidiFile(midi_file)

                has_key_signature = False
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "key_signature":
                            has_key_signature = True
                            break
                    if has_key_signature:
                        break

                should_apply_estimated_key = False

                if key_estimation == "always_read_from_file":
                    if not has_key_signature:
                        raise ValueError(f"No key signature found in MIDI file: {midi_file}")
                    continue
                elif key_estimation == "infer_if_necessary":
                    should_apply_estimated_key = not has_key_signature
                elif key_estimation == "always_infer":
                    should_apply_estimated_key = True
                else:
                    raise ValueError(f"Invalid key_estimation value: {key_estimation}")

                if should_apply_estimated_key:
                    midi_dict = import_midi(midi_file)
                    if midi_dict is None:
                        logger.warning(f"Could not import {midi_file}, using original file")
                        continue

                    melody = Melody(midi_dict)
                    if not melody.pitches:
                        logger.warning(
                            f"No pitches found in {midi_file}, using original file"
                        )
                        continue

                    key_name, mode = infer_key_from_pitches(
                        melody.pitches,
                        algorithm=key_finding_algorithm
                    )
                    if key_name and mode:
                        detected_key = f"{key_name} {mode}"
                    else:
                        logger.warning(
                            f"Could not infer key for {midi_file}, using original file"
                        )
                        continue

                    mido_key = to_mido_key_string(detected_key)

                    for track in mid.tracks:
                        track[:] = [
                            msg for msg in track if not (msg.type == "key_signature")
                        ]

                    key_msg = MetaMessage("key_signature", key=mido_key, time=0)
                    mid.tracks[0].insert(0, key_msg)

                    mid.save(output_path)

            except Exception as e:
                logger.warning(
                    f"Could not add key signature to {midi_file}: {str(e)}, using original file"
                )

        except Exception as e:
            logger.error(f"Could not copy {midi_file}: {str(e)}")
            continue

    created_files = glob.glob(os.path.join(temp_dir, "*.mid"))

    logger.info(
        f"Successfully created {len(created_files)} files in temporary directory"
    )

    return temp_dir

def _run_idyom_analysis(
    input: Union[os.PathLike, List[os.PathLike]], config: Config
) -> Dict[str, dict]:
    """Run IDyOM analysis for all configurations.

    Parameters
    ----------
    input : Union[os.PathLike, List[os.PathLike]]
        Path to input directory, list of MIDI file paths, or single MIDI file path
    config : Config
        Configuration object containing IDyOM settings

    Returns
    -------
    Dict[str, dict]
        Dictionary mapping IDyOM configuration names to their results
    """
    logger = logging.getLogger("melody_features")
    idyom_results_dict = {}
    from melody_features import features as features_module

    if isinstance(input, list):
        temp_dir = tempfile.mkdtemp(prefix="idyom_input_")
        try:
            for i, file_path in enumerate(input):
                if isinstance(file_path, (str, os.PathLike)) and str(file_path).lower().endswith(('.mid', '.midi')):
                    import shutil
                    file_ext = os.path.splitext(str(file_path))[1]
                    temp_file_path = os.path.join(temp_dir, f"file_{i+1:04d}{file_ext}")
                    shutil.copy2(str(file_path), temp_file_path)

            idyom_input_path = temp_dir
        except Exception as e:
            logger.error(f"Error creating temporary directory for IDyOM: {e}")
            return {}
    elif os.path.isdir(input):
        idyom_input_path = input
    elif isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi')):
        temp_dir = tempfile.mkdtemp(prefix="idyom_input_")
        try:
            import shutil
            file_ext = os.path.splitext(str(input))[1]
            temp_file_path = os.path.join(temp_dir, f"file_0001{file_ext}")
            shutil.copy2(str(input), temp_file_path)
            idyom_input_path = temp_dir
        except Exception as e:
            logger.error(f"Error creating temporary directory for IDyOM: {e}")
            return {}
    else:
        logger.error(f"Unsupported input type for IDyOM: {type(input)}")
        return {}

    for idyom_name, idyom_config in config.idyom.items():
        idyom_corpus = _resolve_idyom_corpus(idyom_config, config_corpus=config.corpus)
        logger.info(
            f"Running IDyOM analysis for '{idyom_name}' with corpus: {idyom_corpus}"
        )

        try:
            idyom_results = features_module.get_idyom_results(
                idyom_input_path,
                idyom_config.target_viewpoints,
                idyom_config.source_viewpoints,
                idyom_config.models,
                idyom_config.ppm_order,
                idyom_corpus,
                f"IDyOM_{idyom_name}_Results",
                config.key_estimation,
            )
            idyom_results_dict[idyom_name] = idyom_results
        except Exception as e:
            logger.error(f"Failed to run IDyOM for '{idyom_name}': {e}")
            idyom_results_dict[idyom_name] = {}

    # Clean up temporary directory if it was created
    if isinstance(input, list) or (isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi'))):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temporary IDyOM directory: {e}")

    return idyom_results_dict

def _cleanup_idyom_temp_output():
    """Clean up any existing IDyOM temporary output directory to prevent conflicts."""
    import shutil
    from pathlib import Path

    idyom_temp_dir = Path("idyom_temp_output")
    if idyom_temp_dir.exists():
        logger = logging.getLogger("melody_features")
        logger.info(f"Cleaning up existing IDyOM temporary directory: {idyom_temp_dir}")
        try:
            shutil.rmtree(idyom_temp_dir)
            logger.info("Successfully cleaned up IDyOM temporary directory")
        except Exception as e:
            logger.warning(f"Could not clean up IDyOM temporary directory: {e}")
