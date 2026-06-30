"""Input loading helpers for the feature extraction pipeline."""

import json
import logging
import os
from typing import List, Union

from natsort import natsorted

from ..io.midi import list_midi_files, load_midi
from ..core.representations import Melody
from ..utils.validation import _check_is_monophonic

FeatureInput = Union[os.PathLike, List[os.PathLike], List[Melody]]


def _is_melody_list(input: object) -> bool:
    """Return whether ``input`` is a non-empty list of :class:`Melody` objects."""
    return isinstance(input, list) and bool(input) and all(
        isinstance(item, Melody) for item in input
    )


def _finalize_melody_data_list(melody_data_list: List[dict], logger: logging.Logger) -> List[dict]:
    """Assign ``melody_num`` and return the finalized melody data list."""
    melody_data_list = [m for m in melody_data_list if m is not None]
    logger.info("Processing %s melodies", len(melody_data_list))

    if not melody_data_list:
        return []

    for idx, melody_data in enumerate(melody_data_list, 1):
        melody_data["melody_num"] = idx

    return melody_data_list


def _load_melody_data(input: FeatureInput) -> List[dict]:
    """Load and validate melody data from MIDI files, JSON, or Melody objects.

    Parameters
    ----------
    input : FeatureInput
        Path to input directory, JSON file, single MIDI file path, list of MIDI
        file paths, or list of :class:`Melody` objects

    Returns
    -------
    List[dict]
        List of valid monophonic melody data dictionaries

    Raises
    ------
    FileNotFoundError
        If no MIDI files found in directory or list
    ValueError
        If input is not a valid type
    """
    logger = logging.getLogger("melody_features")

    melody_data_list: List[dict] = []

    if _is_melody_list(input):
        for melody in input:
            melody_id = melody.id or "Unknown"
            if not _check_is_monophonic(melody):
                logger.warning("Skipping detected polyphonic melody: %s", melody_id)
                continue
            melody_data_list.append(dict(melody.midi_data))

        if not melody_data_list:
            return []

        for idx, melody_data in enumerate(melody_data_list, 1):
            if not melody_data.get("ID"):
                melody_data["ID"] = f"melody_{idx}"

        return _finalize_melody_data_list(melody_data_list, logger)

    if isinstance(input, list):
        if any(isinstance(item, Melody) for item in input):
            raise ValueError(
                "Input list must contain only Melody objects or only file paths, not a mix"
            )

        midi_files = []
        for file_path in input:
            if isinstance(file_path, (str, os.PathLike)):
                file_path = str(file_path)
                if file_path.lower().endswith(('.mid', '.midi')):
                    midi_files.append(file_path)
                else:
                    logger.warning(f"Skipping non-MIDI file: {file_path}")
            else:
                logger.warning(f"Skipping invalid file path: {file_path}")

        if not midi_files:
            raise FileNotFoundError("No valid MIDI files found in the provided list")

        midi_files = natsorted(midi_files)

    elif os.path.isdir(input):
        midi_files = list_midi_files(input)

    elif isinstance(input, (str, os.PathLike)) and str(input).lower().endswith(('.mid', '.midi')):
        # Handle single MIDI file
        midi_files = [str(input)]

    elif isinstance(input, (str, os.PathLike)) and str(input).endswith(".json"):
        with open(input, encoding="utf-8") as f:
            all_data = json.load(f)

        # Filter for monophonic melodies from the JSON data.
        for melody_data in all_data:
            if melody_data:
                temp_mel = Melody(melody_data)
                if _check_is_monophonic(temp_mel):
                    melody_data_list.append(melody_data)
                else:
                    logger.warning(
                        f"Skipping polyphonic melody from JSON: {melody_data.get('ID', 'Unknown ID')}"
                    )

        return _finalize_melody_data_list(melody_data_list, logger)

    else:
        raise ValueError(
            "Input must be a directory containing MIDI files, a JSON file, a list "
            "of MIDI file paths, a list of Melody objects, or a single MIDI file path. "
            f"Got: {input}"
        )

    for midi_file in midi_files:
        try:
            melody = load_midi(midi_file)
            if melody and _check_is_monophonic(melody):
                melody_data_list.append(melody.midi_data)
            elif melody:
                logger.warning(f"Skipping polyphonic file: {midi_file}")
        except Exception as e:
            logger.error(f"Error importing {midi_file}: {str(e)}")
            continue

    return _finalize_melody_data_list(melody_data_list, logger)
