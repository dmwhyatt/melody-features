# Suppress warnings from external libraries BEFORE any imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)

"""
Module for computing corpus-based features from melodic n-grams, similar to FANTASTIC's
implementation. This module handles the corpus analysis and saves statistics to JSON.
The actual feature calculations are handled in features.py.
"""
import json
import logging
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

from natsort import natsorted
from tqdm import tqdm

from melody_features.import_mid import import_midi
from melody_features.melody_tokenizer import FantasticTokenizer
from melody_features.representations import Melody, read_midijson

# Corpus paths for easy access
try:
    essen_corpus = resources.files("melody_features") / "corpora" / "Essen_Corpus"
except ImportError:
    # Fallback for development or when package is not installed
    essen_corpus = Path(__file__).parent / "corpora" / "Essen_Corpus"





def process_melody_ngrams(args) -> set:
    """Process n-grams for a single melody.

    Parameters
    ----------
    args : tuple
        Tuple containing (melody, n_range)

    Returns
    -------
    set
        Set of unique n-grams found in the melody
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

    melody, n_range = args
    tokenizer = FantasticTokenizer()

    # Segment the melody first
    segments = tokenizer.segment_melody(melody, phrase_gap=1.5, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, segment.starts, segment.ends
        )
        all_tokens.extend(segment_tokens)

    unique_ngrams = set()
    for n in range(n_range[0], n_range[1] + 1):
        # Count n-grams in the combined tokens
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i : i + n])
            unique_ngrams.add(ngram)

    return unique_ngrams


def compute_corpus_ngrams(
    melodies: List[Melody], n_range: Tuple[int, int] = (1, 6)
) -> Dict:
    """Compute n-gram frequencies across the entire corpus using multiprocessing.

    Parameters
    ----------
    melodies : List[Melody]
        List of Melody objects to analyze
    n_range : Tuple[int, int]
        Range of n-gram lengths to consider (min, max)

    Returns
    -------
    Dict
        Dictionary containing corpus-wide n-gram statistics
    """
    # Suppress warnings at the system level
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="pkg_resources"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
    )

    # Process melodies sequentially
    results = []
    for melody in tqdm(melodies, desc="Computing n-grams sequentially"):
        result = process_melody_ngrams((melody, n_range))
        results.append(result)

    # Count document frequency (number of melodies containing each n-gram)
    doc_freq = Counter()
    for ngrams in results:
        doc_freq.update(ngrams)

    # Format results for JSON serialization
    frequencies = {"document_frequencies": {}}
    for k, v in doc_freq.items():
        frequencies["document_frequencies"][str(k)] = {"count": v}

    return {
        "document_frequencies": frequencies["document_frequencies"],
        "corpus_size": len(melodies),
        "n_range": n_range,
    }


def save_corpus_stats(stats: Dict, filename: str) -> None:
    """Save corpus statistics to a JSON file.

    Parameters
    ----------
    stats : Dict
        Corpus statistics from compute_corpus_ngrams
    filename : str
        Path to save JSON file
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename = filename + ".json"

    # Ensure the directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_corpus_stats(filename: str) -> Dict:
    """Load corpus statistics from a JSON file.

    Parameters
    ----------
    filename : str
        Path to JSON file. If no extension is provided, .json will be added.

    Returns
    -------
    Dict
        Corpus statistics dictionary
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename = filename + ".json"

    with open(filename, encoding="utf-8") as f:
        stats = json.load(f)

    return stats


def load_melody(idx: int, filename: str) -> Melody:
    """Load a single melody from a JSON file.

    Parameters
    ----------
    idx : int
        Index of melody to load
    filename : str
        Path to JSON file

    Returns
    -------
    Melody
        Loaded melody object
    """
    melody_data = read_midijson(filename)
    if idx >= len(melody_data):
        raise IndexError(
            f"Index {idx} is out of range for file with {len(melody_data)} melodies"
        )
    return Melody(melody_data[idx], tempo=100)


def load_midi_melody(midi_path: str) -> Melody:
    """Load a melody from a MIDI file.

    Parameters
    ----------
    midi_path : str
        Path to MIDI file

    Returns
    -------
    Melody or None
        Loaded melody object, or None if the file could not be loaded
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

    logger = logging.getLogger("melody_features")
    try:
        melody_data = import_midi(midi_path)
        if melody_data is None:
            return None
        return Melody(melody_data, tempo=100)
    except Exception as e:
        logger.warning(f"Error creating Melody object from {midi_path}: {str(e)}")
        return None


def load_melodies_from_directory(
    directory: str, file_type: str = "json"
) -> List[Melody]:
    """Load melodies from a directory containing either JSON or MIDI files.

    Parameters
    ----------
    directory : str
        Path to directory containing melody files
    file_type : str
        Type of files to load ("json" or "midi")

    Returns
    -------
    List[Melody]
        List of loaded melody objects
    """
    logger = logging.getLogger("melody_features")
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if file_type == "json":
        # For JSON, we expect a single file containing multiple melodies
        json_files = list(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")
        if len(json_files) > 1:
            raise ValueError(
                f"Multiple JSON files found in {directory}. Please specify a single file."
            )

        melody_data = read_midijson(str(json_files[0]))
        num_melodies = len(melody_data)
        logger.info(f"Found {num_melodies} melodies in {json_files[0]}")

        # Load melodies sequentially
        melodies = []
        for i in tqdm(range(num_melodies), desc="Loading melodies sequentially"):
            melody = load_melody(i, str(json_files[0]))
            melodies.append(melody)

    elif file_type == "midi":
        # For MIDI, we expect multiple files, each containing one melody
        midi_files = list(directory.glob("*.mid")) + list(directory.glob("*.midi"))
        if not midi_files:
            raise FileNotFoundError(f"No MIDI files found in {directory}")

        logger.info(f"Found {len(midi_files)} MIDI files")

        # Load melodies sequentially
        melodies = []
        for midi_file in tqdm(midi_files, desc="Loading MIDI files sequentially"):
            melody = load_midi_melody(str(midi_file))
            melodies.append(melody)
    else:
        raise ValueError("file_type must be either 'json' or 'midi'")

    return melodies


def make_corpus_stats(midi_dir: str, output_file: str) -> None:
    """Process a directory of MIDI files and save corpus statistics.

    Parameters
    ----------
    midi_dir : str
        Path to directory containing MIDI files
    output_file : str
        Path where to save the corpus statistics JSON file
    """
    logger = logging.getLogger("melody_features")
    # Load melodies from MIDI files
    melodies = load_melodies_from_directory(midi_dir, file_type="midi")
    # Filter out None values
    melodies = [m for m in melodies if m is not None]
    if not melodies:
        raise ValueError(
            "No valid melodies could be processed from the directory. Check if the files are valid MIDI files."
        )
    logger.info(f"Processing {len(melodies)} valid melodies")

    # Compute corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies)

    # Save to JSON
    save_corpus_stats(corpus_stats, output_file)

    # Load and verify
    loaded_stats = load_corpus_stats(output_file)
    logger.info("Corpus statistics saved and loaded successfully.")
    logger.info(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    logger.info(f"N-gram lengths: {loaded_stats['n_range']}")


def make_corpus_stats_from_json(
    json_file: str, output_file: str, n_range: Tuple[int, int] = (1, 6)
) -> None:
    """Process a JSON file containing melody data and save corpus statistics.

    Parameters
    ----------
    json_file : str
        Path to JSON file containing melody data
    output_file : str
        Path where to save the corpus statistics JSON file
    n_range : Tuple[int, int], optional
        Range of n-gram lengths to consider (min, max), by default (1, 6)
    """
    logger = logging.getLogger("melody_features")
    # Load melody data from JSON
    logger.info(f"Loading melodies from JSON file: {json_file}")
    melody_data = read_midijson(json_file)

    if not melody_data:
        logger.error("No melody data found in JSON file")
        exit(1)

    logger.info(f"Found {len(melody_data)} melodies in JSON file")

    # Convert to Melody objects
    melodies = []
    for i, data in enumerate(tqdm(melody_data, desc="Converting to Melody objects")):
        try:
            melody = Melody(data, tempo=100)
            melodies.append(melody)
        except Exception as e:
            logger.warning(f"Error creating Melody object from entry {i}: {str(e)}")
            continue

    # Filter out None values
    melodies = [m for m in melodies if m is not None]
    if not melodies:
        raise ValueError("No valid melodies could be processed from the JSON file.")
    logger.info(f"Processing {len(melodies)} valid melodies")

    # Compute corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies, n_range)

    # Save to JSON
    save_corpus_stats(corpus_stats, output_file)

    # Load and verify
    loaded_stats = load_corpus_stats(output_file)
    logger.info("Corpus statistics saved and loaded successfully.")
    logger.info(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    logger.info(f"N-gram lengths: {loaded_stats['n_range']}")


def get_corpus_path(corpus_name: str) -> Path:
    """Get the path to a bundled corpus.

    Parameters
    ----------
    corpus_name : str
        Name of the corpus. Currently supports: 'essen'

    Returns
    -------
    Path
        Path to the corpus directory

    Raises
    ------
    ValueError
        If the corpus name is not recognized
    FileNotFoundError
        If the corpus directory does not exist
    """
    corpus_paths = {"essen": essen_corpus}

    if corpus_name not in corpus_paths:
        available = ", ".join(corpus_paths.keys())
        raise ValueError(
            f"Unknown corpus '{corpus_name}'. Available corpora: {available}"
        )

    corpus_path = corpus_paths[corpus_name]

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    return corpus_path


def get_corpus_files(corpus_name: str, max_files: int = None) -> List[Path]:
    """Get a list of MIDI files from a bundled corpus.

    Parameters
    ----------
    corpus_name : str
        Name of the corpus. Currently supports: 'essen'
    max_files : int, optional
        Maximum number of files to return. If None, returns all files.

    Returns
    -------
    List[Path]
        List of MIDI file paths

    Raises
    ------
    ValueError
        If the corpus name is not recognized
    FileNotFoundError
        If the corpus directory does not exist
    """
    corpus_path = get_corpus_path(corpus_name)
    
    midi_files = list(corpus_path.glob("*.mid"))
    midi_files.extend(corpus_path.glob("*.midi"))
    
    # Sort files naturally
    midi_files = natsorted(midi_files)
    
    if max_files is not None:
        midi_files = midi_files[:max_files]
    
    return midi_files


def list_available_corpora() -> List[str]:
    """List all available bundled corpora.

    Returns
    -------
    List[str]
        List of available corpus names
    """
    return ["essen"]
