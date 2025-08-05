# Suppress warnings from external libraries BEFORE any imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

"""
Module for computing corpus-based features from melodic n-grams, similar to FANTASTIC's
implementation. This module handles the corpus analysis and saves statistics to JSON.
The actual feature calculations are handled in features.py.
"""
from collections import Counter
import json
import logging
from typing import List, Dict, Tuple
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from melodic_feature_set.melody_tokenizer import FantasticTokenizer
from melodic_feature_set.representations import Melody, read_midijson
from melodic_feature_set.import_mid import import_midi


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
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
    
    melody, n_range = args
    tokenizer = FantasticTokenizer()

    # Segment the melody first
    segments = tokenizer.segment_melody(melody, phrase_gap=1.5, units="quarters")

    # Get tokens for each segment
    all_tokens = []
    for segment in segments:
        segment_tokens = tokenizer.tokenize_melody(
            segment.pitches, 
            segment.starts, 
            segment.ends
        )
        all_tokens.extend(segment_tokens)

    unique_ngrams = set()
    for n in range(n_range[0], n_range[1] + 1):
        # Count n-grams in the combined tokens
        for i in range(len(all_tokens) - n + 1):
            ngram = tuple(all_tokens[i:i + n])
            unique_ngrams.add(ngram)

    return unique_ngrams

def compute_corpus_ngrams(melodies: List[Melody], n_range: Tuple[int, int] = (1, 6)) -> Dict:
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
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
    
    # Prepare arguments for multiprocessing
    args = [(melody, n_range) for melody in melodies]

    # Use all available CPU cores
    num_cores = mp.cpu_count()

    # Create a pool of workers
    with mp.Pool(num_cores) as pool:
        # Process melodies in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_melody_ngrams, args),
            total=len(melodies),
            desc=f"Computing n-grams using {num_cores} cores"
        ))

    # Count document frequency (number of melodies containing each n-gram)
    doc_freq = Counter()
    for ngrams in results:
        doc_freq.update(ngrams)

    # Format results for JSON serialization
    frequencies = {'document_frequencies': {}}
    for k, v in doc_freq.items():
        frequencies['document_frequencies'][str(k)] = {'count': v}

    return {
        'document_frequencies': frequencies['document_frequencies'],
        'corpus_size': len(melodies),
        'n_range': n_range
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
    if not filename.endswith('.json'):
        filename = filename + '.json'
    
    # Ensure the directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
    with open(filename, 'w', encoding='utf-8') as f:
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
    if not filename.endswith('.json'):
        filename = filename + '.json'
        
    with open(filename, encoding='utf-8') as f:
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
        raise IndexError(f"Index {idx} is out of range for file with {len(melody_data)} melodies")
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
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
    
    logger = logging.getLogger('melodic_feature_set')
    try:
        melody_data = import_midi(midi_path)
        if melody_data is None:
            return None
        return Melody(melody_data, tempo=100)
    except Exception as e:
        logger.warning(f"Error creating Melody object from {midi_path}: {str(e)}")
        return None

def load_melodies_from_directory(directory: str, file_type: str = "json") -> List[Melody]:
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
    logger = logging.getLogger('melodic_feature_set')
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    if file_type == "json":
        # For JSON, we expect a single file containing multiple melodies
        json_files = list(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {directory}. Please specify a single file.")
            
        melody_data = read_midijson(str(json_files[0]))
        num_melodies = len(melody_data)
        logger.info(f"Found {num_melodies} melodies in {json_files[0]}")
        
        # Create arguments for parallel loading
        num_cores = mp.cpu_count()
        melody_indices = [(i, str(json_files[0])) for i in range(num_melodies)]
        
        # Load melodies in parallel
        with mp.Pool(num_cores) as pool:
            melodies = list(tqdm(
                pool.starmap(load_melody, melody_indices),
                total=len(melody_indices),
                desc=f"Loading melodies using {num_cores} cores"
            ))
            
    elif file_type == "midi":
        # For MIDI, we expect multiple files, each containing one melody
        midi_files = list(directory.glob("*.mid")) + list(directory.glob("*.midi"))
        if not midi_files:
            raise FileNotFoundError(f"No MIDI files found in {directory}")
            
        logger.info(f"Found {len(midi_files)} MIDI files")
        
        # Load melodies in parallel
        num_cores = mp.cpu_count()
        with mp.Pool(num_cores) as pool:
            melodies = list(tqdm(
                pool.imap(load_midi_melody, [str(f) for f in midi_files]),
                total=len(midi_files),
                desc=f"Loading MIDI files using {num_cores} cores"
            ))
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
    logger = logging.getLogger('melodic_feature_set')
    # Load melodies from MIDI files
    melodies = load_melodies_from_directory(midi_dir, file_type="midi")
    # Filter out None values
    melodies = [m for m in melodies if m is not None]
    if not melodies:
        raise ValueError("No valid melodies could be processed from the directory. Check if the files are valid MIDI files.")
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


def make_corpus_stats_from_json(json_file: str, output_file: str, n_range: Tuple[int, int] = (1, 6)) -> None:
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
    logger = logging.getLogger('melodic_feature_set')
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
