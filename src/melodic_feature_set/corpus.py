"""
Module for computing corpus-based features from melodic n-grams, similar to FANTASTIC's
implementation. This module handles the corpus analysis and saves statistics to JSON.
The actual feature calculations are handled in features.py.
"""
from collections import Counter
import json
from typing import List, Dict, Tuple
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from melodic_feature_set.melody_tokenizer import FantasticTokenizer
from melodic_feature_set.representations import Melody, read_midijson
from melodic_feature_set.import_mid import import_midi


def _convert_strings_to_tuples(key: str) -> Tuple:
    """Convert a string-encoded tuple key back to a tuple.
    
    Parameters
    ----------
    key : str
        String-encoded tuple key
        
    Returns
    -------
    Tuple
        Tuple converted from string
    """
    try:
        # Remove parentheses and split on comma
        key_str = key.strip('()').split(',')
        # Convert "None" strings back to None and strip quotes/spaces
        tuple_key = tuple(None if x.strip().strip("'\"") == "None" else x.strip().strip("'\"") for x in key_str)
        return tuple_key
    except (AttributeError, ValueError):
        # If not a tuple string, return the original key
        return key

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

    # Convert string keys back to tuples where needed
    stats['document_frequencies'] = {
        _convert_strings_to_tuples(k): v for k, v in stats['document_frequencies'].items()
    }

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
    try:
        melody_data = import_midi(midi_path)
        if melody_data is None:
            return None
        return Melody(melody_data, tempo=100)
    except Exception as e:
        print(f"Warning: Error creating Melody object from {midi_path}: {str(e)}")
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
        print(f"Found {num_melodies} melodies in {json_files[0]}")
        
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
            
        print(f"Found {len(midi_files)} MIDI files")
        
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
    # Load melodies from MIDI files
    melodies = load_melodies_from_directory(midi_dir, file_type="midi")
    # Filter out None values
    melodies = [m for m in melodies if m is not None]
    if not melodies:
        print("Error: No valid melodies found")
        exit(1)
    print(f"Processing {len(melodies)} valid melodies")

    # Compute corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies)

    # Save to JSON
    save_corpus_stats(corpus_stats, output_file)

    # Load and verify
    loaded_stats = load_corpus_stats(output_file)
    print("Corpus statistics saved and loaded successfully.")
    print(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    print(f"N-gram lengths: {loaded_stats['n_range']}")


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
    # Load melody data from JSON
    print(f"Loading melodies from JSON file: {json_file}")
    melody_data = read_midijson(json_file)
    
    if not melody_data:
        print("Error: No melody data found in JSON file")
        exit(1)
    
    print(f"Found {len(melody_data)} melodies in JSON file")
    
    # Convert to Melody objects
    melodies = []
    for i, data in enumerate(tqdm(melody_data, desc="Converting to Melody objects")):
        try:
            melody = Melody(data, tempo=100)
            melodies.append(melody)
        except Exception as e:
            print(f"Warning: Error creating Melody object from entry {i}: {str(e)}")
            continue
    
    # Filter out None values
    melodies = [m for m in melodies if m is not None]
    if not melodies:
        print("Error: No valid melodies found")
        exit(1)
    print(f"Processing {len(melodies)} valid melodies")

    # Compute corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies, n_range)

    # Save to JSON
    save_corpus_stats(corpus_stats, output_file)

    # Load and verify
    loaded_stats = load_corpus_stats(output_file)
    print("Corpus statistics saved and loaded successfully.")
    print(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    print(f"N-gram lengths: {loaded_stats['n_range']}")
