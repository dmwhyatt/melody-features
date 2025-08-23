"""
This is a Python wrapper for the R package 'melsim' (https://github.com/sebsilas/melsim).
This wrapper allows the user to easily interface with the melsim package using numpy arrays
representing melodies.

Melsim is a package for computing similarity between melodies, and is being developed by
Sebastian Silas (https://sebsilas.com/) and Klaus Frieler
(https://www.aesthetics.mpg.de/en/the-institute/people/klaus-frieler.html).

Melsim is based on SIMILE, which was written by Daniel Müllensiefen and Klaus Frieler in 2003/2004.
This package is used to compare two or more melodies pairwise across a range of similarity measures.
Not all similarity measures are implemented in melsim, but the ones that are can be used here.

All of the following similarity measures are implemented and functional in melsim:
Please be aware that the names of the similarity measures are case-sensitive.

Num:        Name:
1           Jaccard
2       Kulczynski2
3            Russel
4             Faith
5          Tanimoto
6              Dice
7            Mozley
8            Ochiai
9            Simpson
10           cosine
11          angular
12      correlation
13        Tschuprow
14           Cramer
15            Gower
16        Euclidean
17        Manhattan
18         supremum
19         Canberra
20            Chord
21         Geodesic
22             Bray
23          Soergel
24           Podani
25        Whittaker
26         eJaccard
27            eDice
28   Bhjattacharyya
29       divergence
30        Hellinger
31    edit_sim_utf8
32         edit_sim
33      Levenshtein
34          sim_NCD
35            const
36          sim_dtw

The following similarity measures are not currently functional in melsim:
1    count_distinct (set-based)
2          tversky (set-based)
3   simple matching
4   braun_blanquet (set-based)
5        minkowski (vector-based)
6           ukkon (distribution-based)
7      sum_common (distribution-based)
8       distr_sim (distribution-based)
9   stringdot_utf8 (sequence-based)
10            pmi (special)
11       sim_emd (special)

Further to the similarity measures, melsim allows the user to specify which domain the
similarity should be calculated for. This is referred to as a "transformation" in melsim,
and all of the following transformations are implemented and functional:

Num:        Name:
1           pitch
2           int
3           fuzzy_int
4           parsons
5           pc
6           ioi_class
7           duration_class
8           int_X_ioi_class
9           implicit_harmonies

The following transformations are not currently functional in melsim:

Num:        Name:
1           ioi
2           phrase_segmentation

"""

import json
import logging
import os
import subprocess
from functools import cache, wraps
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential
from tqdm import tqdm

from melody_features.import_mid import import_midi

r_base_packages = ["base", "utils"]
r_cran_packages = [
    "tibble",
    "R6",
    "remotes",
    "dplyr",
    "magrittr",
    "proxy",
    "purrr",
    "purrrlyr",
    "tidyr",
    "yaml",
    "stringr",
    "emdist",
    "dtw",
    "ggplot2",
    "cba",
]
r_github_packages = ["melsim"]
github_repos = {
    "melsim": "sebsilas/melsim",
}


def check_r_packages_installed(install_missing: bool = False, n_retries: int = 3):
    """Check if required R packages are installed."""
    # Create R script to check package installation
    check_script = """
    packages <- c({packages})
    missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
    if (length(missing) > 0) {{
        cat(jsonlite::toJSON(missing))
    }}
    """

    # Format package list
    packages_str = ", ".join([f'"{p}"' for p in r_cran_packages + r_github_packages])
    check_script = check_script.format(packages=packages_str)

    # Run R script
    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script], capture_output=True, text=True, check=True
        )
        missing_packages = json.loads(result.stdout.strip())

        if missing_packages:
            if install_missing:
                for package in missing_packages:
                    try:
                        for attempt in Retrying(
                            stop=stop_after_attempt(n_retries),
                            wait=wait_exponential(multiplier=1, min=1, max=10),
                        ):
                            with attempt:
                                install_r_package(package)
                    except RetryError as e:
                        raise RuntimeError(
                            f"Failed to install R package '{package}' after {n_retries} attempts. "
                            "See above for the traceback."
                        ) from e
            else:
                raise ImportError(
                    f"Packages {missing_packages} are required but not installed. "
                    "You can install them by running: install_dependencies()"
                )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error checking R packages: {e.stderr}")


def install_r_package(package: str):
    """Install an R package."""
    logger = logging.getLogger("melody_features")
    if package in r_cran_packages:
        logger.info(f"Installing CRAN package '{package}'...")
        install_script = f"""
        utils::chooseCRANmirror(ind=1)
        utils::install.packages("{package}", dependencies=TRUE)
        """
        subprocess.run(["Rscript", "-e", install_script], check=True)
    elif package in r_github_packages:
        logger.info(f"Installing GitHub package '{package}'...")
        repo = github_repos[package]
        install_script = f"""
        if (!requireNamespace("remotes", quietly = TRUE)) {{
            utils::install.packages("remotes")
        }}
        remotes::install_github("{repo}", upgrade="always", dependencies=TRUE)
        """
        subprocess.run(["Rscript", "-e", install_script], check=True)
    else:
        raise ValueError(f"Unknown package type for '{package}'")


def install_dependencies():
    """Install all required R packages."""
    logger = logging.getLogger("melody_features")
    # Check which packages need to be installed
    check_script = """
    packages <- c({packages})
    missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
    cat(jsonlite::toJSON(missing))  # Always return a JSON array, even if empty
    """

    # Check CRAN packages
    packages_str = ", ".join([f'"{p}"' for p in r_cran_packages])
    check_script_cran = check_script.format(packages=packages_str)

    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script_cran],
            capture_output=True,
            text=True,
            check=True,
        )
        missing_cran = json.loads(result.stdout.strip())

        if missing_cran:
            logger.info("Installing missing CRAN packages...")
            cran_script = f"""
            utils::chooseCRANmirror(ind=1)
            utils::install.packages(c({", ".join([f'"{p}"' for p in missing_cran])}), dependencies=TRUE)
            """
            subprocess.run(["Rscript", "-e", cran_script], check=True)
        else:
            logger.info("Skipping install: All CRAN packages are already installed.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error checking CRAN packages: {e.stderr}")

    # Check GitHub packages
    packages_str = ", ".join([f'"{p}"' for p in r_github_packages])
    check_script_github = check_script.format(packages=packages_str)

    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script_github],
            capture_output=True,
            text=True,
            check=True,
        )
        missing_github = json.loads(result.stdout.strip())

        if missing_github:
            logger.info("Installing missing GitHub packages...")
            for package in missing_github:
                repo = github_repos[package]
                logger.info(f"Installing {package} from {repo}...")
                install_script = f"""
                if (!requireNamespace("remotes", quietly = TRUE)) {{
                    utils::install.packages("remotes")
                }}
                remotes::install_github("{repo}", upgrade="always", dependencies=TRUE)
                """
                subprocess.run(["Rscript", "-e", install_script], check=True)
        else:
            logger.info("Skipping install: All GitHub packages are already installed.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error checking GitHub packages: {e.stderr}")

    logger.info("All dependencies are installed and up to date.")


def check_python_package_installed(package: str):
    """Check if a Python package is installed."""
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Package '{package}' is required but not installed. "
            f"Please install it using pip: pip install {package}"
        )


def get_similarity(
    melody1_pitches: np.ndarray,
    melody1_starts: np.ndarray,
    melody1_ends: np.ndarray,
    melody2_pitches: np.ndarray,
    melody2_starts: np.ndarray,
    melody2_ends: np.ndarray,
    method: str,
    transformation: str,
) -> float:
    """Calculate similarity between two melodies using the specified method."""
    # Convert arrays to comma-separated strings
    pitches1_str = ",".join(map(str, melody1_pitches))
    starts1_str = ",".join(map(str, melody1_starts))
    ends1_str = ",".join(map(str, melody1_ends))
    pitches2_str = ",".join(map(str, melody2_pitches))
    starts2_str = ",".join(map(str, melody2_starts))
    ends2_str = ",".join(map(str, melody2_ends))

    # Create R script for similarity calculation
    r_script = f"""
    library(melsim)
    
    # Create melody objects
    melody1 <- melody_factory$new(mel_data = tibble::tibble(
        onset = c({starts1_str}),
        pitch = c({pitches1_str}),
        duration = c({ends1_str}) - c({starts1_str})
    ))
    
    melody2 <- melody_factory$new(mel_data = tibble::tibble(
        onset = c({starts2_str}),
        pitch = c({pitches2_str}),
        duration = c({ends2_str}) - c({starts2_str})
    ))
    
    # Create similarity measure
    sim_measure <- sim_measure_factory$new(
        name = "{method}",
        full_name = "{method}",
        transformation = "{transformation}",
        parameters = list(),
        sim_measure = "{method}"
    )
    
    # Calculate similarity
    result <- melody1$similarity(melody2, sim_measure)
    cat(jsonlite::toJSON(result$sim))
    """

    # Run R script
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script], capture_output=True, text=True, check=True
        )
        return float(json.loads(result.stdout.strip()))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating similarity: {e.stderr}")


def _convert_strings_to_tuples(d: Dict) -> Dict:
    """Convert string keys back to tuples where needed."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_strings_to_tuples(v)
        else:
            result[k] = v
    return result


def load_midi_file(
    file_path: Union[str, Path]
) -> Tuple[List[int], List[float], List[float]]:
    """Load MIDI file and extract melody attributes.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to MIDI file

    Returns
    -------
    Tuple[List[int], List[float], List[float]]
        Tuple of (pitches, start_times, end_times)
    """
    midi_data = import_midi(str(file_path))

    if midi_data is None:
        raise ValueError(f"Could not import MIDI file: {file_path}")

    return midi_data["pitches"], midi_data["starts"], midi_data["ends"]


def _compute_similarity(args: Tuple) -> float:
    """Compute similarity between two melodies using R script.

    Parameters
    ----------
    args : Tuple
        Tuple containing (melody1_data, melody2_data, method, transformation)
        where melody_data is a tuple of (pitches, starts, ends)

    Returns
    -------
    float
        Similarity value
    """
    melody1_data, melody2_data, method, transformation = args

    # Convert lists to comma-separated strings
    pitches1_str = ",".join(map(str, melody1_data[0]))
    starts1_str = ",".join(map(str, melody1_data[1]))
    ends1_str = ",".join(map(str, melody1_data[2]))
    pitches2_str = ",".join(map(str, melody2_data[0]))
    starts2_str = ",".join(map(str, melody2_data[1]))
    ends2_str = ",".join(map(str, melody2_data[2]))

    # Create R script for similarity calculation
    r_script = f"""
    library(melsim)
    
    # Create melody objects
    melody1 <- melody_factory$new(mel_data = tibble::tibble(
        onset = c({starts1_str}),
        pitch = c({pitches1_str}),
        duration = c({ends1_str}) - c({starts1_str})
    ))
    
    melody2 <- melody_factory$new(mel_data = tibble::tibble(
        onset = c({starts2_str}),
        pitch = c({pitches2_str}),
        duration = c({ends2_str}) - c({starts2_str})
    ))
    
    # Create similarity measure
    sim_measure <- sim_measure_factory$new(
        name = "{method}",
        full_name = "{method}",
        transformation = "{transformation}",
        parameters = list(),
        sim_measure = "{method}"
    )
    
    # Calculate similarity
    result <- melody1$similarity(melody2, sim_measure)
    cat(jsonlite::toJSON(result$sim))
    """

    # Run R script
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script], capture_output=True, text=True, check=True
        )
        output = json.loads(result.stdout.strip())
        # Handle both single values and lists
        if isinstance(output, list):
            return output[0]  # Return first value if it's a list
        return float(output)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating similarity: {e.stderr}")


def _batch_compute_similarities(args_list: List[Tuple]) -> List[float]:
    """Compute similarities for a batch of melody pairs.

    Parameters
    ----------
    args_list : List[Tuple]
        List of argument tuples for _compute_similarity

    Returns
    -------
    List[float]
        List of similarity values
    """
    # Create R script for batch similarity calculation with improved efficiency
    r_script = """
    library(melsim)
    library(jsonlite)
    library(purrr)
    
    # Function to create melody object
    create_melody <- function(pitches, starts, ends) {
        melody_factory$new(mel_data = tibble::tibble(
            onset = as.numeric(strsplit(starts, ",")[[1]]),
            pitch = as.numeric(strsplit(pitches, ",")[[1]]),
            duration = as.numeric(strsplit(ends, ",")[[1]]) - as.numeric(strsplit(starts, ",")[[1]])
        ))
    }
    
    # Function to calculate similarity
    calc_similarity <- function(melody1, melody2, method, transformation) {
        sim_measure <- sim_measure_factory$new(
            name = method,
            full_name = method,
            transformation = transformation,
            parameters = list(),
            sim_measure = method
        )
        result <- melody1$similarity(melody2, sim_measure)
        result$sim
    }
    
    # Process command line arguments
    args <- commandArgs(trailingOnly = TRUE)
    n_args <- length(args)
    n_comparisons <- n_args / 8  # Each comparison has 8 arguments
    
    # Pre-allocate results vector
    results <- numeric(n_comparisons)
    
    # Create a cache for melody objects
    melody_cache <- new.env()
    
    # Process in chunks for better memory management
    chunk_size <- 1000
    n_chunks <- ceiling(n_comparisons / chunk_size)
    
    for (chunk in seq_len(n_chunks)) {
        start_idx <- (chunk - 1) * chunk_size + 1
        end_idx <- min(chunk * chunk_size, n_comparisons)
        
        # Process chunk
        for (i in start_idx:end_idx) {
            idx <- (i-1) * 8 + 1
            
            # Get or create melody1
            melody1_key <- paste(args[idx], args[idx+1], args[idx+2], sep="|")
            if (!exists(melody1_key, envir=melody_cache)) {
                melody_cache[[melody1_key]] <- create_melody(args[idx], args[idx+1], args[idx+2])
            }
            melody1 <- melody_cache[[melody1_key]]
            
            # Get or create melody2
            melody2_key <- paste(args[idx+3], args[idx+4], args[idx+5], sep="|")
            if (!exists(melody2_key, envir=melody_cache)) {
                melody_cache[[melody2_key]] <- create_melody(args[idx+3], args[idx+4], args[idx+5])
            }
            melody2 <- melody_cache[[melody2_key]]
            
            method <- args[idx+6]
            transformation <- args[idx+7]
            
            results[i] <- calc_similarity(melody1, melody2, method, transformation)
        }
        
        # Force garbage collection after each chunk
        gc()
    }
    
    cat(toJSON(results))
    """

    # Prepare all arguments
    all_args = []
    for melody1_data, melody2_data, method, transformation in args_list:
        # Convert lists to comma-separated strings
        pitches1_str = ",".join(map(str, melody1_data[0]))
        starts1_str = ",".join(map(str, melody1_data[1]))
        ends1_str = ",".join(map(str, melody1_data[2]))
        pitches2_str = ",".join(map(str, melody2_data[0]))
        starts2_str = ",".join(map(str, melody2_data[1]))
        ends2_str = ",".join(map(str, melody2_data[2]))

        all_args.extend(
            [
                pitches1_str,
                starts1_str,
                ends1_str,
                pitches2_str,
                starts2_str,
                ends2_str,
                method,
                transformation,
            ]
        )

    # Run R script with all arguments
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script] + all_args,
            capture_output=True,
            text=True,
            check=True,
        )
        return [float(x) for x in json.loads(result.stdout.strip())]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating similarities: {e.stderr}")


def _load_melody(file):
    """Helper function to load a melody file for parallel processing."""
    logger = logging.getLogger("melody_features")
    try:
        return file.name, load_midi_file(file)
    except Exception as e:
        logger.warning(f"Could not load {file.name}: {str(e)}")
        return None


def get_similarity_from_midi(
    midi_path1: Union[str, Path],
    midi_path2: Union[str, Path] = None,
    method: Union[str, List[str]] = "Jaccard",
    transformation: Union[str, List[str]] = "pitch",
    output_file: Union[str, Path] = None,
    n_cores: int = None,
    batch_size: int = 1000,  # Increased from 500 to 1000 for better performance
) -> Union[float, Dict[Tuple[str, str, str, str], float]]:
    """Calculate similarity between MIDI files.

    If midi_path1 is a directory, performs pairwise comparisons between all MIDI files
    in the directory, ignoring midi_path2.

    You can provide a single method and transformation, or a list of methods and transformations.
    If you provide a list of methods and transformations, the function will return a dictionary
    mapping tuples of (file1, file2, method, transformation) to their similarity values.

    Parameters
    ----------
    midi_path1 : Union[str, Path]
        Path to first MIDI file or directory containing MIDI files
    midi_path2 : Union[str, Path], optional
        Path to second MIDI file. Ignored if midi_path1 is a directory
    method : Union[str, List[str]], default="Jaccard"
        Name of the similarity method(s) to use. Can be a single method or a list of methods.
    transformation : Union[str, List[str]], default="pitch"
        Name of the transformation(s) to use. Can be a single transformation or a list of transformations.
    output_file : Union[str, Path], optional
        If provided and doing pairwise comparisons, save results to this file.
        If no extension is provided, .json will be added.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing. Defaults to all available cores.
    batch_size : int, default=1000
        Number of comparisons to process in each batch

    Returns
    -------
    Union[float, Dict[Tuple[str, str, str, str], float]]
        If comparing two files, returns similarity value.
        If comparing all files in a directory, returns dictionary mapping tuples of
        (file1, file2, method, transformation) to their similarity values
    """
    # Convert single method/transformation to lists
    methods = [method] if isinstance(method, str) else method
    transformations = (
        [transformation] if isinstance(transformation, str) else transformation
    )

    midi_path1 = Path(midi_path1)

    # If midi_path1 is a directory, do pairwise comparisons
    if midi_path1.is_dir():
        midi_files = list(midi_path1.glob("*.mid"))

        if not midi_files:
            raise ValueError(f"No MIDI files found in {midi_path1}")

        # Load all melodies in parallel with progress bar
        logger = logging.getLogger("melody_features")
        logger.info("Loading melodies...")
        n_cores = n_cores or cpu_count()

        with Pool(n_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(_load_melody, midi_files),
                    total=len(midi_files),
                    desc="Loading MIDI files",
                )
            )

        melody_data = {name: data for name, data in results if data is not None}

        if len(melody_data) < 2:
            raise ValueError("Need at least 2 valid MIDI files for comparison")

        # Prepare arguments for parallel processing
        logger.info("Computing similarities...")
        args = []
        file_pairs = []

        # Pre-compute all combinations for better performance
        combinations_list = list(combinations(melody_data.items(), 2))
        for (name1, data1), (name2, data2) in combinations_list:
            for m in methods:
                for t in transformations:
                    args.append((data1, data2, m, t))
                    file_pairs.append((name1, name2, m, t))

        # Process in larger batches for better performance
        similarities_list = []
        for i in tqdm(range(0, len(args), batch_size), desc="Processing batches"):
            batch = args[i : i + batch_size]
            similarities_list.extend(_batch_compute_similarities(batch))

        # Create dictionary of results
        similarities = dict(zip(file_pairs, similarities_list))

        # Save to file if output file specified
        if output_file:
            logger.info("Saving results...")
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "file1": f1,
                        "file2": f2,
                        "method": m,
                        "transformation": t,
                        "similarity": sim,
                    }
                    for (f1, f2, m, t), sim in similarities.items()
                ]
            )

            # Ensure output file has .json extension
            output_file = Path(output_file)
            if not output_file.suffix:
                output_file = output_file.with_suffix(".json")

            df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Results saved to {output_file}")

        return similarities

    # For single file comparison, only use first method and transformation
    if len(methods) > 1 or len(transformations) > 1:
        logger.warning(
            "Multiple methods/transformations provided for two-file pairwise comparison. Using first method and transformation."
        )

    # Load MIDI files
    melody1_pitches, melody1_starts, melody1_ends = load_midi_file(midi_path1)
    melody2_pitches, melody2_starts, melody2_ends = load_midi_file(midi_path2)

    # Calculate similarity
    return _compute_similarity(
        (
            (melody1_pitches, melody1_starts, melody1_ends),
            (melody2_pitches, melody2_starts, melody2_ends),
            methods[0],
            transformations[0],
        )
    )
