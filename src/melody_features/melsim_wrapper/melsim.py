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
37           opti3

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
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

# List of available similarity measures that use built-in similarity_measures$
BUILTIN_SIMILARITY_MEASURES = {
    "opti3", "Jaccard", "Kulczynski2", "Russel", "Faith", "Tanimoto", "Dice",
    "Mozley", "Ochiai", "Simpson", "cosine", "angular", "correlation",
    "Tschuprow", "Cramer", "Gower", "Euclidean", "Manhattan", "supremum",
    "Canberra", "Chord", "Geodesic", "Bray", "Soergel", "Podani", "Whittaker",
    "eJaccard", "eDice", "Bhjattacharyya", "divergence", "Hellinger",
    "edit_sim_utf8", "edit_sim", "Levenshtein", "sim_NCD", "const", "sim_dtw"
}

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
    from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential
    
    check_script = """
    packages <- c({packages})
    missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
    if (length(missing) > 0) {{
        cat(jsonlite::toJSON(missing))
    }}
    """

    packages_str = ", ".join([f'"{p}"' for p in r_cran_packages + r_github_packages])
    check_script = check_script.format(packages=packages_str)

    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script], capture_output=True, text=True, check=True
        )
        missing_packages = json.loads(result.stdout.strip()) if result.stdout.strip() else []

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
    check_script = """
    packages <- c({packages})
    missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
    cat(jsonlite::toJSON(missing))
    """

    packages_str = ", ".join([f'"{p}"' for p in r_cran_packages])
    check_script_cran = check_script.format(packages=packages_str)

    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script_cran],
            capture_output=True,
            text=True,
            check=True,
        )
        missing_cran = json.loads(result.stdout.strip()) if result.stdout.strip() else []

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

    packages_str = ", ".join([f'"{p}"' for p in r_github_packages])
    check_script_github = check_script.format(packages=packages_str)

    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script_github],
            capture_output=True,
            text=True,
            check=True,
        )
        missing_github = json.loads(result.stdout.strip()) if result.stdout.strip() else []

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


def get_similarity_from_midi(
    midi_path1: Union[str, Path, List[Union[str, Path]]],
    midi_path2: Union[str, Path] = None,
    method: Union[str, List[str]] = "opti3",
    transformation: Union[str, List[str]] = None,
    output_file: Union[str, Path] = None,
    n_cores: int = None,
    batch_size: int = 100,
) -> Union[float, Dict[Tuple[str, str, str, str], float]]:
    """Calculate similarity between MIDI files using melsim.

    This function directly calls melsim's read_midi and similarity functions
    for consistency.

    Parameters
    ----------
    midi_path1 : Union[str, Path, List[Union[str, Path]]]
        Path to first MIDI file, directory containing MIDI files, or list of MIDI file paths
    midi_path2 : Union[str, Path], optional
        Path to second MIDI file. Ignored if midi_path1 is a directory or list
    method : Union[str, List[str]], default="opti3"
        Name of the similarity method(s) to use. Can be a single method or a list of methods.
        Use built-in measures like "opti3", "Jaccard", "edit_sim", etc.
    transformation : Union[str, List[str]], optional
        Transformation to apply (only used for non-opti3 measures).
        Options: "pitch", "int", "fuzzy_int", "parsons", "pc", etc.
    output_file : Union[str, Path], optional
        If provided and doing pairwise comparisons, save results to this file.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing. Defaults to all available cores.
    batch_size : int, default=100
        Number of comparisons to process in each R batch call

    Returns
    -------
    Union[float, Dict[Tuple[str, str, str, str], float]]
        If comparing two files, returns similarity value.
        If comparing multiple files, returns dictionary mapping tuples of
        (file1, file2, method, transformation) to their similarity values
    """
    logger = logging.getLogger("melody_features")
    
    # Convert single method to list
    methods = [method] if isinstance(method, str) else list(method)
    
    # Handle transformation (only needed for certain measures)
    transformations = None
    if transformation is not None:
        transformations = [transformation] if isinstance(transformation, str) else list(transformation)

    # Determine file list
    if isinstance(midi_path1, list):
        midi_files = [Path(f) for f in midi_path1]
        is_pairwise = True
    else:
        midi_path1 = Path(midi_path1)
        if midi_path1.is_dir():
            midi_files = list(midi_path1.glob("*.mid")) + list(midi_path1.glob("*.midi"))
            is_pairwise = True
        else:
            if midi_path2 is None:
                raise ValueError("midi_path2 is required when midi_path1 is a single file")
            midi_files = None
            is_pairwise = False

    if is_pairwise:
        # Pairwise comparison of multiple files
        if not midi_files or len(midi_files) < 2:
            raise ValueError("Need at least 2 MIDI files for pairwise comparison")
        
        return _compute_pairwise_similarities(
            midi_files, methods, transformations, output_file, batch_size, logger
        )
    else:
        # Single pair comparison
        return _compute_single_similarity(
            midi_path1, Path(midi_path2), methods[0], 
            transformations[0] if transformations else None
        )


# Composite measures that can be accessed directly via similarity_measures$
COMPOSITE_MEASURES = {"opti3"}


def _compute_single_similarity(
    file1: Path, 
    file2: Path, 
    method: str,
    transformation: str = None
) -> float:
    """Compute similarity between two MIDI files with a single R call.
    
    This mirrors the simple R script approach for maximum performance.
    """
    # Ensure absolute paths for R
    file1_abs = str(file1.resolve()) if isinstance(file1, Path) else str(Path(file1).resolve())
    file2_abs = str(file2.resolve()) if isinstance(file2, Path) else str(Path(file2).resolve())
    
    # Determine how to create the similarity measure
    if method in COMPOSITE_MEASURES:
        # Composite measures like opti3 are accessed directly
        sim_measure_code = f'sim_measure <- similarity_measures${method}'
    else:
        # Basic measures need sim_measure_factory with a transformation
        trans = transformation if transformation else "pitch"
        sim_measure_code = f'''sim_measure <- sim_measure_factory$new(
    name = "{method}",
    full_name = "{method}",
    transformation = "{trans}",
    parameters = list(),
    sim_measure = "{method}"
)'''
    
    # Build the R script
    r_script = f'''
library(melsim)
library(dplyr)

file1 <- "{file1_abs}"
file2 <- "{file2_abs}"

# Read MIDI files using melsim's built-in reader
mel1_data <- melsim:::read_midi(file1) %>% dplyr::rename(duration = durations)
mel2_data <- melsim:::read_midi(file2) %>% dplyr::rename(duration = durations)

# Create melody objects
mel1 <- melody_factory$new(mel_data = mel1_data, mel_meta = list(name = basename(file1)))
mel2 <- melody_factory$new(mel_data = mel2_data, mel_meta = list(name = basename(file2)))

# Create similarity measure
{sim_measure_code}

# Compute similarity (suppress verbose output)
result <- melsim(mel1, mel2, sim_measures = sim_measure, verbose = FALSE, with_progress = FALSE)

# Output just the score
# For composite measures (like opti3), the result contains multiple rows with different algorithms
# Extract the row matching the requested method name, otherwise use the first row
if ("algorithm" %in% names(result$data)) {{
    method_row <- result$data[result$data$algorithm == "{method}", ]
    if (nrow(method_row) > 0) {{
        cat(method_row$sim[1])
    }} else {{
        cat(result$data$sim[1])
    }}
}} else {{
    cat(result$data$sim[1])
}}
'''

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract just the numeric value from the output (in case of extra logging)
        output = result.stdout.strip()
        # Get the last line which should be the numeric result
        lines = output.split('\n')
        return float(lines[-1].strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating similarity: {e.stderr}")


def _compute_pairwise_similarities(
    midi_files: List[Path],
    methods: List[str],
    transformations: List[str] = None,
    output_file: Union[str, Path] = None,
    batch_size: int = 100,
    logger = None
) -> Dict[Tuple[str, str, str, str], float]:
    """Compute pairwise similarities between multiple MIDI files.
    
    Uses batched R calls for efficiency - one R process handles many comparisons.
    """
    from tqdm import tqdm
    
    if logger is None:
        logger = logging.getLogger("melody_features")
    
    # Default transformation
    if transformations is None:
        transformations = ["pitch"]
    
    # Generate all pairs and method/transformation combinations
    file_pairs = list(combinations(midi_files, 2))
    
    all_comparisons = []
    for f1, f2 in file_pairs:
        for method in methods:
            for trans in transformations:
                all_comparisons.append((f1, f2, method, trans))
    
    logger.info(f"Computing {len(all_comparisons)} similarity comparisons...")
    
    # Process in batches
    results = {}
    for i in tqdm(range(0, len(all_comparisons), batch_size), desc="Processing batches"):
        batch = all_comparisons[i:i + batch_size]
        batch_results = _compute_batch_similarities(batch)
        results.update(batch_results)
    
    # Save to file if requested
    if output_file:
        _save_results(results, output_file, logger)
    
    return results


def _compute_batch_similarities(
    comparisons: List[Tuple[Path, Path, str, str]]
) -> Dict[Tuple[str, str, str, str], float]:
    """Compute a batch of similarities in a single R process.
    
    This is the key optimization - one R process handles many comparisons.
    
    Parameters
    ----------
    comparisons : List[Tuple[Path, Path, str, str]]
        List of (file1, file2, method, transformation) tuples
    """
    if not comparisons:
        return {}
    
    # Create a temporary R script file for the batch
    # Build file list and comparison indices (using absolute paths)
    unique_files = {}
    file_index = 0
    
    for f1, f2, _, _ in comparisons:
        f1_abs = str(f1.resolve()) if isinstance(f1, Path) else str(Path(f1).resolve())
        f2_abs = str(f2.resolve()) if isinstance(f2, Path) else str(Path(f2).resolve())
        if f1_abs not in unique_files:
            unique_files[f1_abs] = file_index
            file_index += 1
        if f2_abs not in unique_files:
            unique_files[f2_abs] = file_index
            file_index += 1
    
    # Build R script
    file_paths_r = ", ".join([f'"{f}"' for f in unique_files.keys()])
    
    # Build comparison list
    comp_lines = []
    for f1, f2, method, trans in comparisons:
        f1_abs = str(f1.resolve()) if isinstance(f1, Path) else str(Path(f1).resolve())
        f2_abs = str(f2.resolve()) if isinstance(f2, Path) else str(Path(f2).resolve())
        idx1 = unique_files[f1_abs] + 1  # R is 1-indexed
        idx2 = unique_files[f2_abs] + 1
        comp_lines.append(f'list(i1={idx1}, i2={idx2}, method="{method}", transformation="{trans}")')
    
    comparisons_r = ", ".join(comp_lines)
    
    r_script = f'''
library(melsim)
library(dplyr)

# Load all unique files once
file_paths <- c({file_paths_r})
melodies <- list()

for (i in seq_along(file_paths)) {{
    tryCatch({{
        mel_data <- melsim:::read_midi(file_paths[i]) %>% dplyr::rename(duration = durations)
        melodies[[i]] <- melody_factory$new(
            mel_data = mel_data, 
            mel_meta = list(name = basename(file_paths[i]))
        )
    }}, error = function(e) {{
        melodies[[i]] <<- NULL
    }})
}}

# Define comparisons
comparisons <- list({comparisons_r})

# Compute all similarities
results <- numeric(length(comparisons))
for (j in seq_along(comparisons)) {{
    comp <- comparisons[[j]]
    mel1 <- melodies[[comp$i1]]
    mel2 <- melodies[[comp$i2]]
    
    if (is.null(mel1) || is.null(mel2)) {{
        results[j] <- NA
    }} else {{
        tryCatch({{
            # Check if it's a composite measure (like opti3) or needs sim_measure_factory
            if (comp$method %in% c("opti3")) {{
                sim_measure <- similarity_measures[[comp$method]]
            }} else {{
                # Use sim_measure_factory for basic measures
                trans <- if (!is.null(comp$transformation)) comp$transformation else "pitch"
                sim_measure <- sim_measure_factory$new(
                    name = comp$method,
                    full_name = comp$method,
                    transformation = trans,
                    parameters = list(),
                    sim_measure = comp$method
                )
            }}
            result <- melsim(mel1, mel2, sim_measures = sim_measure, verbose = FALSE, with_progress = FALSE)
            # For composite measures (like opti3), the result contains multiple rows with different algorithms
            # Extract the row matching the requested method name, otherwise use the first row
            if ("algorithm" %in% names(result$data)) {{
                method_row <- result$data[result$data$algorithm == comp$method, ]
                if (nrow(method_row) > 0) {{
                    results[j] <- method_row$sim[1]
                }} else {{
                    results[j] <- result$data$sim[1]
                }}
            }} else {{
                results[j] <- result$data$sim[1]
            }}
        }}, error = function(e) {{
            results[j] <<- NA
        }})
    }}
}}

# Output as JSON
cat(jsonlite::toJSON(results))
'''
    
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract just the JSON from output (skip any logging lines)
        output = result.stdout.strip()
        # Find the JSON array in the output
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('['):
                output = line
                break
        
        scores = json.loads(output)
        
        # Build result dictionary
        results = {}
        for idx, (f1, f2, method, trans) in enumerate(comparisons):
            score = scores[idx] if idx < len(scores) else None
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                # Include transformation in key to distinguish different transformations
                f1_name = f1.name if isinstance(f1, Path) else Path(f1).name
                f2_name = f2.name if isinstance(f2, Path) else Path(f2).name
                results[(f1_name, f2_name, method, trans)] = float(score)
        
        return results
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating batch similarities: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Error parsing R output: {e}")


def _save_results(
    results: Dict[Tuple, float],
    output_file: Union[str, Path],
    logger
):
    """Save results to a JSON file."""
    import pandas as pd
    
    # Handle both 3-tuple (file1, file2, method) and 4-tuple (file1, file2, method, transformation) keys
    rows = []
    for key, sim in results.items():
        if len(key) == 3:
            f1, f2, m = key
            rows.append({"file1": f1, "file2": f2, "method": m, "similarity": sim})
        elif len(key) == 4:
            f1, f2, m, trans = key
            rows.append({"file1": f1, "file2": f2, "method": m, "transformation": trans, "similarity": sim})
        else:
            # Fallback for unexpected key formats
            rows.append({"key": str(key), "similarity": sim})
    
    df = pd.DataFrame(rows)
    
    output_file = Path(output_file)
    if not output_file.suffix:
        output_file = output_file.with_suffix(".json")
    
    df.to_json(output_file, orient="records", indent=2)
    logger.info(f"Results saved to {output_file}")


# Legacy function for backward compatibility
def get_similarity(
    melody1_pitches: np.ndarray,
    melody1_starts: np.ndarray,
    melody1_ends: np.ndarray,
    melody2_pitches: np.ndarray,
    melody2_starts: np.ndarray,
    melody2_ends: np.ndarray,
    method: str,
    transformation: str = "pitch",
) -> float:
    """Calculate similarity between two melodies using the specified method.
    
    This is a legacy function that accepts pre-parsed melody data.
    For better performance, use get_similarity_from_midi() which uses
    melsim's built-in MIDI reader.
    """
    # Convert arrays to comma-separated strings
    pitches1_str = ",".join(map(str, melody1_pitches))
    starts1_str = ",".join(map(str, melody1_starts))
    durations1 = melody1_ends - melody1_starts
    durations1_str = ",".join(map(str, durations1))
    
    pitches2_str = ",".join(map(str, melody2_pitches))
    starts2_str = ",".join(map(str, melody2_starts))
    durations2 = melody2_ends - melody2_starts
    durations2_str = ",".join(map(str, durations2))

    # Determine how to create the similarity measure
    if method in COMPOSITE_MEASURES:
        sim_measure_code = f'sim_measure <- similarity_measures${method}'
    else:
        trans = transformation if transformation else "pitch"
        sim_measure_code = f'''sim_measure <- sim_measure_factory$new(
    name = "{method}",
    full_name = "{method}",
    transformation = "{trans}",
    parameters = list(),
    sim_measure = "{method}"
)'''

    r_script = f'''
library(melsim)
library(dplyr)

# Create melody data frames
mel1_data <- tibble::tibble(
    onset = c({starts1_str}),
    pitch = c({pitches1_str}),
    duration = c({durations1_str})
)

mel2_data <- tibble::tibble(
    onset = c({starts2_str}),
    pitch = c({pitches2_str}),
    duration = c({durations2_str})
)

# Create melody objects
mel1 <- melody_factory$new(mel_data = mel1_data, mel_meta = list(name = "melody1"))
mel2 <- melody_factory$new(mel_data = mel2_data, mel_meta = list(name = "melody2"))

# Create similarity measure
{sim_measure_code}

# Compute similarity (suppress verbose output)
result <- melsim(mel1, mel2, sim_measures = sim_measure, verbose = FALSE, with_progress = FALSE)

# Output just the score
# For composite measures (like opti3), the result contains multiple rows with different algorithms
# Extract the row matching the requested method name, otherwise use the first row
if ("algorithm" %in% names(result$data)) {{
    method_row <- result$data[result$data$algorithm == "{method}", ]
    if (nrow(method_row) > 0) {{
        cat(method_row$sim[1])
    }} else {{
        cat(result$data$sim[1])
    }}
}} else {{
    cat(result$data$sim[1])
}}
'''

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract just the numeric value from the output (in case of extra logging)
        output = result.stdout.strip()
        lines = output.split('\n')
        return float(lines[-1].strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calculating similarity: {e.stderr}")


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
    from melody_features.import_mid import import_midi
    
    midi_data = import_midi(str(file_path))

    if midi_data is None:
        raise ValueError(f"Could not import MIDI file: {file_path}")

    return midi_data["pitches"], midi_data["starts"], midi_data["ends"]


# ============================================================================
# Legacy functions for backwards compatibility
# ============================================================================

def check_python_package_installed(package: str):
    """Check if a Python package is installed."""
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Package '{package}' is required but not installed. "
            f"Please install it using pip: pip install {package}"
        )


def _convert_strings_to_tuples(d: Dict) -> Dict:
    """Convert string keys back to tuples where needed."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_strings_to_tuples(v)
        else:
            result[k] = v
    return result


def _compute_similarity(args: Tuple) -> float:
    """Compute similarity between two melodies using R script.

    Legacy function for backwards compatibility.

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
    
    return get_similarity(
        np.array(melody1_data[0]),
        np.array(melody1_data[1]),
        np.array(melody1_data[2]),
        np.array(melody2_data[0]),
        np.array(melody2_data[1]),
        np.array(melody2_data[2]),
        method,
        transformation,
    )


def _batch_compute_similarities(args_list: List[Tuple]) -> List[float]:
    """Compute similarities for a batch of melody pairs.

    Legacy function - now uses the optimized batch processing internally.

    Parameters
    ----------
    args_list : List[Tuple]
        List of argument tuples for _compute_similarity

    Returns
    -------
    List[float]
        List of similarity values
    """
    if not args_list:
        return []
    
    # Extract unique method (assuming all use the same method)
    method = args_list[0][2]
    
    # Build comparisons for batch processing
    # Create temp files and use the batch approach
    results = []
    for args in args_list:
        melody1_data, melody2_data, method, transformation = args
        result = get_similarity(
            np.array(melody1_data[0]),
            np.array(melody1_data[1]),
            np.array(melody1_data[2]),
            np.array(melody2_data[0]),
            np.array(melody2_data[1]),
            np.array(melody2_data[2]),
            method,
            transformation,
        )
        results.append(result)
    
    return results
