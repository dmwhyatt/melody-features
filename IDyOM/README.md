# IDyOM Standalone Analyzer

A functional Python interface to IDyOM (Information Dynamics of Music) that doesn't require Docker or Jupyter. This tool provides a streamlined way to run IDyOM experiments for music analysis and prediction.

## Features

- **No Docker required**: Runs directly on your system with local IDyOM installation
- **Functional interface**: Simple Python API for common analysis tasks
- **Command-line support**: Run analyses directly from the terminal
- **Multiple export formats**: Export results to CSV or MATLAB formats
- **Melody and harmony analysis**: Support for both melodic and harmonic viewpoints
- **Comprehensive logging**: Track analysis progress and debug issues

## Installation

### Prerequisites

- Python 3.7+
- SBCL (Steel Bank Common Lisp)
- SQLite3
- wget and unzip utilities

### Step 1: Install System Dependencies

**macOS (using Homebrew):**
```bash
brew install sbcl sqlite3
```

**Ubuntu/Debian:**
```bash
sudo apt-get install sbcl sqlite3 libsqlite3-dev wget unzip
```

**Windows:**
- Install SBCL from: http://www.sbcl.org/platform-table.html
- Install SQLite from: https://sqlite.org/download.html

### Step 2: Set up IDyOM

```bash
# Make the setup script executable
chmod +x setup_idyom.sh

# Run the setup script
./setup_idyom.sh
```

This script will:
- Install Quicklisp (Common Lisp package manager)
- Download and install IDyOM
- Set up the IDyOM database
- Configure your SBCL environment

### Step 3: Install Python Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make the analyzer script executable
chmod +x idyom_analyzer.py
```

## Usage

### Python API

```python
from idyom_analyzer import IDyOMAnalyzer

# Initialize analyzer
analyzer = IDyOMAnalyzer(
    test_dataset_path="data/small_kern_subset/",
    output_dir="./results",
    experiment_name="my_experiment"
)

# Run melody analysis
results_path = analyzer.analyze_melody(
    target_viewpoints=['cpitch'],
    source_viewpoints=['cpitch'],
    models=':both',
    k=10
)

# Export results
analyzer.export_results(results_path, format='csv')
```

### Command Line Interface

```bash
# Basic melody analysis
python idyom_analyzer.py data/small_kern_subset/ --texture melody --export-format csv

# Harmony analysis with specific viewpoints
python idyom_analyzer.py data/small_kern_subset/ \
    --texture harmony \
    --target-viewpoints pc-chord bass-pc \
    --source-viewpoints pc-chord bass-pc \
    --models :both \
    --k 5 \
    --export-format mat

# With pretraining dataset
python idyom_analyzer.py data/small_kern_subset/ \
    --pretrain-dataset midi_billboard/ \
    --output-dir ./my_results \
    --experiment-name billboard_analysis
```

### Available Parameters

#### Analysis Parameters
- `target_viewpoints`: List of viewpoints to predict (e.g., ['cpitch', 'dur'])
- `source_viewpoints`: List of viewpoints to use as context, or ':select' for automatic selection
- `models`: Model type - ':stm' (short-term), ':ltm' (long-term), or ':both'
- `k`: Number of cross-validation folds (default: 10)
- `detail`: Output detail level 1-3 (default: 3)

#### Common Viewpoints

**Melodic Viewpoints:**
- `cpitch`: Chromatic pitch
- `dur`: Duration
- `ioi`: Inter-onset interval
- `cpint`: Chromatic pitch interval
- `contour`: Melodic contour

**Harmonic Viewpoints:**
- `pc-chord`: Pitch-class chords
- `bass-pc`: Bass pitch class
- `root-pc`: Root pitch class
- `pc-set`: Pitch-class sets
- `gct-chord-type`: General chord type

For a complete list, see the README table in the original project.

## File Structure

```
idyom-standalone/
├── idyom_analyzer.py      # Main analyzer class
├── example_analysis.py    # Example usage
├── setup_idyom.sh        # IDyOM installation script
├── requirements.txt      # Python dependencies
├── py2lisp/             # Original py2lisp modules
├── data/                # Sample datasets
└── results/             # Analysis outputs
```

## Examples

### Example 1: Basic Melody Analysis

```python
from idyom_analyzer import IDyOMAnalyzer

analyzer = IDyOMAnalyzer(
    test_dataset_path="data/small_kern_subset/",
    output_dir="./results/melody",
    experiment_name="basic_melody"
)

results_path = analyzer.analyze_melody(
    target_viewpoints=['cpitch'],
    source_viewpoints=['cpitch'],
    models=':both'
)

analyzer.export_results(results_path, format='csv')
```

### Example 2: Harmonic Analysis

```python
analyzer = IDyOMAnalyzer(
    test_dataset_path="data/small_kern_subset/",
    output_dir="./results/harmony"
)

results_path = analyzer.analyze_harmony(
    target_viewpoints=['pc-chord', 'bass-pc'],
    source_viewpoints=['pc-chord', 'bass-pc'],
    models=':both',
    k=5
)

analyzer.export_results(results_path, format='mat')
```

### Example 3: Command Line Usage

```bash
# Quick melody analysis
python idyom_analyzer.py data/small_kern_subset/ --texture melody --k 5

# Detailed harmony analysis with export
python idyom_analyzer.py data/small_kern_subset/ \
    --texture harmony \
    --target-viewpoints pc-chord bass-pc \
    --models :both \
    --detail 3 \
    --export-format csv \
    --output-dir ./harmony_results
```

## Output

Results are saved in timestamped folders with the following structure:

```
results/
└── experiment_name_TIMESTAMP/
    ├── experiment_input_data_folder/
    │   └── test_dataset/           # Copied input files
    ├── experiment_output_data_folder/
    │   └── *.dat                   # Raw IDyOM output
    ├── outputs_in_csv/             # CSV exports
    ├── outputs_in_mat/             # MATLAB exports
    └── compute.lisp                # Generated LISP script
```

## Troubleshooting

### Common Issues

1. **SBCL not found**: Make sure SBCL is installed and in your PATH
2. **Database errors**: Run the setup script again to reinitialize the database
3. **Memory issues**: IDyOM can be memory-intensive. Consider using smaller datasets or fewer cross-validation folds
4. **Import errors**: Ensure all Python dependencies are installed

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual IDyOM Test

Test your IDyOM installation:

```bash
sbcl --eval "(start-idyom)" --eval "(idyom-db:describe-database)" --eval "(quit)"
```

## Performance Tips

- Use fewer cross-validation folds (k=3-5) for faster results
- Start with small datasets to test your setup
- Use pretraining datasets for better predictions
- Monitor memory usage with large datasets

## Contributing

To extend this tool:

1. Add new analysis methods to the `IDyOMAnalyzer` class
2. Extend the command-line interface in the `main()` function
3. Add new export formats in the `export_results()` method

## License

This project builds upon the original IDyOM and py2lispIDyOM projects. Please respect their respective licenses.

## References

- IDyOM Project: https://github.com/mtpearce/idyom
- py2lispIDyOM: https://github.com/xinyiguan/py2lispIDyOM
- Original Docker version: https://github.com/frshdjfry/py2lispIDyOMJupyter 