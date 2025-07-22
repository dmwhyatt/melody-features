# I am aware of some pathing issues in py2lispIDyOM - this interface script
# seeks to provide a slightly 'hacky' fix. As such, this script ought to 
# allow the user to provide a dir of midi files and run IDyOM on them.
"""
This script provides a simplified interface to run IDyOM, using py2lispIDyOM.
It draws inspiration from both the original repo, and the Jupyter notebook version:
https://github.com/xinyiguan/py2lispIDyOM
https://github.com/frshdjfry/py2lispIDyOMJupyter

The top level function is run_idyom, which takes a directory of MIDI files and runs IDyOM on them.
By default, it will assume that pretraining is required, and will use the same directory for both the test and pretraining datasets.

The script will also check if IDyOM is installed, and if not, will enable an easy install. The `install_idyom.sh` script facilitates this,
and will be called by the script if the user chooses to install IDyOM. You should probably not have to run this yourself, but if you
wanted to do so, first make it executable:
chmod +x install_idyom.sh

Then, run it:
./install_idyom.sh
"""
import os
import subprocess
from pathlib import Path
from natsort import natsorted
from glob import glob
from typing import Optional

def is_idyom_installed():
    """Check if IDyOM is installed by verifying SBCL, the IDyOM database, and source directory."""
    # Check SBCL
    sbcl_path = subprocess.run(["which", "sbcl"], capture_output=True, text=True)
    if sbcl_path.returncode != 0 or not sbcl_path.stdout.strip():
        return False
    # Check IDyOM database
    db_path = Path.home() / "idyom" / "db" / "database.sqlite"
    if not db_path.exists():
        return False
    # Check IDyOM source directory
    idyom_dir = Path.home() / "quicklisp" / "local-projects" / "idyom"
    if not idyom_dir.exists():
        return False

    # Check for a valid .sbclrc configuration file.
    sbclrc_path = Path.home() / ".sbclrc"
    if not sbclrc_path.exists():
        return False
    # Also, ensure our specific configuration is present.
    try:
        with open(sbclrc_path, 'r') as f:
            # This marker must match the one in install_idyom.sh
            if ";; IDyOM Configuration (v3)" not in f.read():
                return False
    except IOError:
        return False

    return True


def install_idyom():
    """Run the install_idyom.sh script to install IDyOM."""
    script_path = os.path.join(os.path.dirname(__file__), "install_idyom.sh")
    result = subprocess.run(["bash", script_path])
    if result.returncode != 0:
        raise RuntimeError("IDyOM installation failed. See output above.")

def start_idyom():
    """Start IDyOM."""
    # Import the library
    import py2lispIDyOM
    
    # Patch the broken _get_files_from_paths method
    from py2lispIDyOM.configuration import ExperimentLogger
    
    def fixed_get_files_from_paths(self, path):
        """Fixed version of _get_files_from_paths that properly filters MIDI and Kern files"""
        # Ensure the path ends with a slash for proper directory traversal
        if not path.endswith('/'):
            path = path + '/'
        
        glob_pattern = path + '*'
        files = []
        all_files = glob(glob_pattern)
        
        for file in all_files:
            # Skip directories, only process files
            if os.path.isfile(file):
                # Fix the broken condition in the original library
                file_extension = file[file.rfind("."):]
                if file_extension == ".mid" or file_extension == ".krn":
                    files.append(file)
            else:
                print(f"Skipping directory: {file}")
        
        return natsorted(files)
    
    # Replace the broken method with the fixed one
    ExperimentLogger._get_files_from_paths = fixed_get_files_from_paths
    
    return py2lispIDyOM

def run_idyom(dir_path=None, output_dir=None, experiment_name=None, description: Optional[str] = None):
    """
    Run IDyOM on a directory of MIDI files.

    This is the main top-level function. It handles checking for a valid
    IDyOM installation, prompting the user to install if needed, starting
    IDyOM, and running the analysis.
    """
    # 1. Check if IDyOM is installed and prompt user if not
    if not is_idyom_installed():
        print("IDyOM installation not found.")
        try:
            response = input("Would you like to install it now? (y/n): ")
            if response.lower().strip() == 'y':
                print("Running installation script...")
                install_idyom()
                print("Installation complete.")
            else:
                print("Installation cancelled. Aborting.")
                return None
        except (EOFError, KeyboardInterrupt):
            print("\nNon-interactive mode detected. Please install IDyOM manually by running install_idyom.sh")
            return None

    # 2. Start IDyOM and get the patched library object
    print("Starting IDyOM...")
    py2lisp = start_idyom()
    if not py2lisp:
        print("Fatal: Failed to start IDyOM.")
        return None

     # 3. Check if directory exists and has MIDI files
    if not dir_path or not Path(dir_path).exists():
        print(f"Error: MIDI directory not found or not provided: {dir_path}")
        return None
    
    midi_files = list(Path(dir_path).glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        print(f"No MIDI files found in {dir_path}!")
        return None
    
    try:
        # Create a safe experiment logger name to avoid path issues.
        if experiment_name:
            logger_name = experiment_name
        elif description:
            # Sanitize the description to make it a valid folder name.
            logger_name = "".join(c for c in description if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        else:
            logger_name = os.path.basename(dir_path)

        # Use output_dir for history if provided, otherwise use the input directory.
        history_folder = output_dir if output_dir else dir_path

        # HACK: Ensure the history folder path ends with a separator.
        # The py2lispIDyOM library concatenates paths incorrectly without it.
        if not history_folder.endswith(os.path.sep):
            history_folder += os.path.sep

        # Create IDyOM experiment directly
        print("Creating IDyOM experiment...")
        experiment = py2lisp.run.IDyOMExperiment(
            test_dataset_path=dir_path,
            pretrain_dataset_path=dir_path,  # Use same dataset for pretraining
            experiment_history_folder_path=history_folder,
            experiment_logger_name=logger_name
        )
        
        print("Setting experiment parameters...")
        experiment.set_parameters(
            target_viewpoints=['cpitch', 'onset'],
            source_viewpoints=['cpitch', 'onset'],
            models=':both',
            k=1,
            detail=3  # Detail level 3
        )
        
        print("Running IDyOM analysis...")
        experiment.run()
        print("Analysis complete!")
        
        # Get the experiment results path from the logger
        results_path = experiment.logger.this_exp_folder
        print(f"Results folder: {results_path}")
        
        # Export results to CSV
        print("Exporting results to CSV...")
        exporter = py2lisp.export.Export(results_path)
        exporter.export2csv()
        print("CSV export completed successfully")
        
        # Get experiment information
        print("\n=== Experiment Summary ===")
        exp_info = py2lisp.extract.ExperimentInfo(results_path)
        print(f"Number of melodies analyzed: {len(exp_info.melodies_dict)}")
        
        # Show available output keywords
        if exp_info.melodies_dict:
            first_melody = next(iter(exp_info.melodies_dict.values()))
            available_keywords = first_melody.get_idyom_output_keyword_list()
            print(f"Available output keywords: {available_keywords[:10]}...")  # Show first 10
        
        print(f"{description} completed successfully!")
        return results_path
        
    except Exception as e:
        print(f"Error running IDyOM analysis on {description}: {e}")
        return None


if __name__ == "__main__":
    # This block provides a simple example of how to use the run_idyom function.
    # It will run IDyOM on the MIDI files in the 'mid' directory.
    
    example_midi_dir = "/Users/davidwhyatt/Documents/mid"

    # Key Fix: Use a relative path for the output directory, just like in the working script.
    # The py2lispIDyOM library fails to initialize correctly when an absolute path
    # is used for the experiment history folder.
    example_output_dir = "idyom_results"
    Path(example_output_dir).mkdir(exist_ok=True)
    
    if Path(example_midi_dir).is_dir():
        print(f"--- Running IDyOM on example directory: '{example_midi_dir}' ---")
        run_idyom(
            dir_path=example_midi_dir,
            output_dir=example_output_dir,
            description="Example run on mid"
        )
    else:
        print(f"--- Example MIDI directory '{example_midi_dir}' not found. ---")
        print("Please create it and add some MIDI files to it,")
        print("or change the 'example_midi_dir' variable in the script.")
