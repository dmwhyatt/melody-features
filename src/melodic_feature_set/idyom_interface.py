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
from typing import Optional, Set
import shutil

# A set of known valid viewpoints for IDyOM. This prevents typos and unsupported values.
VALID_VIEWPOINTS = {
    'onset', 'cpitch', 'dur', 'keysig', 'mode', 'tempo', 'pulses', 'barlength', 
    'deltast', 'bioi', 'phrase', 'mpitch', 'accidental', 'dyn', 'voice', 'ornament', 
    'comma', 'articulation', 'ioi', 'posinbar', 'dur-ratio', 'referent', 'cpint',
    'contour', 'cpitch-class', 'cpcint', 'cpintfref', 'cpintfip', 'cpintfiph', 'cpintfib',
    'inscale', 'ioi-ratio', 'ioi-contour', 'metaccent', 'bioi-ratio', 'bioi-contour',
    'lphrase', 'cpint-size', 'newcontour', 'cpcint-size', 'cpcint-2', 'cpcint-3',
    'cpcint-4', 'cpcint-5', 'cpcint-6', 'octave', 'tessitura', 'mpitch-class', 
    'registral-direction', 'intervallic-difference', 'registral-return', 'proximity',
    'closure', 'fib', 'crotchet', 'tactus', 'fiph', 'liph', 'thr-cpint-fib',
    'thr-cpint-fiph', 'thr-cpint-liph', 'thr-cpint-crotchet', 'thr-cpint-tactus', 
    'thr-cpintfref-liph', 'thr-cpintfref-fib','thr-cpint-cpintfref-liph' ,'thr-cpint-cpintfref-fib'}

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

def run_idyom(
    input_path=None,
    pretraining_path=None,
    output_dir='.', 
    experiment_name=None, 
    description: Optional[str] = None,
    target_viewpoints=['cpitch', 'onset'],
    source_viewpoints=['cpitch', 'onset'],
    models=':both',
    k=1,
    detail=3
):
    """
    Run IDyOM on a directory of MIDI files.

    This is the main top-level function. It handles checking for a valid
    IDyOM installation, prompting the user to install if needed, starting
    IDyOM, and running the analysis. If no `pretraining_path` is supplied,
    no pre-training will be performed.
    """
    # --- Viewpoint Validation ---
    all_provided_viewpoints = set()
    
    # Process both target and source viewpoints
    for viewpoints in [target_viewpoints, source_viewpoints]:
        for viewpoint in viewpoints:
            # Handle both single viewpoints and tuples
            if isinstance(viewpoint, (list, tuple)):
                if len(viewpoint) != 2:
                    raise ValueError(
                        f"Linked viewpoints must be pairs, got {len(viewpoint)} elements: {viewpoint}"
                    )
                all_provided_viewpoints.update(viewpoint)
            else:
                all_provided_viewpoints.add(viewpoint)
    
    invalid_viewpoints = all_provided_viewpoints - VALID_VIEWPOINTS

    # TODO: Support linked viewpoints.
    if invalid_viewpoints:
        raise ValueError(
            f"Invalid viewpoint(s) provided: {', '.join(invalid_viewpoints)}.\n"
            f"Valid viewpoints are: {', '.join(sorted(list(VALID_VIEWPOINTS)))}"
        )
        
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

    # --- Path Validation ---
    if not input_path or not Path(input_path).exists():
        print(f"Error: Input MIDI directory not found or not provided: {input_path}")
        return None
    # Only validate the pretraining path if it has been provided.
    if pretraining_path and not Path(pretraining_path).exists():
        print(f"Error: Pre-training MIDI directory not found: {pretraining_path}")
        return None
    
    midi_files = list(Path(input_path).glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files in input directory.")
    
    if len(midi_files) == 0:
        print(f"No MIDI files found in {input_path}!")
        return None
    
    try:
        # Create a safe experiment logger name to avoid path issues.
        if experiment_name:
            logger_name = experiment_name
        elif description:
            # Sanitize the description to make it a valid folder name.
            logger_name = "".join(c for c in description if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        else:
            logger_name = os.path.basename(input_path)

        # Use a temporary directory for the py2lispIDyOM library's output.
        # we will delete it later
        temp_history_folder = str(Path("idyom_temp_output").resolve())

        if not temp_history_folder.endswith('/'):
            temp_history_folder += '/'

        # Create IDyOM experiment directly
        print("Creating IDyOM experiment...")
        experiment = py2lisp.run.IDyOMExperiment(
            test_dataset_path=input_path,
            pretrain_dataset_path=pretraining_path,
            experiment_history_folder_path=temp_history_folder,
            experiment_logger_name=logger_name
        )
        
        print("Setting experiment parameters...")
        experiment.set_parameters(
            target_viewpoints=target_viewpoints,
            source_viewpoints=source_viewpoints,
            models=models,
            k=k,
            detail=detail
        )
        
        print("Running IDyOM analysis...")
        experiment.run()
        print("Analysis complete!")
        
        results_path = Path(experiment.logger.this_exp_folder)
        
        # --- Clean up and extract only the .dat file ---
        # The .dat file is located in a specific subfolder.
        data_folder_path = results_path / "experiment_output_data_folder"
        
        if not data_folder_path.exists():
            print(f"Error: Expected data folder not found at {data_folder_path}.")
            return None

        dat_files = list(data_folder_path.glob('*.dat'))
        
        if not dat_files:
            print(f"Warning: No .dat file found in {data_folder_path}.")
            return None
            
        dat_file_path = dat_files[0]
        if len(dat_files) > 1:
            print(f"Warning: Found multiple .dat files, using the first one: {dat_file_path}")

        # Move the .dat file and cleanup everything else.
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{logger_name}.dat"
        
        shutil.move(str(dat_file_path), str(destination_path))
        shutil.rmtree(temp_history_folder)
        
        print(f"✓ {description} completed successfully! Output: {destination_path}")
        return str(destination_path)

    except Exception as e:
        print(f"Error running IDyOM analysis on {description}: {e}")
        return None


if __name__ == "__main__":
    # This block provides a simple example of how to use the run_idyom function.
    # It will run IDyOM on the MIDI files in the supplied directory.
    
    example_midi_dir = "/Users/davidwhyatt/Downloads/Essen_First_10"
    
    if Path(example_midi_dir).is_dir():
        print(f"--- Running IDyOM on example directory: '{example_midi_dir}' ---")
        run_idyom(
            input_path=example_midi_dir,
            # The output .dat file will be placed in the current directory by default.
            description="Example run on Essen First 10",
            # Target viewpoints can only be onset, cpitch, or ioi.
            # Typically people focus on analysing just cpitch, because that's
            # where IDyOM has been most validated.
            target_viewpoints=['cpitch'],
            # Source viewpoints can be all kinds of things.
            # Marcus's favourite is a linked viewpoint comprising:
            # - cpint: the chromatic interval between the current and previous note
            # - cpintfref: the chromatic interval between the current note and the tonic
            # Note however that we need to make sure that key signature is encoded in the MIDI file.
            # source_viewpoints=['cpitch'],
            # source_viewpoints=['cpintfref'],
            source_viewpoints=[('cpint', 'cpintfref'), "cpcint"],
            models=':both',
            detail=2
        )
    else:
        print(f"--- Example MIDI directory '{example_midi_dir}' not found. ---")
        print("Please create it and add some MIDI files to it,")
        print("or change the 'example_midi_dir' variable in the script.")
