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
import logging
import os
import re
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Optional

from natsort import natsorted
from melody_features.corpus import get_corpus_files
import tempfile

# A set of known valid viewpoints for IDyOM. This prevents typos and unsupported values.
VALID_VIEWPOINTS = {
    "onset",
    "cpitch",
    "dur",
    "keysig",
    "mode",
    "tempo",
    "pulses",
    "barlength",
    "deltast",
    "bioi",
    "phrase",
    "mpitch",
    "accidental",
    "dyn",
    "voice",
    "ornament",
    "comma",
    "articulation",
    "ioi",
    "posinbar",
    "dur-ratio",
    "referent",
    "cpint",
    "contour",
    "cpitch-class",
    "cpcint",
    "cpintfref",
    "cpintfip",
    "cpintfiph",
    "cpintfib",
    "inscale",
    "ioi-ratio",
    "ioi-contour",
    "metaccent",
    "bioi-ratio",
    "bioi-contour",
    "lphrase",
    "cpint-size",
    "newcontour",
    "cpcint-size",
    "cpcint-2",
    "cpcint-3",
    "cpcint-4",
    "cpcint-5",
    "cpcint-6",
    "octave",
    "tessitura",
    "mpitch-class",
    "registral-direction",
    "intervallic-difference",
    "registral-return",
    "proximity",
    "closure",
    "fib",
    "crotchet",
    "tactus",
    "fiph",
    "liph",
    "thr-cpint-fib",
    "thr-cpint-fiph",
    "thr-cpint-liph",
    "thr-cpint-crotchet",
    "thr-cpint-tactus",
    "thr-cpintfref-liph",
    "thr-cpintfref-fib",
    "thr-cpint-cpintfref-liph",
    "thr-cpint-cpintfref-fib",
}


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
        with open(sbclrc_path, "r") as f:
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
        logger = logging.getLogger("melody_features")
        # Ensure the path ends with a slash for proper directory traversal
        if not path.endswith("/"):
            path = path + "/"

        glob_pattern = path + "*"
        files = []
        all_files = glob(glob_pattern)

        for file in all_files:
            # Skip directories, only process files
            if os.path.isfile(file):
                # Fix the broken condition in the original library
                file_extension = file[file.rfind(".") :]
                if file_extension == ".mid" or file_extension == ".krn":
                    files.append(file)
            else:
                logger.debug(f"Skipping directory: {file}")

        return natsorted(files)

    # Replace the broken method with the fixed one
    ExperimentLogger._get_files_from_paths = fixed_get_files_from_paths

    return py2lispIDyOM


def run_idyom(
    input_path=None,
    pretraining_path=None,
    output_dir=".",
    experiment_name=None,
    description: Optional[str] = None,
    target_viewpoints=["cpitch", "onset"],
    source_viewpoints=["cpitch", "onset"],
    models=":both",
    k=1,
    detail=3,
    ppm_order=None,
    sbcl_dynamic_space_size: Optional[int] = 8192,
):
    logger = logging.getLogger("melody_features")
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
                if len(viewpoint) < 2:
                    raise ValueError(
                        f"Linked viewpoints must have at least 2 elements, got {len(viewpoint)} elements: {viewpoint}"
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

    logger = logging.getLogger("melody_features")
    if not is_idyom_installed():
        logger.warning("IDyOM installation not found.")
        try:
            response = input("Would you like to install it now? (y/n): ")
            if response.lower().strip() == "y":
                logger.info("Running installation script...")
                install_idyom()
                logger.info("Installation complete.")
            else:
                logger.info("Installation cancelled. Aborting.")
                return None
        except (EOFError, KeyboardInterrupt):
            logger.warning(
                "Non-interactive mode detected. Please install IDyOM manually by running install_idyom.sh"
            )
            return None

    logger.info("Starting IDyOM...")
    py2lisp = start_idyom()
    if not py2lisp:
        logger.error("Failed to start IDyOM.")
        return None

    if not input_path or not Path(input_path).exists():
        logger.error(f"Input MIDI directory not found or not provided: {input_path}")
        return None
    if pretraining_path and not Path(pretraining_path).exists():
        logger.error(f"Pre-training MIDI directory not found: {pretraining_path}")
        return None

    # Debug: Check what files are in the pretraining directory
    if pretraining_path:
        pretrain_files = list(Path(pretraining_path).glob("*.mid"))
        pretrain_files.extend(list(Path(pretraining_path).glob("*.midi")))
        logger.info(
            f"Found {len(pretrain_files)} MIDI files in pretraining directory: {pretraining_path}"
        )
        if len(pretrain_files) == 0:
            logger.warning(
                f"No MIDI files found in pretraining directory: {pretraining_path}"
            )
    else:
        logger.info("No pretraining path provided, will run without pretraining")

    # Look for both .mid and .midi files
    midi_files = list(Path(input_path).glob("*.mid"))
    midi_files.extend(list(Path(input_path).glob("*.midi")))
    logger.info(f"Found {len(midi_files)} MIDI files in input directory.")

    if len(midi_files) == 0:
        logger.error(f"No MIDI files found in {input_path}!")
        return None

    try:
        if experiment_name:
            logger_name = experiment_name
        elif description:
            # Ensure it has a valid folder name
            logger_name = (
                "".join(c for c in description if c.isalnum() or c in (" ", "_"))
                .rstrip()
                .replace(" ", "_")
            )
        else:
            logger_name = os.path.basename(input_path)

        # Use a unique temporary directory for each experiment to avoid conflicts
        import tempfile

        temp_history_folder = str(Path(tempfile.mkdtemp(prefix="idyom_")).resolve())

        if not temp_history_folder.endswith("/"):
            temp_history_folder += "/"

        # Create IDyOM experiment directly
        logger.info(f"Creating IDyOM experiment with logger_name: {logger_name}")
        logger.info(f"Using temp_history_folder: {temp_history_folder}")

        # Initialize pretrain_dir variable
        pretrain_dir = None
        final_pretrain_path = pretraining_path

        # If pretraining path is provided, copy it to the experiment directory
        if pretraining_path:
            pretrain_dir = Path(temp_history_folder) / "pretrain_dataset"
            pretrain_dir.mkdir(parents=True, exist_ok=True)

            # Copy all MIDI files from pretraining directory to experiment pretrain directory
            # Convert .midi files to .mid files for IDyOM compatibility
            pretrain_files = list(Path(pretraining_path).glob("*.mid"))
            midi_files = list(Path(pretraining_path).glob("*.midi"))

            # Copy .mid files as-is
            for file in pretrain_files:
                shutil.copy2(file, pretrain_dir)

            # Copy .midi files with .mid extension
            for file in midi_files:
                new_name = file.stem + ".mid"
                shutil.copy2(file, pretrain_dir / new_name)

            logger.info(
                f"Copied {len(pretrain_files)} .mid files and {len(midi_files)} .midi files (as .mid) to {pretrain_dir}"
            )

            # Use the copied pretraining directory if it exists and has files
            if pretrain_dir.exists() and any(pretrain_dir.iterdir()):
                final_pretrain_path = str(pretrain_dir)
                logger.info(
                    f"Using copied pretraining directory: {final_pretrain_path}"
                )
            else:
                logger.warning(
                    f"Pretraining directory is empty or doesn't exist: {pretrain_dir}"
                )
                final_pretrain_path = None

        experiment = py2lisp.run.IDyOMExperiment(
            test_dataset_path=input_path,
            pretrain_dataset_path=final_pretrain_path,
            experiment_history_folder_path=temp_history_folder,
            experiment_logger_name=logger_name,
        )

        logger.info("Setting experiment parameters...")

        parameter_kwargs = {
            "target_viewpoints": target_viewpoints,
            "source_viewpoints": source_viewpoints,
            "models": models,
            "k": k,
            "detail": detail,
        }

        if ppm_order is not None:
            parameter_kwargs["ltmo_order_bound"] = ppm_order
            parameter_kwargs["stmo_order_bound"] = ppm_order
        else:
            logger.debug("ppm_order is None; will use inf ltmo/stmo order bounds")

        experiment.set_parameters(**parameter_kwargs)

        logger.info("Running IDyOM analysis...")

        original_os_system = getattr(py2lisp.run, "os", None)
        original_system_call = None
        patched_system = None
        if sbcl_dynamic_space_size is not None and original_os_system is not None:
            original_system_call = original_os_system.system

            def _patched_system(command: str) -> int:
                if command.strip().startswith("sbcl"):
                    if "--dynamic-space-size" in command:
                        command = re.sub(
                            r"--dynamic-space-size\s+\d+",
                            f"--dynamic-space-size {sbcl_dynamic_space_size}",
                            command,
                        )
                    else:
                        command = command.replace(
                            "sbcl",
                            f"sbcl --dynamic-space-size {sbcl_dynamic_space_size}",
                            1,
                        )
                    logger.debug(
                        "Using SBCL dynamic space size %s MB for command: %s",
                        sbcl_dynamic_space_size,
                        command,
                    )
                return original_system_call(command)

            patched_system = _patched_system
            original_os_system.system = patched_system

        try:
            experiment.run()
        finally:
            if (
                patched_system is not None
                and original_os_system is not None
                and original_system_call is not None
            ):
                original_os_system.system = original_system_call

        logger.info("IDyOM analysis complete!")

        results_path = Path(experiment.logger.this_exp_folder)

        # find the dat file in temp output
        data_folder_path = results_path / "experiment_output_data_folder"

        if not data_folder_path.exists():
            logger.error(f"Expected data folder not found at {data_folder_path}.")
            return None

        dat_files = list(data_folder_path.glob("*.dat"))

        if not dat_files:
            logger.warning(f"No .dat file found in {data_folder_path}.")
            return None

        dat_file_path = dat_files[0]
        if len(dat_files) > 1:
            logger.warning(
                f"Found multiple .dat files, using the first one: {dat_file_path}"
            )

        # Move the .dat file and cleanup everything else
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{logger_name}.dat"

        shutil.move(str(dat_file_path), str(destination_path))
        shutil.rmtree(temp_history_folder)

        logger.info(
            f"IDyOM processing completed successfully! Output: {destination_path}"
        )
        return str(destination_path)

    except Exception as e:
        logger.error(f"Error running IDyOM analysis on {description}: {e}")
        return None


if __name__ == "__main__":
    # This block provides a simple example of how to use the run_idyom function.
    # It will run IDyOM on the first 10 MIDI files from the Essen Folksong Collection.

    logger = logging.getLogger("melody_features")
    
    # Get the first 10 files from the Essen corpus
    essen_files = get_corpus_files("essen", max_files=10)
    
    if not essen_files:
        logger.warning("No MIDI files found in Essen corpus.")
        logger.info("Please ensure the corpus is properly installed.")
    else:
        # Create a temporary directory and copy the first 10 files there
        temp_dir = tempfile.mkdtemp(prefix="idyom_example_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        try:
            for file_path in essen_files:
                shutil.copy2(file_path, temp_dir)
            
            logger.info(f"--- Running IDyOM on first 10 files from Essen corpus ---")
            logger.info(f"Using temporary directory: {temp_dir}")
            
            result_path = run_idyom(
                input_path=temp_dir,
                # The output .dat file will be placed in the current directory by default.
                description="Example run on Essen First 10",
                # Target viewpoints can only be onset, cpitch, or ioi.
                # Typically people focus on analysing just cpitch, because that's
                # where IDyOM has been most validated.
                target_viewpoints=["cpitch"],
                # Source viewpoints can be all kinds of things.
                # Marcus's favourite is a linked viewpoint comprising:
                # - cpint: the chromatic interval between the current and previous note
                # - cpintfref: the chromatic interval between the current note and the tonic
                # Note however that we need to make sure that key signature is encoded in the MIDI file.
                # source_viewpoints=['cpitch'],
                # source_viewpoints=['cpintfref'],
                source_viewpoints=[("cpint", "cpintfref"), "cpcint"],
                models=":both",
                detail=2,
                ppm_order=1,  # Set the order of the PPM models
            )
            if result_path is not None:
                print(f"IDyOM output .dat file located at: {result_path}")
                try:
                    with open(result_path, "r") as dat_file:
                        for i, line in enumerate(dat_file):
                            print(line.rstrip())
                            if i >= 19:
                                print("...(output truncated)...")
                                break
                except Exception as e:
                    print(f"Could not read .dat file: {e}")
            else:
                print("IDyOM did not produce an output .dat file.")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")
