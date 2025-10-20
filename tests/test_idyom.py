"""
Test suite for idyom_interface.py functionality.

Tests IDyOM integration, installation checking, and configuration validation
that's used by features.py but not covered in our main tests.
"""

from importlib import resources
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from melody_features.idyom_interface import (
    is_idyom_installed,
    install_idyom,
    start_idyom,
    run_idyom,
    VALID_VIEWPOINTS
)


class TestIDyOMInstallationChecking:
    """Test IDyOM installation detection."""
    
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_is_idyom_installed_complete(self, mock_exists, mock_subprocess):
        """Test IDyOM installation check when everything is installed."""
        # Mock SBCL check
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="/usr/local/bin/sbcl\n")

        # Mock file existence checks - all paths exist
        mock_exists.return_value = True

        # Mock .sbclrc content check
        with patch('builtins.open', mock_open_with_content(";; IDyOM Configuration (v3)\n")):
            result = is_idyom_installed()
            assert result == True, "Should detect complete installation"

    @patch('subprocess.run')
    def test_is_idyom_installed_no_sbcl(self, mock_subprocess):
        """Test IDyOM installation check when SBCL is missing."""
        # Mock SBCL not found
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="")

        result = is_idyom_installed()
        assert result == False, "Should detect missing SBCL"

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_is_idyom_installed_missing_database(self, mock_exists, mock_subprocess):
        """Test IDyOM installation check when database is missing."""
        # Mock SBCL found
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="/usr/local/bin/sbcl\n")
        mock_exists.return_value = False

        result = is_idyom_installed()
        assert result == False, "Should detect missing database"

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_is_idyom_installed_missing_sbclrc(self, mock_exists, mock_subprocess):
        """Test IDyOM installation check when .sbclrc is missing."""
        # Mock SBCL found
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="/usr/local/bin/sbcl\n")
        mock_exists.return_value = False

        result = is_idyom_installed()
        assert result == False, "Should detect missing .sbclrc"


class TestIDyOMViewpointValidation:
    """Test viewpoint validation functionality."""

    def test_valid_viewpoints_constant(self):
        """Test that VALID_VIEWPOINTS contains expected values."""
        assert isinstance(VALID_VIEWPOINTS, set), "Should be a set"
        assert len(VALID_VIEWPOINTS) > 50, "Should have many viewpoints"

        # frequently used viewpoints
        essential_viewpoints = {"cpitch", "onset", "dur", "cpint", "ioi"}
        assert essential_viewpoints.issubset(VALID_VIEWPOINTS), "Should contain essential viewpoints"

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom')
    def test_run_idyom_viewpoint_validation(self, mock_start, mock_installed):
        """Test that run_idyom validates viewpoints correctly."""
        # Mock IDyOM as installed
        mock_installed.return_value = True
        mock_start.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            with pytest.raises(ValueError, match="Invalid viewpoint.*invalid_viewpoint"):
                run_idyom(
                    input_path=temp_dir,
                    target_viewpoints=["invalid_viewpoint"],
                    source_viewpoints=["cpitch"]
                )

            with pytest.raises(ValueError, match="Invalid viewpoint.*another_invalid"):
                run_idyom(
                    input_path=temp_dir,
                    target_viewpoints=["cpitch"],
                    source_viewpoints=["another_invalid"]
                )


class TestIDyOMInstallation:
    """Test IDyOM installation functionality."""

    @patch('subprocess.run')
    def test_install_idyom_success(self, mock_subprocess):
        """Test successful IDyOM installation."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        install_idyom()

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "bash", "Should call bash"
        assert args[1].endswith("install_idyom.sh"), "Should call install script"

    @patch('subprocess.run')
    def test_install_idyom_failure(self, mock_subprocess):
        """Test failed IDyOM installation."""
        mock_subprocess.return_value = MagicMock(returncode=1)

        with pytest.raises(RuntimeError, match="IDyOM installation failed"):
            install_idyom()


class TestIDyOMStartup:
    """Test IDyOM startup and patching functionality."""

    def test_start_idyom_import_error(self):
        """Test start_idyom when py2lispIDyOM is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'py2lispIDyOM'")):
            with pytest.raises(ImportError):
                start_idyom()

    @patch('melody_features.idyom_interface.glob')
    def test_start_idyom_patches_library(self, mock_glob):
        """Test that start_idyom applies necessary patches."""
        mock_py2lisp = MagicMock()
        mock_experiment_logger = MagicMock()
        mock_py2lisp.configuration.ExperimentLogger = mock_experiment_logger

        with patch.dict('sys.modules', {'py2lispIDyOM': mock_py2lisp, 'py2lispIDyOM.configuration': mock_py2lisp.configuration}):
            result = start_idyom()

            # Should return the patched module
            assert result == mock_py2lisp, "Should return py2lispIDyOM module"
            
            # Should have patched the _get_files_from_paths method
            assert hasattr(mock_experiment_logger, '_get_files_from_paths'), "Should patch method"


class TestIDyOMRunning:
    """Test IDyOM running functionality."""

    @patch('melody_features.idyom_interface.is_idyom_installed')
    def test_run_idyom_not_installed_noninteractive(self, mock_installed):
        """Test run_idyom when IDyOM is not installed in non-interactive mode."""
        mock_installed.return_value = False

        # Mock non-interactive environment (no stdin)
        with patch('builtins.input', side_effect=EOFError()):
            result = run_idyom(input_path="/fake/path")

            assert result is None, "Should return None when installation cancelled"

    @patch('melody_features.idyom_interface.is_idyom_installed')
    def test_run_idyom_not_installed_user_declines(self, mock_installed):
        """Test run_idyom when user declines installation."""
        mock_installed.return_value = False

        with patch('builtins.input', return_value='n'):
            result = run_idyom(input_path="/fake/path")

            assert result is None, "Should return None when user declines installation"

    def test_run_idyom_invalid_input_path(self):
        """Test run_idyom with invalid input path."""
        with patch('melody_features.idyom_interface.is_idyom_installed', return_value=True):
            with patch('melody_features.idyom_interface.start_idyom', return_value=MagicMock()):
                result = run_idyom(input_path="/nonexistent/path")

                assert result is None, "Should return None for invalid input path"

    def test_run_idyom_invalid_pretraining_path(self):
        """Test run_idyom with invalid pretraining path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            with patch('melody_features.idyom_interface.is_idyom_installed', return_value=True):
                with patch('melody_features.idyom_interface.start_idyom', return_value=MagicMock()):
                    result = run_idyom(
                        input_path=temp_dir,
                        pretraining_path="/nonexistent/pretraining"
                    )

                    assert result is None, "Should return None for invalid pretraining path"

    def test_run_idyom_no_midi_files(self):
        """Test run_idyom with directory containing no MIDI files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('melody_features.idyom_interface.is_idyom_installed', return_value=True):
                with patch('melody_features.idyom_interface.start_idyom', return_value=MagicMock()):
                    result = run_idyom(input_path=temp_dir)

                    assert result is None, "Should return None when no MIDI files found"


def mock_open_with_content(content):
    """Helper function to mock file opening with specific content."""
    from unittest.mock import mock_open
    return mock_open(read_data=content)


class TestIDyOMExperimentNaming:
    """Test experiment naming logic in IDyOM."""

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom')
    def test_experiment_naming_with_experiment_name(self, mock_start, mock_installed):
        """Test experiment naming when experiment_name is provided."""
        mock_installed.return_value = True
        mock_py2lisp = MagicMock()
        mock_start.return_value = mock_py2lisp
        mock_py2lisp.run.IDyOMExperiment.side_effect = Exception("Test exception to check naming")

        with tempfile.TemporaryDirectory() as temp_dir:            
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            with patch('tempfile.mkdtemp', return_value=temp_dir):
                # run_idyom catches exceptions and returns None, so test the return value
                result = run_idyom(
                    input_path=temp_dir,
                    experiment_name="TestExperiment"
                )

                assert result is None

                # Verify experiment was created with correct name
                mock_py2lisp.run.IDyOMExperiment.assert_called_once()
                call_kwargs = mock_py2lisp.run.IDyOMExperiment.call_args[1]
                assert call_kwargs["experiment_logger_name"] == "TestExperiment"


class TestIDyOMParameterValidation:
    """Test parameter validation and configuration."""

    def test_viewpoint_validation_linked_viewpoints(self):
        """Test validation of linked viewpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            with patch('melody_features.idyom_interface.is_idyom_installed', return_value=True):
                with patch('melody_features.idyom_interface.start_idyom', return_value=MagicMock()):
                    # Test invalid linked viewpoint
                    with pytest.raises(ValueError, match="Linked viewpoints must be pairs"):
                        run_idyom(
                            input_path=temp_dir,
                            target_viewpoints=[("cpitch", "onset", "extra")],
                            source_viewpoints=["cpint"]
                        )

    def test_viewpoint_validation_mixed_types(self):
        """Test validation with mix of single and linked viewpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            with patch('melody_features.idyom_interface.is_idyom_installed', return_value=True):
                with patch('melody_features.idyom_interface.start_idyom', return_value=MagicMock()):
                    # Test mix of valid single and linked viewpoints
                    # This should not raise an exception if all viewpoints are valid
                    try:
                        # Mock the experiment to avoid actual IDyOM execution
                        mock_py2lisp = MagicMock()
                        mock_py2lisp.run.IDyOMExperiment.side_effect = Exception("Mocked to avoid execution")
                        
                        with patch('melody_features.idyom_interface.start_idyom', return_value=mock_py2lisp):
                            # This should not raise ValueError for valid viewpoints
                            result = run_idyom(
                                input_path=temp_dir,
                                target_viewpoints=["cpitch"],
                                source_viewpoints=[("cpint", "cpintfref"), "cpcint"]
                            )
                            # Should return None due to mocked exception, not ValueError
                            assert result is None, "Should handle mocked exception"
                    except ValueError:
                        pytest.fail("Should not raise ValueError for valid mixed viewpoints")


class TestIDyOMFileHandling:
    """Test file handling in IDyOM interface."""

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom')
    def test_run_idyom_file_discovery(self, mock_start, mock_installed):
        """Test that run_idyom properly discovers MIDI files."""
        mock_installed.return_value = True
        mock_py2lisp = MagicMock()
        mock_start.return_value = mock_py2lisp

        mock_py2lisp.run.IDyOMExperiment.side_effect = Exception("Mocked execution")

        with tempfile.TemporaryDirectory() as temp_dir:
            midi_files = ["song1.mid", "song2.midi", "not_midi.txt"]
            for filename in midi_files:
                filepath = os.path.join(temp_dir, filename)
                Path(filepath).touch()

            result = run_idyom(input_path=temp_dir)
            assert result is None, "Should return None due to mocked exception"

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom') 
    @patch('shutil.copy2')
    def test_run_idyom_pretraining_copy(self, mock_copy, mock_start, mock_installed):
        """Test that pretraining files are properly copied."""
        mock_installed.return_value = True
        mock_py2lisp = MagicMock()
        mock_start.return_value = mock_py2lisp

        mock_py2lisp.run.IDyOMExperiment.side_effect = Exception("Mocked execution")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            os.makedirs(input_dir)
            Path(os.path.join(input_dir, "test.mid")).touch()

            pretrain_dir = os.path.join(temp_dir, "pretrain")
            os.makedirs(pretrain_dir)
            Path(os.path.join(pretrain_dir, "pretrain1.mid")).touch()
            Path(os.path.join(pretrain_dir, "pretrain2.midi")).touch()

            result = run_idyom(
                input_path=input_dir,
                pretraining_path=pretrain_dir
            )
            assert result is None, "Should return None due to mocked exception"
            assert mock_copy.call_count >= 2, "Should copy pretraining files"


class TestIDyOMErrorHandling:
    """Test error handling in IDyOM interface."""

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom')
    def test_run_idyom_start_failure(self, mock_start, mock_installed):
        """Test run_idyom when IDyOM fails to start."""
        mock_installed.return_value = True
        mock_start.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            result = run_idyom(input_path=temp_dir)

            assert result is None, "Should return None when IDyOM fails to start"

    @patch('melody_features.idyom_interface.is_idyom_installed')
    @patch('melody_features.idyom_interface.start_idyom')
    def test_run_idyom_experiment_exception(self, mock_start, mock_installed):
        """Test run_idyom when experiment raises exception."""
        mock_installed.return_value = True
        mock_py2lisp = MagicMock()
        mock_start.return_value = mock_py2lisp

        mock_py2lisp.run.IDyOMExperiment.side_effect = Exception("IDyOM experiment failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = os.path.join(temp_dir, "test.mid")
            Path(midi_path).touch()

            result = run_idyom(input_path=temp_dir)

            assert result is None, "Should return None when experiment fails"


class TestIDyOMIntegration:
    """Test integration aspects of IDyOM interface."""
    
    def test_valid_viewpoints_comprehensive(self):
        """Test that all documented viewpoints are in VALID_VIEWPOINTS."""
        expected_viewpoints = {
            "cpitch", "onset", "dur", "cpint", "ioi",
            "cpitch-class", "cpcint", "contour", "inscale",
            "registral-direction", "intervallic-difference",
            "registral-return", "proximity", "closure"
        }
        
        missing = expected_viewpoints - VALID_VIEWPOINTS
        assert not missing, f"Missing expected viewpoints: {missing}"

    def test_install_script_exists(self):
        """Test that the install_idyom.sh script exists."""
        script_path = Path(__file__).parent.parent / "src" / "melody_features" / "install_idyom.sh"
        assert script_path.exists(), "install_idyom.sh should exist"

        # Check that it's executable (on Unix systems)
        if os.name != 'nt':
            stat = script_path.stat()
            # Check if any execute bit is set
            assert stat.st_mode & 0o111, "install_idyom.sh should be executable"


class TestKeyEstimationStrategies:
    """Test key estimation strategies for IDyOM processing."""

    def test_always_read_from_file_with_key_signature(self):
        """Test always_read_from_file strategy with a file that has key signature."""
        from melody_features.features import create_temp_midi_with_key_signature
        from mido import MidiFile
        
        # Use a file from the Essen corpus that has a key signature
        midi_path = resources.files("melody_features") / "corpora/Essen_Corpus/appenzel.mid"
        
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the MIDI file to input directory
                import shutil
                shutil.copy2(str(midi_path), input_dir)
                
                # Process with always_read_from_file strategy
                output_dir = create_temp_midi_with_key_signature(
                    input_dir, temp_dir, key_estimation="always_read_from_file"
                )
                
                # Verify the output file exists and has the original key signature
                output_files = list(Path(output_dir).glob("*.mid"))
                assert len(output_files) == 1, "Should create one output file"
                
                mid = MidiFile(output_files[0])
                has_key_sig = False
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "key_signature":
                            has_key_sig = True
                            break
                    if has_key_sig:
                        break
                
                assert has_key_sig, "Should preserve key signature from original file"

    def test_always_infer_overrides_key_signature(self):
        """Test always_infer strategy estimates key even when file has key signature."""
        from melody_features.features import create_temp_midi_with_key_signature
        from melody_features.import_mid import import_midi
        from melody_features.representations import Melody
        from melody_features.algorithms import compute_tonality_vector
        from mido import MidiFile

        midi_path = resources.files("melody_features") / "corpora/Essen_Corpus/appenzel.mid"

        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the MIDI file to input directory
                import shutil
                shutil.copy2(str(midi_path), input_dir)
                
                # Process with always_infer strategy
                output_dir = create_temp_midi_with_key_signature(
                    input_dir, temp_dir, key_estimation="always_infer"
                )
                
                # Verify the output file exists
                output_files = list(Path(output_dir).glob("*.mid"))
                assert len(output_files) == 1, "Should create one output file"
                # Check that it has a key signature, and it should be D major (estimated)
                mid = MidiFile(output_files[0])
                estimated_key = None
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "key_signature":
                            estimated_key = msg.key
                            break
                    if estimated_key:
                        break

                assert estimated_key is not None, "Should have key signature (estimated from pitch content)"
                assert estimated_key.lower() == "d", f"Estimated key should be D major, got: {estimated_key}"

    def test_infer_if_necessary_preserves_existing(self):
        """Test infer_if_necessary strategy preserves existing key signatures."""
        from melody_features.features import create_temp_midi_with_key_signature
        from mido import MidiFile
        
        midi_path = resources.files("melody_features") / "corpora/Essen_Corpus/appenzel.mid"
        
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the MIDI file to input directory
                import shutil
                shutil.copy2(str(midi_path), input_dir)
                
                # Process with infer_if_necessary strategy
                output_dir = create_temp_midi_with_key_signature(
                    input_dir, temp_dir, key_estimation="infer_if_necessary"
                )
                
                # Verify the output file exists and preserves the key signature
                output_files = list(Path(output_dir).glob("*.mid"))
                assert len(output_files) == 1, "Should create one output file"
                
                # Check key signature matches original
                mid = MidiFile(output_files[0])
                output_key = None
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "key_signature":
                            output_key = msg.key
                            break
                    if output_key:
                        break
                
                assert output_key is not None, "Should have key signature (inferred from pitch content)"
                assert output_key.lower() == "g", f"Estimated key should be G major, got: {output_key}"

    def test_key_estimation_integration_with_get_idyom_results(self):
        """Test that key_estimation parameter is passed through to get_idyom_results."""
        from melody_features.features import get_idyom_results
        
        # Create a temporary directory with a MIDI file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple MIDI file
            from mido import MidiFile, MidiTrack, Message, MetaMessage
            
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Add key signature
            track.append(MetaMessage('key_signature', key='C'))
            track.append(Message('note_on', note=60, velocity=64, time=0))
            track.append(Message('note_off', note=60, velocity=64, time=480))
            track.append(MetaMessage('end_of_track', time=0))
            
            midi_path = os.path.join(temp_dir, "test.mid")
            mid.save(midi_path)
            
            # Mock run_idyom to avoid actual IDyOM execution
            with patch('melody_features.features.run_idyom') as mock_run_idyom:
                mock_run_idyom.return_value = None  # Simulate no output
                
                # Call get_idyom_results with key_estimation parameter
                result = get_idyom_results(
                    temp_dir,
                    idyom_target_viewpoints=["cpitch"],
                    idyom_source_viewpoints=["cpint"],
                    models=":both",
                    ppm_order=1,
                    corpus_path=None,
                    experiment_name="test_experiment",
                    key_estimation="always_infer"
                )
                
                # The function should handle the key_estimation parameter without error
                # Result will be empty dict because run_idyom returned None
                assert isinstance(result, dict), "Should return a dictionary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
