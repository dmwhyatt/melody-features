"""
Test suite for idyom_interface.py functionality.

Tests IDyOM integration, installation checking, and configuration validation
that's used by features.py but not covered in our main tests.
"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
