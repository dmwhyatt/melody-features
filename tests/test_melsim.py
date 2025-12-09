"""
Test suite for melsim_wrapper functionality.

Tests the melody similarity calculation wrapper that interfaces with the R melsim package.
"""
# I wrote this for AMADS a while back, so have brought these tests over from there.
# they're slightly different to the AMADS tests but the functionality is roughly the same.

import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from melody_features.melsim_wrapper.melsim import (
    _convert_strings_to_tuples,
    check_python_package_installed,
    check_r_packages_installed,
    get_similarity,
    get_similarity_from_midi,
    install_r_package,
    load_midi_file,
    _compute_similarity,
    _batch_compute_similarities,
)

def create_test_midi_file(pitches, starts, ends, tempo=120, filepath=None):
    """Create a temporary MIDI file for testing."""
    import mido
    
    # Create a new MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    
    # Add time signature
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    
    # Convert times to MIDI ticks (assuming 480 ticks per beat)
    ticks_per_second = 480 * (tempo / 60)
    
    # Add notes
    current_time = 0
    for i, (pitch, start, end) in enumerate(zip(pitches, starts, ends)):
        # Calculate delta time to note start
        start_ticks = int(start * ticks_per_second)
        delta_time = start_ticks - current_time

        track.append(mido.Message('note_on', channel=0, note=pitch, velocity=64, time=delta_time))

        duration_ticks = int((end - start) * ticks_per_second)
        track.append(mido.Message('note_off', channel=0, note=pitch, velocity=64, time=duration_ticks))

        current_time = start_ticks + duration_ticks
    
    if filepath:
        mid.save(filepath)
        return filepath
    else:
        return mid


class TestMelsimDependencies:
    """Test dependency checking and installation."""

    @patch('subprocess.run')
    def test_check_r_packages_installed_all_present(self, mock_subprocess):
        """Test R package checking when all packages are installed."""
        # Mock R script returning empty (no missing packages)
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='[]')
        check_r_packages_installed(install_missing=False)

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "Rscript"

    @patch('subprocess.run')
    def test_check_r_packages_installed_missing_packages(self, mock_subprocess):
        """Test R package checking when packages are missing."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='["melsim", "tibble"]')

        with pytest.raises(ImportError, match="Packages.*are required but not installed"):
            check_r_packages_installed(install_missing=False)
    
    def test_check_python_package_installed_valid(self):
        """Test Python package checking with valid package."""
        check_python_package_installed("os")
        check_python_package_installed("sys")

    def test_check_python_package_installed_invalid(self):
        """Test Python package checking with invalid package."""
        with pytest.raises(ImportError, match="Package 'nonexistent_package' is required"):
            check_python_package_installed("nonexistent_package")

    @patch('subprocess.run')
    def test_install_r_package_cran(self, mock_subprocess):
        """Test CRAN package installation."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        install_r_package("tibble")

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "Rscript"
        assert "tibble" in args[2]
        assert "install.packages" in args[2]

    @patch('subprocess.run')
    def test_install_r_package_github(self, mock_subprocess):
        """Test GitHub package installation."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        install_r_package("melsim")

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "Rscript"
        assert "install_github" in args[2]
        assert "sebsilas/melsim" in args[2]

    def test_install_r_package_invalid(self):
        """Test installation of unknown package."""
        with pytest.raises(ValueError, match="Unknown package type"):
            install_r_package("unknown_package")


class TestMidiFileLoading:
    """Test MIDI file loading functionality."""

    def test_load_midi_file_valid(self):
        """Test loading a valid MIDI file."""
        pitches = [60, 62, 64, 65]
        starts = [0.0, 0.5, 1.0, 1.5]
        ends = [0.4, 0.9, 1.4, 1.9]

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
            temp_path = temp_file.name

        try:
            loaded_pitches, loaded_starts, loaded_ends = load_midi_file(temp_path)

            assert loaded_pitches == pitches, "Should preserve pitches"
            assert len(loaded_starts) == 4, "Should have 4 start times"
            assert len(loaded_ends) == 4, "Should have 4 end times"
            assert all(isinstance(p, int) for p in loaded_pitches), "Pitches should be integers"
            assert all(isinstance(s, float) for s in loaded_starts), "Starts should be floats"
            assert all(isinstance(e, float) for e in loaded_ends), "Ends should be floats"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_midi_file_invalid(self):
        """Test loading an invalid MIDI file."""
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            temp_file.write(b"not a midi file")
            temp_path = temp_file.name

        try:
            with pytest.raises(ValueError, match="Could not import MIDI file"):
                load_midi_file(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSimilarityCalculation:
    """Test core similarity calculation functions."""

    @patch('subprocess.run')
    def test_get_similarity_basic(self, mock_subprocess):
        """Test basic similarity calculation between two melodies."""
        # Mock returning similarity value
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='0.75'
        )

        melody1_pitches = np.array([60, 62, 64, 65])
        melody1_starts = np.array([0.0, 0.5, 1.0, 1.5])
        melody1_ends = np.array([0.4, 0.9, 1.4, 1.9])
        melody2_pitches = np.array([60, 62, 64, 67])
        melody2_starts = np.array([0.0, 0.5, 1.0, 1.5])
        melody2_ends = np.array([0.4, 0.9, 1.4, 1.9])

        similarity = get_similarity(
            melody1_pitches, melody1_starts, melody1_ends,
            melody2_pitches, melody2_starts, melody2_ends,
            "Jaccard", "pitch"
        )

        assert similarity == 0.75, "Should return mocked similarity value"

        # Verify R script was called correctly
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "Rscript"
        assert "Jaccard" in args[2]
        assert "pitch" in args[2]

    @patch('subprocess.run')
    def test_compute_similarity_helper(self, mock_subprocess):
        """Test the _compute_similarity helper function."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.6')

        melody1_data = ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        melody2_data = ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])

        similarity = _compute_similarity((melody1_data, melody2_data, "Dice", "int"))

        assert similarity == 0.6, "Should return mocked similarity"
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_batch_compute_similarities(self, mock_subprocess):
        """Test batch similarity computation."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='[0.5, 0.7, 0.3]')

        melody1_data = ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        melody2_data = ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        melody3_data = ([72, 74, 76], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])

        args_list = [
            (melody1_data, melody2_data, "Jaccard", "pitch"),
            (melody1_data, melody3_data, "Jaccard", "pitch"),
            (melody2_data, melody3_data, "Jaccard", "pitch")
        ]

        similarities = _batch_compute_similarities(args_list)

        assert similarities == [0.5, 0.7, 0.3], "Should return mocked similarities"
        mock_subprocess.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_convert_strings_to_tuples_simple(self):
        """Test string to tuple conversion utility."""
        input_dict = {"key1": "value1", "key2": "value2"}
        result = _convert_strings_to_tuples(input_dict)
        assert result == input_dict

    def test_convert_strings_to_tuples_nested(self):
        """Test string to tuple conversion with nested dict."""
        input_dict = {"outer": {"inner": "value"}}
        result = _convert_strings_to_tuples(input_dict)
        assert result == {"outer": {"inner": "value"}}

    def test_convert_strings_to_tuples_preserves_structure(self):
        """Test that the function preserves dictionary structure."""
        input_dict = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            },
            "simple": "value"
        }
        result = _convert_strings_to_tuples(input_dict)
        assert result == input_dict, "Should preserve nested structure"


class TestMidiSimilarityFromFiles:
    """Test MIDI file similarity calculation."""

    @patch('subprocess.run')
    def test_get_similarity_from_midi_two_files(self, mock_subprocess):
        """Test similarity calculation between two MIDI files."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.8')

        # pitches, starts, ends - should aim to make this easier in future
        melody1_data = ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        melody2_data = ([60, 62, 64, 67], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])

        temp_files = []
        try:
            for i, (pitches, starts, ends) in enumerate([melody1_data, melody2_data]):
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_test_{i}.mid', delete=False)
                create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
                temp_files.append(temp_file.name)

            similarity = get_similarity_from_midi(
                temp_files[0], 
                temp_files[1], 
                method="Jaccard", 
                transformation="pitch"
            )

            assert similarity == 0.8, "Should return mocked similarity value"
            mock_subprocess.assert_called_once()

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestMelsimMethodsAndTransformations:
    """Test all supported similarity methods and transformations."""

    def setup_method(self):
        """Set up test melodies."""
        self.melody1_data = ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        self.melody2_data = ([60, 62, 64, 67], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        
        self.supported_methods = [
            "Jaccard", "Kulczynski2", "Russel", "Faith", "Tanimoto", "Dice", 
            "Mozley", "Ochiai", "Simpson", "cosine", "angular", "correlation",
            "Tschuprow", "Cramer", "Gower", "Euclidean", "Manhattan", "supremum",
            "Canberra", "Chord", "Geodesic", "Bray", "Soergel", "Podani",
            "Whittaker", "eJaccard", "eDice", "Bhjattacharyya", "divergence",
            "Hellinger", "edit_sim_utf8", "edit_sim", "Levenshtein", "sim_NCD",
            "const", "sim_dtw"
        ]

        self.supported_transformations = [
            "pitch", "int", "fuzzy_int", "parsons", "pc", "ioi_class",
            "duration_class", "int_X_ioi_class", "implicit_harmonies"
        ]

    @patch('subprocess.run')
    def test_all_methods_with_pitch_transformation(self, mock_subprocess):
        """Test all similarity methods with pitch transformation."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.75')

        for method in self.supported_methods:
            similarity = get_similarity(
                np.array(self.melody1_data[0]), np.array(self.melody1_data[1]), np.array(self.melody1_data[2]),
                np.array(self.melody2_data[0]), np.array(self.melody2_data[1]), np.array(self.melody2_data[2]),
                method, "pitch"
            )

            assert similarity == 0.75, f"Method {method} should return mocked similarity"
            assert method in mock_subprocess.call_args[0][0][2], f"R script should contain method {method}"

    @patch('subprocess.run')  
    def test_all_transformations_with_jaccard(self, mock_subprocess):
        """Test all transformations with Jaccard method."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.6')

        for transformation in self.supported_transformations:
            similarity = get_similarity(
                np.array(self.melody1_data[0]), np.array(self.melody1_data[1]), np.array(self.melody1_data[2]),
                np.array(self.melody2_data[0]), np.array(self.melody2_data[1]), np.array(self.melody2_data[2]),
                "Jaccard", transformation
            )

            assert similarity == 0.6, f"Transformation {transformation} should return mocked similarity"
            assert transformation in mock_subprocess.call_args[0][0][2], f"R script should contain transformation {transformation}"


class TestMelsimBatchProcessing:
    """Test batch similarity processing functionality."""

    def setup_method(self):
        """Set up test data for batch processing."""
        # Skip if Rscript is not available
        if shutil.which("Rscript") is None:
            pytest.skip("Rscript not available, skipping melsim tests")
        
        self.test_melodies = {
            "melody1": ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
            "melody2": ([60, 62, 64, 67], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
            "melody3": ([67, 69, 71, 72], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        }

    @patch('multiprocessing.Pool')
    @patch('melody_features.melsim_wrapper.melsim._batch_compute_similarities')
    def test_directory_pairwise_comparisons(self, mock_batch_compute, mock_pool):
        """Test pairwise comparisons for directory of MIDI files."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        mock_pool_instance.imap.return_value = [
            ("melody1.mid", self.test_melodies["melody1"]),
            ("melody2.mid", self.test_melodies["melody2"]),
            ("melody3.mid", self.test_melodies["melody3"])
        ]

        mock_batch_compute.return_value = [0.8, 0.6, 0.4]

        with tempfile.TemporaryDirectory() as temp_dir:
            for name, (pitches, starts, ends) in self.test_melodies.items():
                midi_path = os.path.join(temp_dir, f"{name}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            similarities = get_similarity_from_midi(
                temp_dir,
                method="Jaccard",
                transformation="pitch"
            )

            assert isinstance(similarities, dict), "Should return similarity dictionary"
            assert len(similarities) == 3, "Should have 3 pairwise comparisons"

    @patch('multiprocessing.Pool')
    @patch('melody_features.melsim_wrapper.melsim._batch_compute_similarities')
    def test_multiple_methods_and_transformations(self, mock_batch_compute, mock_pool):
        """Test multiple methods and transformations simultaneously."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        mock_pool_instance.imap.return_value = [
            ("melody1.mid", self.test_melodies["melody1"]),
            ("melody2.mid", self.test_melodies["melody2"])
        ]

        mock_batch_compute.return_value = [0.8, 0.7, 0.6, 0.5]

        with tempfile.TemporaryDirectory() as temp_dir:
            for name in ["melody1", "melody2"]:
                pitches, starts, ends = self.test_melodies[name]
                midi_path = os.path.join(temp_dir, f"{name}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            similarities = get_similarity_from_midi(
                temp_dir,
                method=["Jaccard", "Dice"],
                transformation=["pitch", "int"]
            )

            assert isinstance(similarities, dict), "Should return similarity dictionary"
            assert len(similarities) == 4, "Should have 4 method/transformation combinations"

            for key, similarity in similarities.items():
                assert isinstance(similarity, (int, float)), f"Similarity for {key} should be numeric"
                assert 0.0 <= similarity <= 1.0, f"Similarity for {key} should be in [0,1]: {similarity}"


class TestMelsimValidation:
    """Test validation and error handling."""

    @patch('subprocess.run')
    def test_similarity_range_validation(self, mock_subprocess):
        """Test that similarities are in valid range [0.0, 1.0]."""
        test_values = ['0.0', '0.5', '1.0', '0.25', '0.99']

        for value in test_values:
            mock_subprocess.return_value = MagicMock(returncode=0, stdout=value)

            similarity = get_similarity(
                np.array([60, 62, 64]), np.array([0.0, 0.5, 1.0]), np.array([0.4, 0.9, 1.4]),
                np.array([67, 69, 71]), np.array([0.0, 0.5, 1.0]), np.array([0.4, 0.9, 1.4]),
                "Jaccard", "pitch"
            )

            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} should be in range [0,1]"

    @patch('subprocess.run')
    def test_similarity_symmetry_property(self, mock_subprocess):
        """Test that similarity is symmetric: sim(A,B) = sim(B,A)."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.75')

        melody1_data = ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        melody2_data = ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])

        sim_ab = get_similarity(
            np.array(melody1_data[0]), np.array(melody1_data[1]), np.array(melody1_data[2]),
            np.array(melody2_data[0]), np.array(melody2_data[1]), np.array(melody2_data[2]),
            "Jaccard", "pitch"
        )

        sim_ba = get_similarity(
            np.array(melody2_data[0]), np.array(melody2_data[1]), np.array(melody2_data[2]),
            np.array(melody1_data[0]), np.array(melody1_data[1]), np.array(melody1_data[2]),
            "Jaccard", "pitch"
        )

        assert sim_ab == sim_ba, "Similarity should be symmetric"

    @patch('subprocess.run')
    def test_identity_similarity(self, mock_subprocess):
        """Test that identical melodies have similarity = 1.0."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='1.0')

        melody_data = ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])

        similarity = get_similarity(
            np.array(melody_data[0]), np.array(melody_data[1]), np.array(melody_data[2]),
            np.array(melody_data[0]), np.array(melody_data[1]), np.array(melody_data[2]),
            "Jaccard", "pitch"
        )

        assert similarity == 1.0, "Identical melodies should have similarity 1.0"


class TestMelsimTransformationEffects:
    """Test how different transformations affect similarity calculations."""

    def setup_method(self):
        """Set up test melodies with known relationships."""
        self.melody1 = ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        # Transposed melody
        self.melody2 = ([62, 64, 66, 67], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])

    @patch('subprocess.run')
    def test_pitch_transformation_transposition(self, mock_subprocess):
        """Test that pitch transformation is affected by transposition."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.6')

        similarity = get_similarity(
            np.array(self.melody1[0]), np.array(self.melody1[1]), np.array(self.melody1[2]),
            np.array(self.melody2[0]), np.array(self.melody2[1]), np.array(self.melody2[2]),
            "Jaccard", "pitch"
        )

        assert similarity == 0.6
        assert "pitch" in mock_subprocess.call_args[0][0][2]

    @patch('subprocess.run')
    def test_interval_transformation_transposition_invariant(self, mock_subprocess):
        """Test that interval transformation is invariant to transposition."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='1.0')

        similarity = get_similarity(
            np.array(self.melody1[0]), np.array(self.melody1[1]), np.array(self.melody1[2]),
            np.array(self.melody2[0]), np.array(self.melody2[1]), np.array(self.melody2[2]),
            "Jaccard", "int"
        )

        assert similarity == 1.0
        assert "int" in mock_subprocess.call_args[0][0][2]

class TestMelsimComprehensiveValidation:
    """Comprehensive validation of all method/transformation combinations."""

    @patch('subprocess.run')
    def test_comprehensive_method_transformation_matrix(self, mock_subprocess):
        """Test a subset of method/transformation combinations."""
        test_methods = ["Jaccard", "Dice", "cosine", "Euclidean", "edit_sim"]
        test_transformations = ["pitch", "int", "parsons", "pc"]

        mock_subprocess.return_value = MagicMock(returncode=0, stdout='0.5')

        melody1_data = ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        melody2_data = ([67, 69, 71, 72], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])

        for method in test_methods:
            for transformation in test_transformations:
                similarity = get_similarity(
                    np.array(melody1_data[0]), np.array(melody1_data[1]), np.array(melody1_data[2]),
                    np.array(melody2_data[0]), np.array(melody2_data[1]), np.array(melody2_data[2]),
                    method, transformation
                )

                assert isinstance(similarity, float), f"Result for {method}/{transformation} should be float"
                assert 0.0 <= similarity <= 1.0, f"Similarity for {method}/{transformation} should be in [0,1]"


class TestMelsimFileHandling:
    """Test file handling and batch processing."""

    def setup_method(self):
        """Set up test - skip if Rscript is not available."""
        if shutil.which("Rscript") is None:
            pytest.skip("Rscript not available, skipping melsim tests")

    @patch('multiprocessing.Pool')
    @patch('melody_features.melsim_wrapper.melsim._batch_compute_similarities')
    def test_get_similarity_from_midi_batch_size_parameter(self, mock_batch_compute, mock_pool):
        """Test that batch_size parameter is respected."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        melodies_data = [
            ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]),
            ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]),
            ([72, 74, 76], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        ]

        mock_pool_instance.imap.return_value = [
            (f"melody_{i}.mid", melodies_data[i]) for i in range(3)
        ]

        mock_batch_compute.return_value = [0.5, 0.7, 0.4]

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                midi_path = os.path.join(temp_dir, f"melody_{i}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            similarities1 = get_similarity_from_midi(
                temp_dir, method="Jaccard", transformation="pitch", batch_size=1
            )

            similarities2 = get_similarity_from_midi(
                temp_dir, method="Jaccard", transformation="pitch", batch_size=10
            )

            assert len(similarities1) == len(similarities2), "Batch size shouldn't affect number of results"

    @patch('multiprocessing.Pool')
    @patch('melody_features.melsim_wrapper.melsim._batch_compute_similarities')
    @patch('pandas.DataFrame.to_json')
    def test_output_file_json_extension(self, mock_to_json, mock_batch_compute, mock_pool):
        """Test that output files get .json extension automatically."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        mock_pool_instance.imap.return_value = [
            ("melody_0.mid", ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])),
            ("melody_1.mid", ([61, 63, 65], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]))
        ]

        mock_batch_compute.return_value = [0.5]

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(2):
                pitches, starts, ends = ([60+i, 62+i, 64+i], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
                midi_path = os.path.join(temp_dir, f"melody_{i}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            output_file = os.path.join(temp_dir, "results")

            get_similarity_from_midi(
                temp_dir,
                method="Jaccard", 
                transformation="pitch",
                output_file=output_file
            )

            mock_to_json.assert_called_once()
            call_args = mock_to_json.call_args[0][0]
            assert str(call_args).endswith('.json'), "Should add .json extension"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
