"""
Test suite for corpus.py functionality.

Tests corpus statistics generation, n-gram computation, and file loading
functionality that's used by features.py but not covered in other main tests.
"""

import pytest
import tempfile
import os
from pathlib import Path

from melody_features.corpus import (
    compute_corpus_ngrams,
    save_corpus_stats,
    load_corpus_stats,
    make_corpus_stats,
    load_midi_melody,
    load_melodies_from_directory,
    process_melody_ngrams,
    get_corpus_path,
    get_corpus_files,
    list_available_corpora
)
from melody_features.representations import Melody


def create_test_midi_file(pitches, starts, ends, tempo=120, filepath=None):
    """Create a temporary MIDI file for testing."""
    import mido

    # Create a new MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

    ticks_per_second = 480 * (tempo / 60)

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


class TestNgramProcessing:
    """Test n-gram processing functionality."""

    def setup_method(self):
        """Set up test data."""
        from melody_features.import_mid import import_midi
        self.import_midi = import_midi

    def test_process_melody_ngrams(self):
        """Test n-gram processing for a single melody."""
        # Create test melody
        pitches = [60, 62, 64, 65, 67]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0]
        ends = [0.4, 0.9, 1.4, 1.9, 2.4]

        # Create temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
            temp_path = temp_file.name

        try:
            melody_data = self.import_midi(temp_path)
            melody = Melody(melody_data)

            ngrams = process_melody_ngrams((melody, (1, 3)))

            assert isinstance(ngrams, set), "Should return set of n-grams"
            assert len(ngrams) > 0, "Should find some n-grams"

            unigrams = [ng for ng in ngrams if len(ng) == 1]
            bigrams = [ng for ng in ngrams if len(ng) == 2]
            trigrams = [ng for ng in ngrams if len(ng) == 3]

            assert len(unigrams) > 0, "Should have unigrams"
            assert len(bigrams) > 0, "Should have bigrams"
            assert len(trigrams) > 0, "Should have trigrams"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_compute_corpus_ngrams(self):
        """Test corpus-wide n-gram computation."""
        # Create two test melodies
        melodies_data = [
            ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
            ([67, 69, 71, 72], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        ]

        melodies = []
        temp_files = []

        try:
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_test_{i}.mid', delete=False)
                create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
                temp_files.append(temp_file.name)

                melody_data = self.import_midi(temp_file.name)
                melody = Melody(melody_data)
                melodies.append(melody)

            corpus_stats = compute_corpus_ngrams(melodies, n_range=(1, 2))

            assert isinstance(corpus_stats, dict), "Should return dictionary"
            assert "document_frequencies" in corpus_stats, "Should have document frequencies"
            assert "corpus_size" in corpus_stats, "Should have corpus size"
            assert "n_range" in corpus_stats, "Should have n-gram range"

            assert corpus_stats["corpus_size"] == 2, "Should have 2 melodies"
            assert corpus_stats["n_range"] == (1, 2), "Should have correct n-gram range"

            doc_freqs = corpus_stats["document_frequencies"]
            assert isinstance(doc_freqs, dict), "Document frequencies should be dict"
            assert len(doc_freqs) > 0, "Should have some n-grams"

            for ngram_str, info in doc_freqs.items():
                assert isinstance(info, dict), f"N-gram info should be dict: {ngram_str}"
                assert "count" in info, f"N-gram should have count: {ngram_str}"
                assert isinstance(info["count"], int), f"Count should be integer: {ngram_str}"
                assert info["count"] > 0, f"Count should be positive: {ngram_str}"

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestCorpusStatsPersistence:
    """Test corpus statistics saving and loading."""

    def test_save_and_load_corpus_stats(self):
        """Test saving and loading corpus statistics."""
        # Create some mock test stats
        test_stats = {
            "document_frequencies": {
                "('0', '1')": {"count": 5},
                "('1', '2')": {"count": 3},
                "('2', '3')": {"count": 2}
            },
            "corpus_size": 10,
            "n_range": (1, 3)
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file = os.path.join(temp_dir, "test_stats")

            # Test saving (should add .json extension)
            save_corpus_stats(test_stats, stats_file)

            # Check file exists with .json extension
            json_file = stats_file + ".json"
            assert os.path.exists(json_file), "JSON file should be created"

            # Test loading
            loaded_stats = load_corpus_stats(stats_file)  # Should work without .json

            # JSON converts tuples to lists, so check individual components
            assert loaded_stats["document_frequencies"] == test_stats["document_frequencies"]
            assert loaded_stats["corpus_size"] == test_stats["corpus_size"]
            assert loaded_stats["n_range"] == list(test_stats["n_range"]), "JSON converts tuple to list"

            # Test loading with .json extension
            loaded_stats2 = load_corpus_stats(json_file)
            assert loaded_stats2["document_frequencies"] == test_stats["document_frequencies"]
            assert loaded_stats2["corpus_size"] == test_stats["corpus_size"]
            assert loaded_stats2["n_range"] == list(test_stats["n_range"]), "JSON converts tuple to list"

    def test_save_corpus_stats_creates_directory(self):
        """Test that save_corpus_stats creates necessary directories."""
        test_stats = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "path", "stats")

            # Should create nested directories
            save_corpus_stats(test_stats, nested_path)

            assert os.path.exists(nested_path + ".json"), "Should create nested directories and file"

            loaded = load_corpus_stats(nested_path)
            assert loaded == test_stats, "Should load correctly from nested path"

    def test_load_corpus_stats_nonexistent_file(self):
        """Test loading stats from non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = os.path.join(temp_dir, "nonexistent.json")

            with pytest.raises(FileNotFoundError):
                load_corpus_stats(nonexistent_file)


class TestMelodyLoading:
    """Test melody loading functionality."""

    def test_load_midi_melody(self):
        """Test loading melody from MIDI file."""
        pitches = [60, 62, 64, 65]
        starts = [0.0, 0.5, 1.0, 1.5]
        ends = [0.4, 0.9, 1.4, 1.9]

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
            temp_path = temp_file.name

        try:
            melody = load_midi_melody(temp_path)

            assert isinstance(melody, Melody), "Should return Melody object"
            assert len(melody.pitches) == 4, "Should have 4 notes"
            assert melody.pitches == pitches, "Should preserve pitches"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_midi_melody_invalid_file(self):
        """Test loading melody from invalid MIDI file."""
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            # Write an invalid MIDI data
            temp_file.write(b"not a midi file")
            temp_path = temp_file.name

        try:
            melody = load_midi_melody(temp_path)
            assert melody is None, "Should return None for invalid MIDI file"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_melodies_from_directory_midi(self):
        """Test loading melodies from directory of MIDI files."""
        melodies_data = [
            ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]),
            ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                midi_path = os.path.join(temp_dir, f"melody_{i}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            melodies = load_melodies_from_directory(temp_dir, file_type="midi")

            assert isinstance(melodies, list), "Should return list"
            assert len(melodies) == 2, "Should load 2 melodies"

            for melody in melodies:
                assert isinstance(melody, Melody), "Each item should be Melody object"
                assert len(melody.pitches) == 3, "Each melody should have 3 notes"

    def test_load_melodies_from_directory_empty(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="No MIDI files found"):
                load_melodies_from_directory(temp_dir, file_type="midi")

    def test_load_melodies_from_directory_invalid_type(self):
        """Test loading with invalid file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="file_type must be either 'json' or 'midi'"):
                load_melodies_from_directory(temp_dir, file_type="invalid")

    def test_load_melodies_from_directory_invalid_directory(self):
        """Test loading from non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = os.path.join(temp_dir, "nonexistent")

            with pytest.raises(FileNotFoundError, match="Directory not found"):
                load_melodies_from_directory(nonexistent_dir, file_type="midi")


class TestCorpusGeneration:
    """Test corpus statistics generation and corpus access."""

    def test_make_corpus_stats(self):
        """Test complete corpus statistics generation from MIDI directory."""
        melodies_data = [
            ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
            ([67, 69, 71, 72], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
            ([60, 64, 67, 60], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9])
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                midi_path = os.path.join(temp_dir, f"corpus_melody_{i}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            stats_file = os.path.join(temp_dir, "corpus_stats.json")
            make_corpus_stats(temp_dir, stats_file)

            assert os.path.exists(stats_file), "Stats file should be created"

            stats = load_corpus_stats(stats_file)

            assert isinstance(stats, dict), "Stats should be dictionary"
            assert stats["corpus_size"] == 3, "Should have 3 melodies"
            assert "document_frequencies" in stats, "Should have document frequencies"
            assert "n_range" in stats, "Should have n-gram range"

            doc_freqs = stats["document_frequencies"]
            assert len(doc_freqs) > 0, "Should have some n-grams"

            for ngram_str, info in doc_freqs.items():
                assert "count" in info, f"N-gram should have count: {ngram_str}"
                assert 1 <= info["count"] <= 3, f"Count should be between 1-3: {ngram_str}"

    def test_make_corpus_stats_empty_directory(self):
        """Test corpus stats generation with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = os.path.join(temp_dir, "empty")
            os.makedirs(empty_dir)

            stats_file = os.path.join(temp_dir, "stats")

            with pytest.raises(FileNotFoundError, match="No MIDI files found"):
                make_corpus_stats(empty_dir, stats_file)

    def test_list_available_corpora(self):
        """Test listing available corpora."""
        corpora = list_available_corpora()

        assert isinstance(corpora, list), "Should return list"
        assert "essen" in corpora, "Should include Essen corpus"
        assert len(corpora) >= 1, "Should have at least one corpus"

    def test_get_corpus_path_essen(self):
        """Test getting Essen corpus path."""
        try:
            corpus_path = get_corpus_path("essen")

            assert isinstance(corpus_path, Path), "Should return Path object"
            # corpus might not necessarily exist in test environment, that's okay

        except FileNotFoundError:
            pytest.skip("Essen corpus not available in test environment")

    def test_get_corpus_path_invalid(self):
        """Test getting invalid corpus path."""
        with pytest.raises(ValueError, match="Unknown corpus 'invalid'"):
            get_corpus_path("invalid")

    def test_get_corpus_files_essen(self):
        """Test getting files from Essen corpus."""
        try:
            corpus_files = get_corpus_files("essen", max_files=5)

            assert isinstance(corpus_files, list), "Should return list"
            assert len(corpus_files) <= 5, "Should respect max_files limit"

            for file_path in corpus_files:
                assert isinstance(file_path, Path), "Each item should be Path object"
                assert file_path.suffix in ['.mid', '.midi'], "Should be MIDI files"

        except FileNotFoundError:
            pytest.skip("Essen corpus not available in test environment")

    def test_get_corpus_files_invalid(self):
        """Test getting files from invalid corpus."""
        with pytest.raises(ValueError, match="Unknown corpus 'invalid'"):
            get_corpus_files("invalid")

class TestMidiToCorpusStatsWorkflow:
    """Test complete end-to-end corpus workflows."""

    def test_end_to_end_corpus_workflow(self):
        """Test complete workflow from MIDI files to corpus statistics."""
        melodies_data = [
            ([60, 62, 64], [0.0, 1.0, 2.0], [0.8, 1.8, 2.8]),
            ([64, 62, 60], [0.0, 1.0, 2.0], [0.8, 1.8, 2.8]),
            ([60, 64, 67], [0.0, 1.0, 2.0], [0.8, 1.8, 2.8])
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = os.path.join(temp_dir, "test_corpus")
            os.makedirs(corpus_dir)

            for i, (pitches, starts, ends) in enumerate(melodies_data):
                midi_path = os.path.join(corpus_dir, f"song_{i:03d}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)

            stats_file = os.path.join(temp_dir, "test_corpus_stats")
            make_corpus_stats(corpus_dir, stats_file)

            stats = load_corpus_stats(stats_file)

            assert stats["corpus_size"] == 3, "Should process 3 melodies"
            assert isinstance(stats["document_frequencies"], dict), "Should have n-gram frequencies"

            doc_freqs = stats["document_frequencies"]
            assert len(doc_freqs) > 0, "Should find n-grams in corpus"

            counts = [info["count"] for info in doc_freqs.values()]
            max_count = max(counts)
            assert max_count <= 3, "No n-gram should appear more than 3 times"
            assert max_count >= 1, "All n-grams should appear at least once"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
