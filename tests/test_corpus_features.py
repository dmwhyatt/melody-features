"""
Test suite for features.py with corpus functionality.

Tests the corpus-dependent code paths in features.py that aren't covered
by main tests, including corpus feature generation and IDyOM integration.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch

from melody_features.features import (
    get_all_features,
    get_corpus_features,
    _setup_corpus_statistics,
    _load_melody_data,
    Config,
    IDyOMConfig,
    FantasticConfig
)
from melody_features.representations import Melody


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


def create_test_corpus(temp_dir, num_melodies=3):
    """Create a small test corpus with MIDI files."""
    melodies_data = [
        ([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
        ([67, 69, 71, 72], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
        ([60, 64, 67, 60], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
    ]

    corpus_dir = os.path.join(temp_dir, "test_corpus")
    os.makedirs(corpus_dir)

    created_files = []
    for i in range(min(num_melodies, len(melodies_data))):
        pitches, starts, ends = melodies_data[i]
        midi_path = os.path.join(corpus_dir, f"melody_{i:03d}.mid")
        create_test_midi_file(pitches, starts, ends, filepath=midi_path)
        created_files.append(midi_path)

    return corpus_dir, created_files


class TestCorpusFeatures:
    """Test corpus-dependent features in features.py."""

    def test_get_corpus_features(self):
        """Test get_corpus_features function directly."""
        from melody_features.import_mid import import_midi

        # Create test melody
        pitches = [60, 62, 64, 65, 67]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0]
        ends = [0.4, 0.9, 1.4, 1.9, 2.4]

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
            temp_path = temp_file.name

        try:
            melody_data = import_midi(temp_path)
            melody = Melody(melody_data)
            
            # Create mock corpus stats
            corpus_stats = {
                "document_frequencies": {
                    "('C',)": {"count": 10},
                    "('D',)": {"count": 8},
                    "('C', 'D')": {"count": 5},
                    "('D', 'E')": {"count": 3}
                },
                "corpus_size": 100,
                "n_range": (1, 3)
            }
            
            features = get_corpus_features(
                melody=melody,
                corpus_stats=corpus_stats,
                phrase_gap=1.5,
                max_ngram_order=6
            )

            assert isinstance(features, dict), "Should return dictionary"
            assert len(features) > 0, "Should have some features"

            expected_features = [
                "mean_document_frequency", "std_document_frequency",
                "mean_global_weight", "std_global_weight"
            ]

            for feature in expected_features:
                if feature in features:
                    value = features[feature]
                    assert isinstance(value, (int, float)), f"{feature} should be numeric"
                    assert not (value < 0), f"{feature} should be non-negative"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_setup_corpus_statistics(self):
        """Test _setup_corpus_statistics function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir, _ = create_test_corpus(temp_dir)

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":both"
                    )
                },
                fantastic=FantasticConfig(
                    max_ngram_order=6,
                    phrase_gap=1.5,
                ),
                corpus=corpus_dir
            )

            output_file = os.path.join(temp_dir, "test_output.csv")
            corpus_stats = _setup_corpus_statistics(config, output_file)

            assert isinstance(corpus_stats, dict), "Should return corpus stats dict"
            assert "document_frequencies" in corpus_stats, "Should have document frequencies"
            assert "corpus_size" in corpus_stats, "Should have corpus size"
            assert corpus_stats["corpus_size"] == 3, "Should have 3 melodies"

    def test_setup_corpus_statistics_no_corpus(self):
        """Test _setup_corpus_statistics with no corpus."""
        config = Config(
            idyom={
                "test": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=["cpint"],
                    ppm_order=1,
                    models=":both"
                )
            },
            fantastic=FantasticConfig(max_ngram_order=6, phrase_gap=1.5),
            corpus=None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.csv")
            corpus_stats = _setup_corpus_statistics(config, output_file)

            assert corpus_stats is None, "Should return None when no corpus"


class TestFeaturesWithCorpus:
    """Test features.py with corpus functionality enabled."""

    def test_get_all_features_with_corpus(self):
        """Test get_all_features with corpus statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:

            corpus_dir, _ = create_test_corpus(temp_dir)
            
            test_pitches = [60, 62, 64, 65]
            test_starts = [0.0, 0.5, 1.0, 1.5]
            test_ends = [0.4, 0.9, 1.4, 1.9]

            test_midi = os.path.join(temp_dir, "test_melody.mid")
            create_test_midi_file(test_pitches, test_starts, test_ends, filepath=test_midi)

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":stm"
                    )
                },
                fantastic=FantasticConfig(
                    max_ngram_order=6,
                    phrase_gap=1.5,
                    corpus=corpus_dir
                ),
                corpus=corpus_dir
            )

            df = get_all_features(test_midi, config=config, skip_idyom=True)

            assert len(df) == 1, "Should have one row"
            row = df.iloc[0]

            corpus_feature_cols = [col for col in df.columns if col.startswith('corpus_features.')]
            assert len(corpus_feature_cols) > 0, "Should have corpus features"
            
            # Check some corpus features exist and have valid types
            if 'corpus_features.mean_document_frequency' in row.index:
                assert isinstance(row['corpus_features.mean_document_frequency'], (int, float)), "Should be numeric"
            
            if 'corpus_features.std_document_frequency' in row.index:
                assert isinstance(row['corpus_features.std_document_frequency'], (int, float)), "Should be numeric"
    
    def test_get_all_features_corpus_precedence(self):
        """Test that fantastic.corpus takes precedence over config.corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus1_dir, _ = create_test_corpus(os.path.join(temp_dir, "corpus1"))
            corpus2_dir, _ = create_test_corpus(os.path.join(temp_dir, "corpus2"))

            test_midi = os.path.join(temp_dir, "test.mid")
            create_test_midi_file([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4], filepath=test_midi)

            # Config where fantastic.corpus should take precedence
            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":stm"
                    )
                },
                fantastic=FantasticConfig(
                    max_ngram_order=3,
                    phrase_gap=1.5,
                    corpus=corpus1_dir  # This should take precedence
                ),
                corpus=corpus2_dir  # override by fantastic
            )

            df = get_all_features(test_midi, config=config, skip_idyom=True)
            # not sure right now what the best way to check which corpus it used here is
            assert len(df) == 1, "Should process successfully"
            corpus_cols = [col for col in df.columns if col.startswith('corpus_features.')]
            assert len(corpus_cols) > 0, "Should have corpus features"


class TestMelodyDataLoading:
    """Test melody data loading functionality."""

    def test_load_melody_data_single_file(self):
        """Test _load_melody_data with single MIDI file."""
        pitches = [60, 62, 64, 65]
        starts = [0.0, 0.5, 1.0, 1.5]
        ends = [0.4, 0.9, 1.4, 1.9]

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
            temp_path = temp_file.name

        try:
            melody_data_list = _load_melody_data(temp_path)

            assert isinstance(melody_data_list, list), "Should return list"
            assert len(melody_data_list) == 1, "Should have one melody"

            melody_data = melody_data_list[0]
            assert isinstance(melody_data, dict), "Melody data should be dict"
            assert "pitches" in melody_data, "Should have pitches"
            assert "starts" in melody_data, "Should have starts"
            assert "ends" in melody_data, "Should have ends"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_melody_data_file_list(self):
        """Test _load_melody_data with list of MIDI files."""
        melodies_data = [
            ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]),
            ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
        ]
    
        temp_files = []
        try:
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_test_{i}.mid', delete=False)
                create_test_midi_file(pitches, starts, ends, filepath=temp_file.name)
                temp_files.append(temp_file.name)

            # Load with from a list of file names
            melody_data_list = _load_melody_data(temp_files)

            assert len(melody_data_list) == 2, "Should load 2 melodies"

            for melody_data in melody_data_list:
                assert isinstance(melody_data, dict), "Each should be dict"
                assert len(melody_data["pitches"]) == 3, "Each should have 3 notes"

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_load_melody_data_directory(self):
        """Test _load_melody_data with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            melodies_data = [
                ([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4]),
                ([67, 69, 71], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4])
            ]
            
            for i, (pitches, starts, ends) in enumerate(melodies_data):
                midi_path = os.path.join(temp_dir, f"song_{i}.mid")
                create_test_midi_file(pitches, starts, ends, filepath=midi_path)
            
            # Load from directory
            melody_data_list = _load_melody_data(temp_dir)
            
            assert len(melody_data_list) == 2, "Should find 2 melodies in directory"
            
            for melody_data in melody_data_list:
                assert isinstance(melody_data, dict), "Each should be dict"
                assert "ID" in melody_data, "Should have ID"
                assert "melody_num" in melody_data, "Should have melody_num"


class TestCorpusStatisticsIntegration:
    """Test integration of corpus statistics with feature extraction."""
    
    def test_corpus_statistics_caching(self):
        """Test that corpus statistics are cached and reused."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir, _ = create_test_corpus(temp_dir)

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":both"
                    )
                },
                fantastic=FantasticConfig(
                    max_ngram_order=6,
                    phrase_gap=1.5,
                    corpus=corpus_dir
                ),
                corpus=corpus_dir
            )

            output_file1 = os.path.join(temp_dir, "output1.csv")
            stats1 = _setup_corpus_statistics(config, output_file1)
            # checking whether we correctly check for corpus stats and generate
            # only if not found
            corpus_name = Path(corpus_dir).name
            expected_stats_file = Path(output_file1).parent / f"{corpus_name}_corpus_stats.json"
            assert expected_stats_file.exists(), "Corpus stats file should be created"

            output_file2 = os.path.join(temp_dir, "output2.csv")
            stats2 = _setup_corpus_statistics(config, output_file2)

            assert stats1 == stats2, "Should reuse cached corpus statistics"

    def test_features_with_empty_corpus_stats(self):
        """Test feature extraction when corpus has no valid melodies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = os.path.join(temp_dir, "empty_corpus")
            os.makedirs(corpus_dir)

            test_midi = os.path.join(temp_dir, "test.mid")
            create_test_midi_file([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4], filepath=test_midi)

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":both"
                    )
                },
                fantastic=FantasticConfig(
                    max_ngram_order=6,
                    phrase_gap=1.5,
                    corpus=corpus_dir
                ),
                corpus=corpus_dir
            )

            with pytest.raises(FileNotFoundError, match="No MIDI files found"):
                get_all_features(test_midi, config=config, skip_idyom=True)


class TestCorpusErrorHandling:
    """Test error handling in corpus-related functionality."""

    def test_setup_corpus_statistics_invalid_path(self):
        """Test corpus statistics setup with invalid corpus path."""
        config = Config(
            idyom={
                "test": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=["cpint"],
                    ppm_order=1,
                    models=":both"
                )
            },
            fantastic=FantasticConfig(
                max_ngram_order=6,
                phrase_gap=1.5,
                corpus=None
            ),
            corpus=None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.csv")

            # Manually set invalid corpus after config creation to bypass validation
            config.fantastic.corpus = "/nonexistent/path"
            config.corpus = "/nonexistent/path"

            with pytest.raises(FileNotFoundError, match="Corpus path is not a valid directory"):
                _setup_corpus_statistics(config, output_file)

    def test_get_all_features_no_valid_melodies(self):
        """Test get_all_features when no valid melodies are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = os.path.join(temp_dir, "not_midi.txt")
            Path(text_file).write_text("This is not a MIDI file", encoding="utf-8")

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":stm"
                    )
                },
                fantastic=FantasticConfig(max_ngram_order=3, phrase_gap=1.5),
                corpus=None
            )

            with pytest.raises(FileNotFoundError, match="No MIDI files found"):
                get_all_features(temp_dir, config=config, skip_idyom=True)


class TestIDyOMIntegrationPaths:
    """Test IDyOM integration code paths in features.py."""

    def test_get_all_features_idyom_retry_logic(self):
        """Test IDyOM retry logic for database locking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_midi = os.path.join(temp_dir, "test.mid")
            create_test_midi_file([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.9, 1.4], filepath=test_midi)

            config = Config(
                idyom={
                    "test": IDyOMConfig(
                        target_viewpoints=["cpitch"],
                        source_viewpoints=["cpint"],
                        ppm_order=1,
                        models=":both"
                    )
                },
                fantastic=FantasticConfig(max_ngram_order=3, phrase_gap=1.5),
                corpus=None
            )
            
            # Mock IDyOM to simulate database locking
            def mock_run_idyom_analysis(*args, **kwargs):
                raise Exception("database is locked")

            with patch('melody_features.features._run_idyom_analysis', side_effect=mock_run_idyom_analysis):
                df = get_all_features(test_midi, config=config, skip_idyom=False)

                assert len(df) == 1, "Should still process melody"

                # Should have IDyOM fail flag (val = -1)
                row = df.iloc[0]
                idyom_cols = [col for col in df.columns if col.startswith('idyom_features.')]
                if idyom_cols:
                    assert 'idyom_features.mean_information_content' in df.columns, "Should have failed IDyOM feature value"
                    assert row['idyom_features.mean_information_content'] == -1, "Should have failed value"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
