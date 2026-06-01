"""Tests for IDyOM consistency when invoked separately from `get_all_features()`."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from melody_features.features import (
    IDyOMConfig,
    _DEFAULT_CORPUS,
    _DEFAULT_IDYOM_CONFIGS,
    _default_idyom_configs,
    _idyom_mean_information_content,
    _resolve_idyom_corpus,
    _run_idyom_analysis,
    _setup_default_config,
    pitch_ltm_mean_information_content,
    pitch_stm_mean_information_content,
)
from melody_features.representations import Melody

SAMPLE_MELODY = Melody(
    {
        "ID": "test",
        "MIDI Sequence": (
            "Note(start=0.0, end=0.5, pitch=60, velocity=80)"
            "Note(start=0.5, end=1.0, pitch=62, velocity=80)"
        ),
    }
)


class TestDefaultIdyomCorpus:
    def test_default_corpus_path_exists(self):
        assert Path(_DEFAULT_CORPUS).exists()
        assert Path(_DEFAULT_CORPUS).is_dir()
        assert "pearce_default_idyom" in str(_DEFAULT_CORPUS)

    def test_default_idyom_configs_set_ltm_corpus(self):
        configs = _default_idyom_configs(_DEFAULT_CORPUS)
        assert configs["pitch_ltm"].corpus == _DEFAULT_CORPUS
        assert configs["rhythm_ltm"].corpus == _DEFAULT_CORPUS
        assert configs["pitch_stm"].corpus is None
        assert configs["rhythm_stm"].corpus is None

    def test_default_idyom_configs_none_leaves_ltm_unset(self):
        configs = _default_idyom_configs(None)
        assert configs["pitch_ltm"].corpus is None
        assert configs["rhythm_ltm"].corpus is None

    def test_module_level_defaults_match_config_corpus(self):
        assert _DEFAULT_IDYOM_CONFIGS["pitch_ltm"].corpus == _DEFAULT_CORPUS
        assert _DEFAULT_IDYOM_CONFIGS["rhythm_ltm"].corpus == _DEFAULT_CORPUS

    def test_setup_default_config_aligns_corpus_and_idyom(self):
        config = _setup_default_config(None)
        assert config.corpus == _DEFAULT_CORPUS
        assert config.idyom["pitch_ltm"].corpus == _DEFAULT_CORPUS
        assert config.idyom["rhythm_ltm"].corpus == _DEFAULT_CORPUS
        assert config.idyom["pitch_stm"].corpus is None
        assert config.idyom["rhythm_stm"].corpus is None


@pytest.fixture
def alternate_corpus():
    """Second bundled corpus when available (for precedence tests)."""
    from melody_features.corpus import essen_corpus

    if not essen_corpus.exists():
        pytest.skip("essen_folksong_collection not bundled in this environment")
    return essen_corpus


class TestResolveIdyomCorpus:
    @pytest.fixture
    def ltm_config(self, alternate_corpus):
        return IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=alternate_corpus,
        )

    @pytest.fixture
    def stm_config(self):
        return IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":stm",
            corpus=None,
        )

    def test_stm_never_uses_corpus(self, stm_config):
        assert _resolve_idyom_corpus(stm_config, config_corpus="/config/corpus") is None
        assert (
            _resolve_idyom_corpus(
                stm_config,
                config_corpus="/config/corpus",
                override="/override/corpus",
            )
            is None
        )

    def test_ltm_prefers_per_config_corpus(self, ltm_config, alternate_corpus):
        assert _resolve_idyom_corpus(ltm_config, config_corpus=_DEFAULT_CORPUS) == (
            alternate_corpus
        )

    def test_ltm_falls_back_to_config_corpus(self, alternate_corpus):
        config = IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=None,
        )
        assert _resolve_idyom_corpus(config, config_corpus=alternate_corpus) == (
            alternate_corpus
        )

    def test_ltm_falls_back_to_package_default(self):
        config = IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=None,
        )
        assert _resolve_idyom_corpus(config) == _DEFAULT_CORPUS

    def test_override_wins_for_ltm(self, alternate_corpus):
        ltm_config = IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=_DEFAULT_CORPUS,
        )
        assert _resolve_idyom_corpus(
            ltm_config,
            config_corpus=_DEFAULT_CORPUS,
            override=alternate_corpus,
        ) == alternate_corpus


class TestIdyomMeanInformationContentCorpus:
    @patch("melody_features.features.get_idyom_results")
    def test_ltm_wrapper_passes_default_corpus(self, mock_get_idyom_results):
        mock_get_idyom_results.return_value = {"1": {"mean_information_content": 1.25}}

        result = pitch_ltm_mean_information_content(SAMPLE_MELODY)

        assert result == 1.25
        mock_get_idyom_results.assert_called_once()
        assert mock_get_idyom_results.call_args.args[5] == _DEFAULT_CORPUS

    @patch("melody_features.features.get_idyom_results")
    def test_stm_wrapper_passes_no_corpus(self, mock_get_idyom_results):
        mock_get_idyom_results.return_value = {"1": {"mean_information_content": 0.75}}

        result = pitch_stm_mean_information_content(SAMPLE_MELODY)

        assert result == 0.75
        mock_get_idyom_results.assert_called_once()
        assert mock_get_idyom_results.call_args.args[5] is None

    @patch("melody_features.features.get_idyom_results")
    def test_explicit_corpus_override(self, mock_get_idyom_results):
        mock_get_idyom_results.return_value = {"1": {"mean_information_content": 2.0}}
        custom_corpus = Path(_DEFAULT_CORPUS).parent / "essen_folksong_collection"
        if not custom_corpus.exists():
            pytest.skip("essen_folksong_collection not bundled in this environment")

        result = _idyom_mean_information_content(
            SAMPLE_MELODY, "pitch_ltm", corpus=custom_corpus
        )

        assert result == 2.0
        assert mock_get_idyom_results.call_args.args[5] == custom_corpus


class TestRunIdyomAnalysisCorpus:
    @patch("melody_features.features.get_idyom_results")
    def test_batch_run_uses_config_corpus_for_ltm(
        self, mock_get_idyom_results, alternate_corpus
    ):
        from melody_features.features import Config, FantasticConfig

        mock_get_idyom_results.return_value = {}
        config = Config(
            corpus=alternate_corpus,
            idyom=_default_idyom_configs(None),
            fantastic=FantasticConfig(max_ngram_order=2, phrase_gap=1.5, corpus=None),
        )

        with tempfile_test_midi_dir() as midi_dir:
            _run_idyom_analysis(midi_dir, config)

        ltm_calls = [
            call
            for call in mock_get_idyom_results.call_args_list
            if call.args[3] == ":ltm"
        ]
        assert len(ltm_calls) == 2
        assert all(call.args[5] == alternate_corpus for call in ltm_calls)

        stm_calls = [
            call
            for call in mock_get_idyom_results.call_args_list
            if call.args[3] == ":stm"
        ]
        assert len(stm_calls) == 2
        assert all(call.args[5] is None for call in stm_calls)


@contextmanager
def tempfile_test_midi_dir():
    """Minimal directory path for _run_idyom_analysis (calls are mocked)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = os.path.join(temp_dir, "test.mid")
        with open(midi_path, "wb") as midi_file:
            midi_file.write(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x00")
        yield temp_dir
