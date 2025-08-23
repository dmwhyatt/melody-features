#!/usr/bin/env python3
"""
Pytest tests to verify corpus path imports work correctly.
"""

from pathlib import Path

import pytest


def test_essen_corpus_import():
    """Test that essen_corpus can be imported and points to a valid path."""
    from melody_features.corpus import essen_corpus

    assert isinstance(essen_corpus, Path)
    assert essen_corpus.exists()
    assert essen_corpus.is_dir()
    assert "Essen_Corpus" in str(essen_corpus)


def test_get_corpus_path_function():
    """Test the get_corpus_path convenience function."""
    from melody_features.corpus import get_corpus_path

    essen_path = get_corpus_path("essen")
    assert isinstance(essen_path, Path)
    assert essen_path.exists()
    assert essen_path.is_dir()
    assert "Essen_Corpus" in str(essen_path)


def test_main_package_import():
    """Test that essen_corpus can be imported from the main package."""
    from melody_features import essen_corpus

    assert isinstance(essen_corpus, Path)
    assert essen_corpus.exists()
    assert essen_corpus.is_dir()


def test_list_available_corpora():
    """Test the list_available_corpora function."""
    from melody_features import list_available_corpora

    corpora = list_available_corpora()
    assert isinstance(corpora, list)
    assert "essen" in corpora
    assert len(corpora) >= 1


def test_invalid_corpus_name():
    """Test that get_corpus_path raises ValueError for invalid corpus names."""
    from melody_features.corpus import get_corpus_path

    with pytest.raises(ValueError):
        get_corpus_path("nonexistent_corpus")


def test_corpus_paths_consistency():
    """Test that different import methods return the same path."""
    from melody_features import essen_corpus as essen_from_main
    from melody_features.corpus import essen_corpus, get_corpus_path

    path1 = essen_corpus
    path2 = get_corpus_path("essen")
    path3 = essen_from_main

    assert path1 == path2 == path3
