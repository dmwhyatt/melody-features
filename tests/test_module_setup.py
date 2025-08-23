"""
Comprehensive test suite to verify package installation and importability.
"""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest


def test_python_environment():
    """Test basic Python environment information."""
    # Basic environment checks
    assert sys.version_info >= (
        3,
        9,
    ), f"Python version {sys.version} is too old. Need 3.9+"
    assert sys.executable is not None, "Python executable not found"
    assert len(sys.path) > 0, "Python path is empty"


def test_package_installation():
    """Test if the package is properly installed."""
    # Test if package is in Python path
    try:
        import melody_features

        assert melody_features is not None, "Package import returned None"
        assert hasattr(
            melody_features, "__file__"
        ), "Package has no __file__ attribute"
        assert Path(
            melody_features.__file__
        ).parent.exists(), "Package directory does not exist"
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")


def test_core_imports():
    """Test if core modules can be imported."""
    core_modules = [
        "melody_features.features",
        "melody_features.corpus",
        "melody_features.algorithms",
        "melody_features.complexity",
        "melody_features.distributional",
        "melody_features.idyom_interface",
        "melody_features.import_mid",
        "melody_features.interpolation_contour",
        "melody_features.melody_tokenizer",
        "melody_features.ngram_counter",
        "melody_features.narmour",
        "melody_features.representations",
        "melody_features.stats",
        "melody_features.step_contour",
        "melody_features.melsim_wrapper.melsim",
    ]

    failed_imports = []
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
            assert module is not None, f"Module {module_name} imported but is None"
        except ImportError as e:
            failed_imports.append(f"{module_name}: {e}")

    if failed_imports:
        pytest.fail(f"Failed to import modules: {', '.join(failed_imports)}")


def test_main_functions():
    """Test if main functions can be imported and called."""
    from melody_features.features import (
        Config,
        FantasticConfig,
        IDyOMConfig,
        get_all_features,
    )

    config = Config(
        idyom={
            "pitch": IDyOMConfig(
                target_viewpoints=["cpitch"],
                source_viewpoints=[("cpint", "cpintfref")],
                ppm_order=1,
                models=":both",
                corpus=None,
            )
        },
        fantastic=FantasticConfig(max_ngram_order=2, phrase_gap=1.5, corpus=None),
    )

    assert config is not None, "Config creation returned None"
    assert hasattr(config, "idyom"), "Config missing idyom attribute"
    assert hasattr(config, "fantastic"), "Config missing fantastic attribute"
    assert config.corpus is None, "Config corpus should be None"
    assert callable(get_all_features), "get_all_features should be callable"


def test_dependencies():
    """Test if key dependencies are available."""
    dependencies = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "mido",
        "pretty_midi",
        "natsort",
        "tqdm",
        "pathlib",
        "importlib.resources",
    ]

    failed_deps = []
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            assert module is not None, f"Module {dep} imported but is None"
        except ImportError as e:
            failed_deps.append(f"{dep}: {e}")

    if failed_deps:
        pytest.fail(f"Failed to import dependencies: {', '.join(failed_deps)}")


def test_file_system_access():
    """Test file system access and permissions."""
    # Test temp directory creation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        test_file.write_text("test")
        assert test_file.exists(), "Failed to create test file in temp directory"
        assert test_file.read_text() == "test", "Failed to read test file content"

    # Test current directory write access
    test_file = Path.cwd() / "test_write_access.txt"
    test_file.write_text("test")
    assert test_file.exists(), "Failed to create test file in current directory"
    test_file.unlink()  # Clean up
    assert not test_file.exists(), "Failed to delete test file"


def test_resource_access():
    """Test importlib.resources access."""
    from importlib import resources

    essen_path = resources.files("melody_features") / "corpora" / "Essen_Corpus"

    assert essen_path is not None, "Resource path is None"
    assert essen_path.exists(), f"Resource path does not exist: {essen_path}"

    midi_files = list(essen_path.glob("*.mid"))
    assert len(midi_files) > 0, f"No MIDI files found in resource path: {essen_path}"


def test_importlib_resources_compatibility():
    """Test that importlib.resources works correctly for the package."""
    from importlib import resources

    # Test that we can access package resources
    try:
        package_files = resources.files("melody_features")
        assert package_files.exists(), "Package files path does not exist"

        # Test that corpora directory exists
        corpora_path = package_files / "corpora"
        assert corpora_path.exists(), "Corpora directory does not exist"

        # Test that Essen_Corpus exists
        essen_path = corpora_path / "Essen_Corpus"
        assert essen_path.exists(), "Essen_Corpus directory does not exist"

    except Exception as e:
        pytest.fail(f"importlib.resources access failed: {e}")


def test_environment_consistency():
    """Test that the environment is consistent across different import methods."""
    # Test that importing from different paths gives consistent results
    from melody_features import essen_corpus as essen_from_main
    from melody_features.corpus import essen_corpus as essen_from_corpus

    assert (
        essen_from_corpus == essen_from_main
    ), "essen_corpus inconsistent between import methods"

    # Test that get_corpus_path gives consistent results
    from melody_features.corpus import get_corpus_path

    essen_from_function = get_corpus_path("essen")

    assert (
        essen_from_function == essen_from_corpus
    ), "get_corpus_path inconsistent with direct import"
