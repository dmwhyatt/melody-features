import inspect
import melody_features.features as features_module


def test_collect_labelled_functions_and_classes():
    labelled_functions = []
    labelled_classes = []

    for name, obj in inspect.getmembers(features_module):
        if inspect.isfunction(obj) and hasattr(obj, "_feature_source"):
            labelled_functions.append(name)

        if (inspect.isclass(obj) or (hasattr(obj, "__call__") and hasattr(obj, "__name__"))) and hasattr(obj, "_feature_source"):
            labelled_classes.append(name)

    assert len(labelled_functions) > 0, "Expected at least one labelled function"
    assert len(labelled_classes) > 0, "Expected at least one labelled class"

    assert "pitch_range" in labelled_functions, "Expected 'pitch_range' to be labelled"
    assert "InverseEntropyWeighting" in labelled_classes, "Expected class to be labelled"

    for fname in labelled_functions:
        func = getattr(features_module, fname)
        assert isinstance(getattr(func, "_feature_source", None), str)

    for cname in labelled_classes:
        cls = getattr(features_module, cname)
        assert isinstance(getattr(cls, "_feature_source", None), str)
    
    all_labelled_items = labelled_functions + labelled_classes
    for name in all_labelled_items:
        obj = getattr(features_module, name)
        assert hasattr(obj, "_feature_citation"), f"Expected {name} to have _feature_citation"
        assert obj._feature_citation is not None, f"Expected {name} to have a non-None citation"
        assert len(obj._feature_citation.strip()) > 0, f"Expected {name} to have a non-empty citation"

