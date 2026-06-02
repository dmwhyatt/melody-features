import numpy as np

from melody_features.features import gradus


def test_gradus_returns_float_without_truncation():
    # Semitone then tritone gives gradus values 11 and 14, mean should be 12.5
    value = gradus([60, 61, 67])
    assert isinstance(value, float)
    assert np.isclose(value, 12.5)
