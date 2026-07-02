"""MIDI Toolbox ``mobility.m`` implementation."""

from __future__ import annotations

import math

import numpy as np


def mobility_vector(pitches: list[int]) -> list[float]:
    """Melodic mobility for each note (MIDI Toolbox ``mobility.m``)."""
    if not pitches:
        return []
    if len(pitches) == 1:
        return [0.0]

    p = np.asarray(pitches, dtype=float)
    n = len(p)
    p_hist = np.zeros(n + 1)
    p2 = np.zeros(n + 1)
    mob = np.zeros(n + 1)
    y = np.zeros(n - 1)

    for i in range(2, n + 1):
        m_im1 = float(np.mean(p[: i - 1]))
        p_hist[i - 1] = p[i - 2] - m_im1
        p2[i] = p[i - 2] - m_im1

        z = np.concatenate([p_hist[1:i], [p_hist[i - 1]]])
        p2_row = p2[1 : len(z) + 1]
        p3 = np.column_stack([p2_row, z])

        if p3.shape[0] >= 2 and np.std(p3[:, 0]) > 0 and np.std(p3[:, 1]) > 0:
            coeff = np.corrcoef(p3, rowvar=False)[0, 1]
            mob[i] = float(coeff) if not np.isnan(coeff) else float("nan")
        else:
            mob[i] = float("nan")

        y[i - 2] = mob[i - 1] * (p[i - 1] - m_im1)

    if len(y) >= 2:
        y[1] = 0.0
    result = np.abs(np.concatenate([[0.0], y]))
    if len(result) > n:
        result = result[:n]
    if len(result) > 3 and math.isnan(result[3]):
        return []
    return [float(value) for value in result]
