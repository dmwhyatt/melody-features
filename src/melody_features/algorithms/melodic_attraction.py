"""MIDI Toolbox `melattraction.m` implementation."""

from __future__ import annotations

import numpy as np

_KK_MAJ_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_KK_MIN_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)


def _kkcc_from_pcd(pcd: np.ndarray) -> np.ndarray:
    majors = np.array([np.roll(_KK_MAJ_PROFILE, k) for k in range(12)])
    minors = np.array([np.roll(_KK_MIN_PROFILE, k) for k in range(12)])
    profiles = np.vstack([majors, minors])
    pcd = np.asarray(pcd, dtype=float).ravel()
    matrix = np.vstack([pcd, profiles]).T
    cm = np.corrcoef(matrix, rowvar=False)
    return cm[0, 1:]


def _stability_distance(weight1: float, weight2: float, proximity: float) -> float:
    if weight1 == 0 or proximity == 0:
        return 0.0
    return (weight2 / weight1) * (1.0 / (proximity**2))


def _kkcc_from_nmat(
    pitches: list[int], starts: list[float], ends: list[float]
) -> np.ndarray:
    from ..feature_definitions.pitch_class import _pcdist1_vector

    pcd = _pcdist1_vector(pitches, starts, ends)
    return _kkcc_from_pcd(pcd)


def _transpose_to_c(
    pitches: list[int], starts: list[float], ends: list[float]
) -> list[int]:
    """`transpose2c.m`: transpose so KKCC argmax key is tonic C."""
    kk = _kkcc_from_nmat(pitches, starts, ends)
    key_index = int(np.argmax(kk)) + 1  # MATLAB 1..24
    return [int(pitch) - key_index + 1 for pitch in pitches]


def _keymode_from_kkcc(kk: np.ndarray) -> int:
    """`keymode.m`: 1 major, 2 minor."""
    if kk[0] > kk[12]:
        return 1
    if kk[0] < kk[12]:
        return 2
    return 0


def _directed_motion_index(pitches: list[int]) -> list[float]:
    """Directed motion index from `melattraction.m`."""
    if len(pitches) < 2:
        return [0.0] * len(pitches)

    pitch_arr = np.asarray(pitches, dtype=float)
    d = np.sign(np.diff(pitch_arr))
    inner = (1 - np.abs(np.sign(np.diff(np.sign(np.sign(np.diff(pitch_arr))))))) * 2 - 1
    motion = np.concatenate([[0.0], inner])
    motion[d == 0] = 0.0
    return [float(value) for value in motion]


def melodic_attraction_vector(
    pitches: list[int], starts: list[float], ends: list[float]
) -> list[float]:
    """Melodic attraction for each note (MIDI Toolbox `melattraction.m`)."""
    if not pitches:
        return []
    if len(pitches) == 1:
        return [0.0]

    transposed = _transpose_to_c(pitches, starts, ends)
    kk = _kkcc_from_nmat(transposed, starts, ends)
    mode = _keymode_from_kkcc(kk)

    if mode == 1:
        anchor_weights = [4, 1, 2, 1, 3, 2, 1, 3, 1, 2, 1, 2]
    else:
        anchor_weights = [4, 1, 2, 3, 1, 2, 1, 3, 2, 2, 1, 2]

    pc0 = [(int(pitch) % 12) + 1 for pitch in transposed]
    pc_aw = [anchor_weights[pc - 1] for pc in pc0]
    pc = transposed
    motion = _directed_motion_index(transposed)
    prox = [0.0] + [abs(transposed[i + 1] - transposed[i]) for i in range(len(transposed) - 1)]

    dd1: list[float] = []
    # MATLAB `sd2b` is never cleared between loop iterations; `length(sd2b)`
    # keeps stale entries from earlier steps (`melattraction.m` lines 118-132).
    sd2b: list[float] = [0.0] * 9
    max_sd2b_len = 0
    j = 0
    for i in range(len(pc_aw) - 1):
        if pc_aw[i] > pc_aw[i + 1]:
            sd1 = 0.0
        elif pc_aw[i] == pc_aw[i + 1]:
            sd1 = 0.0
        else:
            sd1 = _stability_distance(pc_aw[i], pc_aw[i + 1], prox[i + 1])

        neighbor0 = pc0[i] - 1
        neighbor1 = [(neighbor0 + offset) - 5 for offset in range(1, 10)]
        neighbor2 = [int(np.mod(value, 12) + 1) for value in neighbor1]
        neighbor3 = [anchor_weights[n - 1] for n in neighbor2]

        eliminate_correct_cont = pc[i + 1] - pc[i] + 5
        if eliminate_correct_cont > 8 or eliminate_correct_cont < 1:
            eliminate_correct_cont = None

        n1 = [1 if pc_aw[0] < weight else 0 for weight in neighbor3]
        n1[4] = 0

        if sum(n1) == 0:
            sd2 = 0.0
        else:
            if eliminate_correct_cont is not None:
                active = [index + 1 for index, flag in enumerate(n1) if flag]
                if not (len(active) == 1 and active[0] == eliminate_correct_cont):
                    n1[eliminate_correct_cont - 1] = 0
                    n1[4] = 0

            more_stable = [weight if flag else 0.0 for flag, weight in zip(n1, neighbor3)]
            candidate_index = 0
            for rel_index, weight in enumerate(more_stable, start=1):
                if weight <= 0.0:
                    continue
                distance = abs(rel_index - 5)
                candidate_index += 1
                j = candidate_index
                sd2b[j - 1] = _stability_distance(
                    pc_aw[i], weight, float(distance)
                )

            max_sd2b_len = max(max_sd2b_len, j)
            if max_sd2b_len > 1:
                values = sd2b[:max_sd2b_len]
                max_sd2 = max(values)
                sd2 = (
                    sum((0.5 if value != max_sd2 else 0.0) * value for value in values)
                    + max_sd2
                )
            else:
                sd2 = sd2b[j - 1] if j > 0 else 0.0

        anchoring = sd1 - sd2
        dd1.append(motion[i] + anchoring)

    scaled = [(value + 1.0) / 5.0 for value in dd1]
    scaled = [max(0.0, min(1.0, value)) for value in scaled]
    return [0.0] + scaled
