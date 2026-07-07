"""IDyOM configuration helpers."""

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional


VALID_VIEWPOINTS = {
    "onset",
    "cpitch",
    "dur",
    "keysig",
    "mode",
    "tempo",
    "pulses",
    "barlength",
    "deltast",
    "bioi",
    "phrase",
    "mpitch",
    "accidental",
    "dyn",
    "voice",
    "ornament",
    "comma",
    "articulation",
    "ioi",
    "posinbar",
    "dur-ratio",
    "referent",
    "cpint",
    "contour",
    "cpitch-class",
    "cpcint",
    "cpintfref",
    "cpintfip",
    "cpintfiph",
    "cpintfib",
    "inscale",
    "ioi-ratio",
    "ioi-contour",
    "metaccent",
    "bioi-ratio",
    "bioi-contour",
    "lphrase",
    "cpint-size",
    "newcontour",
    "cpcint-size",
    "cpcint-2",
    "cpcint-3",
    "cpcint-4",
    "cpcint-5",
    "cpcint-6",
    "octave",
    "tessitura",
    "mpitch-class",
    "registral-direction",
    "intervallic-difference",
    "registral-return",
    "proximity",
    "closure",
    "fib",
    "crotchet",
    "tactus",
    "fiph",
    "liph",
    "thr-cpint-fib",
    "thr-cpint-fiph",
    "thr-cpint-liph",
    "thr-cpint-crotchet",
    "thr-cpint-tactus",
    "thr-cpintfref-liph",
    "thr-cpintfref-fib",
    "thr-cpint-cpintfref-liph",
    "thr-cpint-cpintfref-fib",
}


def _validate_viewpoints(viewpoints: list[str], name: str) -> None:
    """Validate that all viewpoints are valid."""
    if not isinstance(viewpoints, list):
        raise ValueError(f"{name} must be a list, got {type(viewpoints)}")

    all_viewpoints = set()
    for viewpoint in viewpoints:
        if isinstance(viewpoint, (list, tuple)):
            if len(viewpoint) < 2:
                raise ValueError(
                    f"Linked viewpoints must have at least 2 elements, got {len(viewpoint)} elements: {viewpoint}"
                )
            all_viewpoints.update(viewpoint)
        else:
            all_viewpoints.add(viewpoint)

    invalid_viewpoints = all_viewpoints - VALID_VIEWPOINTS
    if invalid_viewpoints:
        raise ValueError(
            f"Invalid viewpoint(s) in {name}: {', '.join(invalid_viewpoints)}.\n"
            f"Valid viewpoints are: {', '.join(sorted(list(VALID_VIEWPOINTS)))}"
        )


@dataclass
class IDyOMConfig:
    """Configuration class for IDyOM analysis."""

    target_viewpoints: list[str]
    source_viewpoints: list[str]
    ppm_order: int
    models: str
    corpus: Optional[os.PathLike] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        _validate_viewpoints(self.target_viewpoints, "target_viewpoints")
        _validate_viewpoints(self.source_viewpoints, "source_viewpoints")

        valid_models = {":stm", ":ltm", ":both"}
        if not isinstance(self.models, str):
            raise ValueError(f"models must be a string, got {type(self.models)}")
        if self.models not in valid_models:
            raise ValueError(f"models must be one of {valid_models}, got {self.models}")

        if self.corpus is not None:
            if self.models == ":stm":
                raise ValueError(
                    "IDyOM short-term models (:stm) do not use a corpus. "
                    "Set corpus=None for :stm configurations."
                )
            if not isinstance(self.corpus, (str, os.PathLike)):
                raise ValueError(
                    f"corpus must be a string or PathLike, got {type(self.corpus)}"
                )
            if not Path(self.corpus).exists():
                raise ValueError(f"corpus path does not exist: {self.corpus}")


_IDYOM_MEAN_INFORMATION_CONTENT_EXPORTS = frozenset({
    "pitch_stm_mean_information_content",
    "pitch_ltm_mean_information_content",
    "rhythm_stm_mean_information_content",
    "rhythm_ltm_mean_information_content",
})

_DEFAULT_CORPUS = resources.files("melody_features") / "corpora/pearce_default_idyom"


def _resolve_idyom_corpus(
    idyom_config: IDyOMConfig,
    config_corpus: Optional[os.PathLike] = None,
    override: Optional[os.PathLike] = None,
) -> Optional[os.PathLike]:
    """Resolve the pretraining corpus for an IDyOM configuration."""
    if idyom_config.models == ":stm":
        return None
    if override is not None:
        return override
    if idyom_config.corpus is not None:
        return idyom_config.corpus
    if config_corpus is not None:
        return config_corpus
    return _DEFAULT_CORPUS


def _default_idyom_configs(corpus: Optional[os.PathLike] = None) -> dict[str, IDyOMConfig]:
    """Build the standard four IDyOM configurations used by `get_all_features`."""
    return {
        "pitch_stm": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":stm",
            corpus=None,
        ),
        "pitch_ltm": IDyOMConfig(
            target_viewpoints=["cpitch"],
            source_viewpoints=[("cpitch", "cpint", "cpintfref")],
            ppm_order=None,
            models=":ltm",
            corpus=corpus,
        ),
        "rhythm_stm": IDyOMConfig(
            target_viewpoints=["onset"],
            source_viewpoints=["ioi", "ioi-ratio"],
            ppm_order=None,
            models=":stm",
            corpus=None,
        ),
        "rhythm_ltm": IDyOMConfig(
            target_viewpoints=["onset"],
            source_viewpoints=["ioi", "ioi-ratio"],
            ppm_order=None,
            models=":ltm",
            corpus=corpus,
        ),
    }


_DEFAULT_IDYOM_CONFIGS = _default_idyom_configs(_DEFAULT_CORPUS)
