# Changelog

All notable changes to melody-features are recorded here. Earlier versions are
described in the [GitHub release notes](https://github.com/dmwhyatt/melody-features/releases).

## [1.3.4] - 2026-11-18

This release packages the code used to extract features for the accompanying
paper. Its source tree is identical to commit `7850776` on the
`complexity-optimization` branch.

### Feature values

With IDyOM skipped, `get_all_features` returns identical values to 1.3.3: all
280 columns matched on 60 randomly sampled Essen melodies. The only change that
can alter a feature value is the IDyOM fix below.

### Fixed

- IDyOM information-content results are now matched to melodies by name rather
  than by position. Previously, files with unpadded numeric names (`m_1`, `m_2`,
  `m_10`) were read back in a different order from the one in which they were
  submitted, so IDyOM values could be assigned to the wrong melody
  ([#9](https://github.com/dmwhyatt/melody-features/pull/9)).
- `get_corpus_files` is now exported from the package root, so the documented
  `mf.get_corpus_files(...)` call works. It was previously importable only from
  `melody_features.corpus`.
- `install_idyom.sh` is now included in the wheel and sdist, so
  `melody_features.idyom.interface.install_idyom()` works after `pip install`
  ([#42](https://github.com/dmwhyatt/melody-features/issues/42)).

### Changed

- Faster complexity, MUST and m-type tokenization code, and more responsive
  progress reporting during parallel extraction. Results are collected
  as they finish and re-sorted by `melody_num`.
- `get_all_features` is annotated and documented as returning `None` when no
  valid melodies can be loaded, which it already did in 1.3.3. It returns an
  empty `DataFrame` when melodies load but no features can be extracted.

### Added

- A Sphinx documentation site with an interactive feature catalogue, API
  reference, and pages on IDyOM setup, configuration, key estimation, contours,
  corpora and melsim ([#8](https://github.com/dmwhyatt/melody-features/pull/8)).
- `CITATION.cff` and this changelog.
- A GitHub Actions workflow that publishes to PyPI when a GitHub release is
  published.

### Documentation

- Citations for Müllensiefen & Frieler / SIMILE use 2006 throughout.
- Expanded docstrings for `Config`, `FantasticConfig` and `IDyOMConfig`.

[1.3.4]: https://github.com/dmwhyatt/melody-features/compare/v1.3.3...v1.3.4
