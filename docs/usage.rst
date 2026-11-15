Usage
=====

This guide documents the main workflows with explicit return types, so you
do not have to guess whether a helper returns a single object, a list, or
``None``.

For the interactive feature table, see :doc:`feature_catalogue`.
For generated API docs, see :doc:`api/index`.

Examples below use ``import melody_features as mf``. Helpers that are not
re-exported on the package root (for example ``load_midi`` and ``Melody``)
still use submodule imports.


Loading melodies
----------------

The package represents one monophonic melody as a
:class:`~melody_features.core.representations.Melody` object. Load a file once into a
``Melody``, then pass its attributes (``melody.pitches``, ``melody.starts``,
…) into feature functions — you should not need to unpack the MIDI yourself.

Single MIDI file
~~~~~~~~~~~~~~~~

.. code-block:: python

   from melody_features.io.midi import load_midi
   from melody_features.core.representations import Melody

   melody = load_midi("example.mid")
   # Returns: Optional[Melody]
   #   - Melody on success
   #   - None if the file cannot be imported or is empty of notes

   assert melody is None or isinstance(melody, Melody)

:func:`~melody_features.io.midi.load_midi` always returns one object (or
``None``). It never returns a list.

Useful attributes on a successful load:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Attribute
     - Type
     - Meaning
   * - ``melody.id``
     - ``str``
     - Source path / identifier
   * - ``melody.pitches``
     - ``list[int]``
     - MIDI pitch numbers in note order
   * - ``melody.starts``
     - ``list[float]``
     - Note onset times (seconds)
   * - ``melody.ends``
     - ``list[float]``
     - Note offset times (seconds)
   * - ``melody.tempo``
     - ``float``
     - Tempo in BPM
   * - ``melody.tempo_changes``
     - ``list[tuple[float, float]]``
     - ``(time_seconds, tempo_bpm)`` changes
   * - ``melody.meter``
     - ``tuple[int, int]``
     - First time signature ``(numerator, denominator)``
   * - ``melody.time_signatures``
     - ``list[tuple[float, int, int]]``
     - All meters as ``(time, num, den)``
   * - ``melody.total_duration``
     - ``float``
     - Total duration in seconds
   * - ``melody.midi_data``
     - ``dict``
     - Raw import dictionary

Lower-level import (dict, not Melody)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from melody_features.io.midi import import_midi

   data = import_midi("example.mid")
   # Returns: dict | None
   # dict keys include: ID, pitches, starts, ends, tempo, tempo_changes,
   # time_signature_info, key_signature_info, total_duration, ...

Use :func:`~melody_features.io.midi.load_midi` unless you specifically need the
raw dictionary.

Directory of MIDI files
~~~~~~~~~~~~~~~~~~~~~~~

Two helpers load many files; both return a list of ``Melody`` objects:

.. code-block:: python

   import melody_features as mf
   from melody_features.io.midi import import_midi_from_directory, list_midi_files

   # Preferred MIDI helper in melody_features.io.midi
   paths = list_midi_files("path/to/dir", recursive=False)
   # Returns: list[str]  (naturally sorted absolute paths)
   # Raises: FileNotFoundError if the directory is missing or has no MIDI files

   melodies = import_midi_from_directory("path/to/dir", recursive=False)
   # Returns: list[Melody]
   # Failed files are skipped (warnings logged); successful loads only.

   # Corpus-oriented loader (MIDI directory; JSON is legacy — see below)
   melodies = mf.load_melodies_from_directory("path/to/dir", file_type="midi")
   # Returns: list[Melody]

Constructing a Melody without a file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from melody_features.core.representations import Melody

   melody = Melody.from_notes(
       pitches=[60, 62, 64],
       starts=[0.0, 0.5, 1.0],
       ends=[0.5, 1.0, 1.5],
       tempo=120.0,
       melody_id="toy",
   )
   # Returns: Melody


Bundled corpora
---------------

.. code-block:: python

   import melody_features as mf

   names = mf.list_available_corpora()
   # Returns: list[str]  e.g. ["essen", "pearce_default_idyom"]

   root = mf.get_corpus_path("essen")
   # Returns: pathlib.Path  (directory of MIDI files)
   # Raises: ValueError for unknown names; FileNotFoundError if missing on disk

   files = mf.get_corpus_files("essen", max_files=10)
   # Returns: list[pathlib.Path]
   # max_files=None returns every MIDI file in natural sort order

Example: load one bundled file and inspect it:

.. code-block:: python

   import melody_features as mf
   from melody_features.io.midi import load_midi

   path = mf.get_corpus_files("essen", max_files=1)[0]  # Path
   melody = load_midi(path)                             # Optional[Melody]
   if melody is None:
       raise RuntimeError(f"failed to load {path}")
   print(len(melody.pitches), melody.tempo, melody.meter)


Calling individual features
---------------------------

Atomic feature functions are imported from the package root (or
``melody_features.features``). Their signatures take note lists (and
sometimes tempo or related fields) rather than a ``Melody`` argument — but
:class:`~melody_features.core.nxt.Melody` exposes exactly those lists as
attributes, so everyday use is still a one-liner after loading:

.. code-block:: python

   import melody_features as mf
   from melody_features.io.midi import load_midi

   melody = load_midi("example.mid")
   if melody is None:
       raise RuntimeError("import failed")

   # Melody gives you the lists; pass them straight into the feature
   span = mf.pitch_range(melody.pitches)
   # pitch_range(pitches: list[int]) -> int

   hist = mf.basic_pitch_histogram(melody.pitches)
   # basic_pitch_histogram(pitches: list[int]) -> dict[int, int]

When a feature needs several fields, pull them from the same ``Melody``:

.. code-block:: python

   import melody_features as mf

   value = mf.melodic_pitch_variety(
       melody.pitches,
       melody.starts,
       tempo=melody.tempo,
   )
   # -> float

Discover what is available:

.. code-block:: python

   import melody_features as mf

   names = mf.list_available_features()
   # Returns: list[str]  (feature function names)

   records = mf.list_available_features(source="jsymbolic", detailed=True)
   # Returns: list[dict] when detailed=True
   # Optional filters: domain ("pitch"|"rhythm"|"both"),
   #                   feature_type (e.g. "absolute", "interval"),
   #                   source (e.g. "fantastic", "jsymbolic")

See also :func:`melody_features.get_feature_metadata` for a full metadata
table, and :doc:`feature_catalogue` for human-readable descriptions.


Batch extraction with ``get_all_features``
------------------------------------------

Signature (simplified):

.. code-block:: text

   get_all_features(
       input: PathLike | list[PathLike] | list[Melody],
       config: Config | None = None,
       log_level: int = logging.INFO,
       skip_idyom: bool = False,
       long_format: bool = False,
       join_metadata: bool = True,
   ) -> pandas.DataFrame | None

**Returns:** a :class:`pandas.DataFrame`, or ``None`` if no valid melodies
load. If melodies load but nothing is extracted, the result is an empty
DataFrame.

Accepted ``input`` forms:

* directory path containing MIDI files → every ``.mid`` / ``.midi`` inside
* single MIDI file path
* list of MIDI file paths
* list of in-memory :class:`~melody_features.core.representations.Melody` objects

Polyphonic files are skipped with a warning.

Minimal example:

.. code-block:: python

   import melody_features as mf
   import pandas as pd

   results = mf.get_all_features("path/to/midi/files", skip_idyom=True)
   assert isinstance(results, pd.DataFrame)
   # Default is wide format (see below).


Wide vs long output
-------------------

The snippets below are trimmed excerpts from a real run on two Essen melodies
(``skip_idyom=True``). A full wide table has on the order of 280 feature
columns; only a few are shown here.

Wide format (default)
~~~~~~~~~~~~~~~~~~~~~

* Shape: one row per melody
* ID columns: ``melody_num``, ``melody_id``
* Feature columns: ``{family}.{feature_name}``
  (e.g. ``absolute_pitch.pitch_range``)
* Cell types: scalars for descriptors; Python objects (lists/dicts) for
  sequence features

.. code-block:: python

   import melody_features as mf

   wide = mf.get_all_features("path/to/midi", skip_idyom=True, long_format=False)
   print(wide.shape)  # e.g. (2, 280)
   print(wide[
       [
           "melody_num",
           "melody_id",
           "absolute_pitch.pitch_range",
           "absolute_pitch.first_pitch",
           "absolute_pitch.basic_pitch_histogram",
       ]
   ])

Example output (paths shortened):

.. code-block:: text

   melody_num   melody_id      absolute_pitch.pitch_range  absolute_pitch.first_pitch  absolute_pitch.basic_pitch_histogram
           1   appenzel.mid                          24                          62  {62: 9, 64: 3, 66: 4, 67: 10, ...}
           2   arabic01.mid                          20                          74  {66: 3, 67: 9, 69: 8, 70: 4, ...}

Here ``pitch_range`` / ``first_pitch`` are **Descriptor** (scalar) columns;
``basic_pitch_histogram`` is a **Sequence** column whose cells are ``dict``
objects.

Long format
~~~~~~~~~~~

* Shape: one row per melody × feature
* Core columns: ``melody_num``, ``melody_id``, ``feature_name``, ``value``
* With ``join_metadata=True`` (default): also
  ``family``, ``source``, ``domain``, ``type``, ``description``, ``notes``,
  ``references``

.. code-block:: python

   import melody_features as mf

   long_df = mf.get_all_features(
       "path/to/midi",
       skip_idyom=True,
       long_format=True,
       join_metadata=True,
   )
   print(long_df[
       ["melody_num", "feature_name", "family", "type", "value"]
   ].head())

Example output (same two melodies; metadata columns beyond ``family`` /
``type`` omitted for space):

.. code-block:: text

   melody_num  feature_name                          family         type        value
           1  absolute_pitch.pitch_range            absolute_pitch Descriptor  24
           2  absolute_pitch.pitch_range            absolute_pitch Descriptor  20
           1  absolute_pitch.first_pitch            absolute_pitch Descriptor  62
           2  absolute_pitch.first_pitch            absolute_pitch Descriptor  74
           1  absolute_pitch.basic_pitch_histogram  absolute_pitch Sequence    {62: 9, 64: 3, ...}
           2  absolute_pitch.basic_pitch_histogram  absolute_pitch Sequence    {66: 3, 67: 9, ...}

Filter by metadata after the fact:

.. code-block:: python

   descriptors = long_df[
       (long_df["type"] == "Descriptor")
       & (long_df["source"].str.contains("jSymbolic", na=False))
   ]

Reshape an existing wide table later:

.. code-block:: python

   import melody_features as mf

   long_df = mf.to_long_format(wide, join_metadata=True)
   # Returns: pandas.DataFrame

   meta = mf.get_feature_metadata()
   # Returns: pandas.DataFrame with columns
   # feature_name, family, source, domain, type, description, notes, references


Configuration
-------------

:class:`~melody_features.Config` controls how *melody-features* behaves when you
call :func:`~melody_features.get_all_features`. Pass a ``Config`` to set:

* **corpus** — reference MIDI directory for FANTASTIC corpus statistics and
  (by default) IDyOM long-term-model pretraining
* **fantastic** — a :class:`~melody_features.FantasticConfig` (n-gram order,
  phrase gap, optional corpus override)
* **idyom** — a dictionary of named :class:`~melody_features.IDyOMConfig`
  objects (see below)
* **key_estimation** / **key_finding_algorithm** — how tonal context is obtained

If you omit ``config``, sensible defaults are used:

* **corpus:** bundled Pearce (2018) 903-melody set
* **fantastic:** ``max_ngram_order=5``, ``phrase_gap=1.5``
* **idyom:** four built-in runs (pitch/rhythm × short-term/long-term models)
* **key_estimation:** ``"infer_if_necessary"``
* **key_finding_algorithm:** ``"krumhansl_schmuckler"``

FANTASTIC settings
~~~~~~~~~~~~~~~~~~

:class:`~melody_features.FantasticConfig` controls FANTASTIC-style features:
phrase-based tokenization, lexical-diversity (m-type) measures, and
corpus-relative n-gram statistics (Müllensiefen, 2009). Its fields are:

* **max_ngram_order** (``int``, default ``5``) — inclusive maximum n-gram /
  m-type length. Features use orders ``1`` through this value (matching
  FANTASTIC's usual ``n.limits`` of 1–5). Raising it includes longer
  patterns; lowering it truncates them. Must be ``>= 1``.
* **phrase_gap** (``float``, default ``1.5``) — inter-onset interval
  threshold in quarter-note units. Within a melody, an IOI larger than
  this value starts a new phrase for tokenization. Smaller values yield
  more / shorter phrases; larger values keep notes together longer. Must
  be ``> 0``.
* **corpus** (optional path) — MIDI directory used to build the reference
  corpus statistics for distributional FANTASTIC features. If ``None``,
  the parent :class:`~melody_features.Config` corpus is used (bundled
  Pearce set by default). This does not change IDyOM pretraining; use
  ``IDyOMConfig.corpus`` for that.

.. code-block:: python

   import melody_features as mf

   mf.FantasticConfig(
       max_ngram_order=5,   # n-grams / m-types of length 1..5
       phrase_gap=1.5,     # new phrase when IOI > 1.5 quarter notes
       corpus=None,        # fall back to Config.corpus
   )

IDyOM viewpoints
~~~~~~~~~~~~~~~~

Each :class:`~melody_features.IDyOMConfig` takes:

* **target_viewpoints** — what to predict (commonly ``["cpitch"]`` or
  ``["onset"]``)
* **source_viewpoints** — what conditions that prediction

Build viewpoint lists from atomic names (strings) and optional linked
viewpoints. A linked viewpoint is a tuple (Python parentheses) of two or
more atomic names; IDyOM treats that combination as one joint source. You can
mix strings and tuples in the same list:

.. code-block:: python

   # Atomic sources only
   source_viewpoints=["cpitch", "cpintfref"]

   # One linked viewpoint: (cpint × cpintfref)
   source_viewpoints=[("cpint", "cpintfref")]

   # Mix linked and atomic
   source_viewpoints=[("cpint", "cpintfref"), "cpcint"]

   # Longer link (used by the package defaults for pitch)
   source_viewpoints=[("cpitch", "cpint", "cpintfref")]

Atomic names must be in
``melody_features.idyom.config.VALID_VIEWPOINTS`` (for example ``cpitch``,
``cpint``, ``cpintfref``, ``onset``, ``ioi``, ``ioi-ratio``). Linked tuples
must contain at least two atoms; every atom is checked against that set.

Multiple IDyOM runs
~~~~~~~~~~~~~~~~~~~

``Config.idyom`` is a non-empty ``dict[str, IDyOMConfig]``. Each entry is a
separate IDyOM job. The dict key is a label that appears in the output
column names for that run, so you can compare models, viewpoints, or corpora
in one ``get_all_features`` call.

For example, run a pitch short-term model (with a linked source viewpoint)
and a rhythm long-term model:

.. code-block:: python

   import melody_features as mf

   config = mf.Config(
       corpus=mf.get_corpus_path("pearce_default_idyom"),
       fantastic=mf.FantasticConfig(max_ngram_order=5, phrase_gap=1.5),
       idyom={
           "pitch_stm": mf.IDyOMConfig(
               target_viewpoints=["cpitch"],
               source_viewpoints=[("cpitch", "cpint", "cpintfref")],
               ppm_order=2,
               models=":stm",
           ),
           "rhythm_ltm": mf.IDyOMConfig(
               target_viewpoints=["onset"],
               source_viewpoints=["ioi", "ioi-ratio"],
               ppm_order=2,
               models=":ltm",
               corpus=mf.get_corpus_path("pearce_default_idyom"),
           ),
       },
       key_estimation="infer_if_necessary",
   )

   results = mf.get_all_features("path/to/midi", config=config)
   # Wide columns are labelled by the dict keys, for example:
   #   idyom.pitch_stm_mean_information_content
   #   idyom.rhythm_ltm_mean_information_content

Pattern: ``idyom.<config_key>_<metric>``. See :doc:`idyom` for the default
four-run column names and how they relate to ``expectation.*`` helpers.

You can add as many named ``IDyOMConfig`` entries as you need (different
``models``, viewpoints, ``ppm_order``, or pretraining ``corpus``). Per-entry
``IDyOMConfig.corpus`` overrides ``Config.corpus`` for that run; short-term
models (``models=":stm"``) do not use a pretraining corpus.

Minimal single-run example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import melody_features as mf

   config = mf.Config(
       corpus=mf.get_corpus_path("pearce_default_idyom"),
       fantastic=mf.FantasticConfig(
           max_ngram_order=5,
           phrase_gap=1.5,
           corpus=None,  # fall back to Config.corpus
       ),
       idyom={
           "default": mf.IDyOMConfig(
               target_viewpoints=["cpitch"],
               # Linked viewpoint via parentheses (tuple), plus an atomic one
               source_viewpoints=[("cpint", "cpintfref"), "cpcint"],
               ppm_order=2,
               models=":both",  # ":stm" | ":ltm" | ":both"
               corpus=None,
           )
       },
       key_estimation="infer_if_necessary",
       # "always_read_from_file" | "infer_if_necessary" | "always_infer"
   )

   results = mf.get_all_features("path/to/midi", config=config)

Skipping IDyOM
~~~~~~~~~~~~~~

.. code-block:: python

   import melody_features as mf

   results = mf.get_all_features("path/to/midi", skip_idyom=True)

Default IDyOM jobs and standalone ``run_idyom`` are documented in :doc:`idyom`.
Install/verify steps: :ref:`install-idyom`.

.. _key-estimation:

Key estimation
--------------

:class:`~melody_features.Config` ``key_estimation`` controls how tonal context
is obtained for tonality features, pitch spelling, and temporary MIDI written
for IDyOM:

* ``"infer_if_necessary"`` (default) — use a key signature from the MIDI file
  when present; otherwise infer with the key-finding algorithm
* ``"always_read_from_file"`` — require key metadata in the file; do not infer
* ``"always_infer"`` — ignore file keys and always infer from note content

``key_finding_algorithm`` is currently ``"krumhansl_schmuckler"`` only.

.. code-block:: python

   import melody_features as mf

   config = mf.Config(
       corpus=mf.get_corpus_path("pearce_default_idyom"),
       fantastic=mf.FantasticConfig(max_ngram_order=5, phrase_gap=1.5),
       idyom={
           "pitch_stm": mf.IDyOMConfig(
               target_viewpoints=["cpitch"],
               source_viewpoints=[("cpitch", "cpint", "cpintfref")],
               ppm_order=2,
               models=":stm",
           ),
       },
       key_estimation="always_infer",
   )
   results = mf.get_all_features("path/to/midi", config=config, skip_idyom=True)


Selective feature extraction
----------------------------

For a single :class:`~melody_features.core.representations.Melody`, scoped helpers return
a ``dict`` of feature name → value without running the full batch pipeline.
They live on :mod:`melody_features.features` (also imported where re-exported).

By source (toolkit provenance):

.. code-block:: python

   from melody_features.features import (
       get_jsymbolic_features,
       get_fantastic_features,
       get_must_features,
       get_midi_toolbox_features,
       get_simile_features,
       get_novel_features,
   )
   from melody_features.io.midi import load_midi

   melody = load_midi("example.mid")
   jsym = get_jsymbolic_features(melody)
   must = get_must_features(melody)

``get_fantastic_features`` accepts optional ``corpus_stats``, ``phrase_gap``,
and ``max_ngram_order``. Precompute stats with
:func:`~melody_features.corpus.make_corpus_stats` (see :doc:`corpora`).
``get_idyom_features`` only collects functions tagged ``@idyom`` that can run
from the melody alone — full IDyOM information-content columns still come from
``get_all_features`` / :doc:`idyom`.

By family / domain (examples):

.. code-block:: python

   from melody_features.features import (
       get_pitch_features,
       get_pitch_class_features,
       get_interval_features,
       get_timing_features,
       get_inter_onset_interval_features,
       get_rhythm_features,
       get_metre_features,
       get_expectation_features,
       get_complexity_features,
       get_contour_features,
       get_tonality_features,
       get_lexical_diversity_features,
       get_corpus_features,
   )

   pitch = get_pitch_features(melody)
   timing = get_timing_features(melody)
   # get_corpus_features(melody, corpus_stats, phrase_gap, max_ngram_order)

For most corpus-scale jobs prefer ``mf.get_all_features``. Contour class
APIs (full vectors): :doc:`contour`.


.. _fantastic-workflow:

FANTASTIC workflow
------------------

FANTASTIC-style features (Müllensiefen, 2009) follow a fixed pipeline. Batch
runs do this for you via :class:`~melody_features.FantasticConfig`; the steps
below are what happens under the hood, and what you can call yourself.

1. **Phrase segmentation** — split the melody on large inter-onset gaps
   (``phrase_gap`` in quarter notes).
2. **M-type tokenization** — each step becomes an
   :class:`~melody_features.melody_tokenizer.MType` (classified pitch interval
   × classified IOI ratio). Default scheme is ``"FANTASTIC"``; ``"SIMILE"``
   uses a coarser signed-interval map.
   FANTASTIC preserves direction and groups major/minor variants, following
   Table 1 and section 6.2.1 of `Müllensiefen (2009)
   <https://www.doc.gold.ac.uk/isms/m4s/FANTASTIC_docs.pdf>`_. Integer pitch
   codes are ``0`` for unison and signed ``1`` through ``9`` for second,
   third, fourth, tritone, fifth, sixth, seventh, octave and larger-than-octave
   intervals. Thus ascending and descending seconds have codes ``1`` and
   ``-1``; minor and major seconds in the same direction share a code.
   Rhythm codes remain ``1`` (shorter), ``2`` (similar), ``3`` (longer), with
   ``0`` for the first interval's unavailable preceding IOI ratio.
3. **N-gram counting** — accumulate m-type n-grams of order ``1`` …
   ``max_ngram_order`` separately within each phrase with
   :class:`~melody_features.ngram_counter.NGramCounter`.
   Counts of the same type are summed across phrases; windows never span
   a phrase boundary.
4. **Lexical / corpus features** — lexical-diversity (m-type) measures use
   the melody’s own counts; distributional corpus features compare those
   counts to reference document frequencies from ``Config.corpus`` (or a
   precomputed stats file — see :doc:`corpora`).

Inspect steps 1–3 directly:

.. code-block:: python

   from melody_features.melody_tokenizer import FantasticTokenizer
   from melody_features.ngram_counter import NGramCounter
   from melody_features.io.midi import load_midi

   melody = load_midi("example.mid")
   tok = FantasticTokenizer()  # scheme="FANTASTIC" (or "SIMILE")
   phrases = tok.tokenize_phrases(melody, phrase_gap=1.5)
   counts = tok.ngram_counts()  # summed within-phrase counts, orders 1..5
   # Or choose a maximum order explicitly:
   counter = NGramCounter()
   counter.count_sequences(phrases, max_order=5)
   counts = counter.get_counts()
   bigrams = counter.get_counts(n=2)

   # For a single phrase, tokenize_melody and count_ngrams remain available.

.. important::

   The corrected FANTASTIC tokenizer changes token identities and derived
   lexical/corpus statistics. Earlier versions discarded interval direction,
   separated major/minor variants, and could count across phrase boundaries.
   Phrase gaps now use the requested units and split before the note following
   the gap. Automatically generated corpus caches use a versioned filename
   (``_corpus_stats_v2_...``), so old caches are not reused. Regenerate any
   externally supplied corpus statistics and affected feature tables; integer
   token codes in old files are not compatible with the corrected mapping.
   Feature computation rejects missing or incompatible ``tokenizer_version``
   metadata, a different ``phrase_gap``, or an ``n_range`` that does not cover
   the requested orders. Rebuild statistics rather than editing their version
   field. ``load_corpus_stats`` still permits inspection of legacy JSON files;
   compatibility is enforced when the statistics are consumed.

   Phrase-gap comparisons allow a numerical tolerance of ``1e-9`` in the
   requested units, so roundoff at an exact threshold does not create a phrase
   boundary. Both tokenization methods replace their cached counts; empty or
   single-note input clears previous results.

Selective helpers that consume the same knobs (and optional ``corpus_stats``):

.. code-block:: python

   from melody_features.features import (
       get_fantastic_features,
       get_lexical_diversity_features,
       get_corpus_features,
   )

   # get_fantastic_features bundles lexical + corpus FANTASTIC outputs
   # get_lexical_diversity_features(melody, phrase_gap=1.5, max_ngram_order=5)
   # get_corpus_features(melody, corpus_stats, phrase_gap=1.5, max_ngram_order=5)

Phrase gap and max n-gram order for ``get_all_features`` are set on
:class:`~melody_features.FantasticConfig` (see Configuration above). API
reference: :doc:`api/tokenizer`.


MUST tokenization
-----------------

:class:`~melody_features.melody_tokenizer.MustTokenizer` builds pitch /
interval / duration distributions used by MUST-derived complexity features.
This is separate from the FANTASTIC m-type pipeline above.

.. code-block:: python

   from melody_features.melody_tokenizer import MustTokenizer
   from melody_features.io.midi import load_midi

   melody = load_midi("example.mid")
   must = MustTokenizer()
   pdist = must.pdist1(melody)          # MustDistribution
   print(pdist.as_dict())
   print(pdist.entropy())


Feature families and types
--------------------------

Features are grouped into families such as absolute pitch, pitch class,
pitch interval, contour, timing, inter-onset interval, tonality, metre,
expectation, complexity, lexical diversity, and corpus statistics.

Return kinds (see the ``type`` column in long format / the catalogue):

* **Descriptor** — scalar (``int``, ``float``, ``bool``)
* **Sequence** — collection (``list``, ``tuple``, ``dict``, …)


Putting it together
-------------------

End-to-end pattern for a small custom run:

.. code-block:: python

   import melody_features as mf
   from melody_features.io.midi import load_midi

   # 1) Inspect one melody
   path = mf.get_corpus_files("essen", max_files=1)[0]
   melody = load_midi(path)
   assert melody is not None
   print("range:", mf.pitch_range(melody.pitches))

   # 2) Batch-extract a few files (skip IDyOM for a faster local run)
   files = mf.get_corpus_files("essen", max_files=5)
   wide = mf.get_all_features(files, skip_idyom=True)
   print(wide.shape)
   print(wide[["melody_id", "absolute_pitch.pitch_range"]].head())


Legacy: JSON melody files
-------------------------

MIDI is the supported input. A legacy path still accepts a directory containing
one JSON file (a list of melody dicts) via
``load_melodies_from_directory(..., file_type="json")``. Each record needs
aligned ``pitches``, ``starts``, and ``ends`` (or a legacy ``MIDI Sequence``
string). Prefer MIDI or :meth:`~melody_features.core.representations.Melody.from_notes`
for new work.
