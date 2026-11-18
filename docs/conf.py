# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DOCS_DIR))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# -- Project information -----------------------------------------------------

project = "melody-features"
author = "David Whyatt, Peter Harrison"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("melody-features")
except Exception:
    release = "1.3.4"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "feature_catalogue",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/__pycache__"]

# Autodoc / Napoleon
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = []
add_module_names = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/melody_features_logo.png"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}

# Use the theme's default sidebars (do not force Alabaster templates).
html_sidebars = {}

html_title = f"{project} {release} documentation"
html_short_title = "melody-features"

html_context = {
    "display_github": True,
    "github_user": "dmwhyatt",
    "github_repo": "melody-features",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


def _is_alias_member(name: str, obj: object) -> bool:
    """True when ``name`` is an alternate binding of another callable/class.

    Simple aliases such as ``ambitus = pitch_range`` keep the original
    ``__name__``, so they should not get a duplicate API entry — the note on
    the canonical feature is enough.
    """
    canonical = getattr(obj, "__name__", None)
    return isinstance(canonical, str) and canonical != name


def setup(app):
    def skip_aliases(app, what, name, obj, skip, options):
        # Return True to skip aliases; return None so other handlers / defaults
        # still apply (emit_firstresult stops at the first non-None result).
        if _is_alias_member(name, obj):
            return True
        return None

    # Run early so a later handler returning False cannot force aliases in.
    app.connect("autodoc-skip-member", skip_aliases, priority=100)
