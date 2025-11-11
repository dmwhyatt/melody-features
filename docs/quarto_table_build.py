"""
Build a feature summary table for Quarto.

Collects labelled feature callables from `melody_features.features`, extracts
metadata, and emits a CSV or Markdown table with columns:

- Name
- Pre-existing Implementations
- Further References
- Description
- Type (Descriptor or Sequence)
- Notes

Usage:
  python docs/quarto_table_build.py --format csv --out /path/to/features_table.csv
  python docs/quarto_table_build.py --format qmd --out /path/to/features_table.qmd
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import os
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
from urllib.parse import quote

import pandas as pd

script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from melody_features import features as features_module
from melody_features.step_contour import StepContour
from melody_features.interpolation_contour import InterpolationContour
from melody_features.polynomial_contour import PolynomialContour
from melody_features.huron_contour import HuronContour
from melody_features.ngram_counter import NGramCounter
from melody_features.feature_decorators import corpus_prevalence, idyom, expectation, pitch, rhythm


@dataclass
class FeatureRow:
    name: str
    implementations: str
    references: str
    description: str
    type_label: str
    notes: str
    category: str
    domain: str
    sort_name: str


SECTION_RE = re.compile(r"^([A-Za-z ]+)\n[-]+$", re.MULTILINE)


def snake_to_title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def capitalize_ioi(text: str) -> str:
    """Capitalize 'IOI' in text while preserving other formatting."""
    if not text:
        return text
    return re.sub(r'\bioi\b', 'IOI', text, flags=re.IGNORECASE)


def normalize_feature_text(text: str) -> str:
    """Normalize acronyms and tokens in free text.
    - IOI -> IOI
    - df -> DF (word-boundary)
    - tfdf -> TFDF (word-boundary)
    - npvi -> NPVI (word-boundary)
    """
    if not text:
        return text
    # First, handle IOI via existing helper
    text = capitalize_ioi(text)
    # Then other acronyms
    text = re.sub(r"\bstm\b", "STM", text, flags=re.IGNORECASE)
    text = re.sub(r"\bltm\b", "LTM", text, flags=re.IGNORECASE)
    text = re.sub(r"\btfdf\b", "TFDF", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdf\b", "DF", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnpvi\b", "NPVI", text, flags=re.IGNORECASE)
    return text


def fix_possessive_feature_names(text: str) -> str:
    """Fix known feature name possessives that are lost by title-casing.
    E.g., 'Honores H' -> "Honore's H", 'Sichels S' -> "Sichel's S", etc.
    """
    if not text:
        return text
    replacements = [
        (r"\bHonores H\b", "Honore's H"),
        (r"\bSichels S\b", "Sichel's S"),
        (r"\bSimpsons D\b", "Simpson's D"),
        (r"\bYules K\b", "Yule's K"),
    ]
    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def extract_sections_from_docstring(doc: str) -> dict[str, str]:
    """Parse simple NumPy-style sections (Parameters, Returns, Notes, Citation, etc.)."""
    if not doc:
        return {}
    text = inspect.cleandoc(doc)
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        sections["Preamble"] = text.strip()
        return sections
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections["Preamble"] = preamble
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[title] = body
    return sections


def determine_type_from_return_annotation(obj) -> str:
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return "Descriptor"

    if ann is inspect.Signature.empty:
        return "Descriptor"

    scalar_types = (int, float, bool)
    sequence_type_names = {"list", "tuple", "dict", "set", "ndarray", "Series", "DataFrame"}

    if isinstance(ann, type):
        return "Descriptor" if issubclass(ann, scalar_types) else "Sequence"

    if isinstance(ann, str):
        lowered = ann.lower()
        if any(t in lowered for t in ("int", "float", "bool")) and not any(t in lowered for t in ("list", "tuple", "dict", "set")):
            return "Descriptor"
        if any(t in lowered for t in ("list", "tuple", "dict", "set", "ndarray")):
            return "Sequence"
        return "Descriptor"

    name = getattr(getattr(ann, "__origin__", None), "__name__", "") or getattr(ann, "_name", "") or str(ann)
    for seq_name in sequence_type_names:
        if seq_name in str(name):
            return "Sequence"
    return "Descriptor"


def collect_feature_rows(objs: Iterable[tuple[str, object]]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    repo_root = script_dir.parent

    def detect_repo_info() -> tuple[str, str]:
        """Return (repo_url, branch). Tries env, then pyproject, falls back to defaults."""
        repo_url = os.getenv("REPO_URL") or os.getenv("FEATURES_REPO_URL")
        branch = os.getenv("REPO_BRANCH") or os.getenv("FEATURES_REPO_BRANCH") or "main"
        if not repo_url:
            try:
                import tomllib
            except ImportError:
                tomllib = None
            if tomllib is not None:
                pyproj = repo_root / "pyproject.toml"
                if pyproj.exists():
                    try:
                        with open(pyproj, "rb") as f:
                            data = tomllib.load(f)
                        repo_url = (
                            data.get("project", {})
                            .get("urls", {})
                            .get("Homepage")
                            or data.get("project", {})
                            .get("urls", {})
                            .get("Repository")
                            or ""
                        )
                    except (OSError, tomllib.TOMLDecodeError):
                        repo_url = None
        if not repo_url:
            repo_url = "https://github.com/dmwhyatt/melody-features"
        return repo_url.rstrip("/"), branch

    REPO_URL, REPO_BRANCH = detect_repo_info()

    def build_source_url(obj: object) -> str:
        target = obj.fget if isinstance(obj, property) else obj

        try:
            target_unwrapped = inspect.unwrap(target)
        except (AttributeError, ValueError):
            target_unwrapped = target
        try:
            file_path_str = inspect.getsourcefile(target_unwrapped) or inspect.getfile(target_unwrapped)
            if not file_path_str:
                return ""
            file_path = Path(file_path_str)
            _, start_line = inspect.getsourcelines(target_unwrapped)
        except (OSError, TypeError):
            return ""

        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
            parts = file_path.parts
            rel_path = None
            if "src" in parts:
                idx = parts.index("src")
                rel_path = Path(*parts[idx:])
            elif "melody_features" in parts:
                idx = parts.index("melody_features")
                rel_path = Path("src") / Path(*parts[idx:])
            if rel_path is None:
                return ""

        quoted_path = quote(rel_path.as_posix())
        return f"{REPO_URL}/blob/{REPO_BRANCH}/{quoted_path}#L{start_line}"
    
    def format_source_name(raw_name: str) -> str:
        """Return canonical display names for pre-existing implementations.
        Falls back to Title Case when not explicitly mapped.
        """
        if not raw_name:
            return ""
        normalized = raw_name.replace("_", " ").strip().lower()
        mapping = {
            "fantastic": "FANTASTIC",
            "jsymbolic": "jSymbolic",
            "midi toolbox": "MIDI Toolbox",
            "midi_toolbox": "MIDI Toolbox",
            "simile": "SIMILE",
            "idyom": "IDyOM",
        }
        return mapping.get(normalized, raw_name.replace("_", " ").strip().title())
    for name, obj in objs:
        if name.startswith("get_"):
            continue
        
        # Skip InverseEntropyWeighting class
        if name == "InverseEntropyWeighting":
            continue
            
        is_property = isinstance(obj, property)
        
        if is_property and hasattr(obj, 'fget') and obj.fget is not None:
            feature_types = getattr(obj.fget, "_feature_types", None)
        else:
            feature_types = getattr(obj, "_feature_types", None)
        if not feature_types and not is_property:
            continue

        if "." in name:
            class_name, prop_name = name.split(".", 1)
            class_display = class_name
            if class_name == "StepContour":
                class_display = "Step Contour"
            elif class_name == "InterpolationContour":
                class_display = "Interpolation Contour"
            elif class_name == "PolynomialContour":
                class_display = "Polynomial Contour"
            elif class_name == "HuronContour":
                class_display = "Huron Contour"
            elif class_name == "NGramCounter":
                class_display = ""
            class_part = f"{class_display} " if class_display else ""
            pretty_name = f"{class_part}{snake_to_title(prop_name)}".strip()
        else:
            pretty_name = snake_to_title(name)

        # Apply possessive fixes and acronym normalization to the display name
        pretty_name = fix_possessive_feature_names(normalize_feature_text(pretty_name))

        source_url = build_source_url(obj)
        display_name = (
            f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">{pretty_name}</a>'
            if source_url
            else pretty_name
        )

        feature_sources = getattr(obj, "_feature_sources", [])
        if feature_sources:
            implementations = ", ".join(sorted({format_source_name(s) for s in feature_sources}))
        else:
            implementations = ""
            if is_property and "." in name:
                class_name = name.split(".", 1)[0]
                class_source_map = {
                    "StepContour": "FANTASTIC",
                    "InterpolationContour": "FANTASTIC",
                    "PolynomialContour": "FANTASTIC",
                    "HuronContour": "FANTASTIC",
                    "NGramCounter": "FANTASTIC",
                }
                mapped = class_source_map.get(class_name)
                if mapped:
                    implementations = mapped

        doc_string = inspect.getdoc(obj.fget) if is_property else inspect.getdoc(obj)
        sections = extract_sections_from_docstring(doc_string or "")
        description = normalize_feature_text(" ".join(sections.get("Preamble", "").split()))
        notes = normalize_feature_text(" ".join(sections.get("Note", "").split()))

        citation_section = sections.get("Citation", "").strip()
        if citation_section:
            references = normalize_feature_text(
                " | ".join([" ".join(p.split()) for p in re.split(r"\n\s*\n", citation_section) if p.strip()])
            )
        else:
            references = ""

        if is_property:
            type_label = "Descriptor"
            if "." in name:
                class_name, prop_name = name.split(".", 1)
                if class_name == "PolynomialContour" and prop_name == "coefficients":
                    type_label = "Sequence"  # list[float]
                if class_name == "HuronContour" and prop_name == "huron_contour":
                    type_label = "Descriptor"  # str
        else:
            type_label = determine_type_from_return_annotation(obj)

        category = _get_feature_category(obj)
        
        # Determine domain from decorator if present
        if is_property and hasattr(obj, 'fget') and obj.fget is not None:
            domain_attr = getattr(obj.fget, "_feature_domain", None)
        else:
            domain_attr = getattr(obj, "_feature_domain", None)

        # Set domain to "pitch" for contour class properties if not already set
        if is_property and "." in name:
            class_name = name.split(".", 1)[0]
            contour_classes = ["StepContour", "InterpolationContour", "PolynomialContour", "HuronContour"]
            if class_name in contour_classes and not domain_attr:
                domain_attr = "pitch"

        domain_for_filter = domain_attr if domain_attr else ""

        rows.append(
            FeatureRow(
                name=display_name,
                implementations=implementations,
                references=references,
                description=description,
                type_label=type_label,
                notes=notes,
                category=category,
                domain=domain_for_filter,
                sort_name=pretty_name,
            )
        )
    return rows


def to_dataframe(rows: list[FeatureRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    sort_cols = ['sort_name'] if 'sort_name' in df.columns else ['name']
    df = df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    
    return df

def _get_feature_category(obj) -> str:
    """Determine the feature category based on the actual feature type decorator.
    Returns a comma-separated string of categories for features that belong to multiple categories.
    """
    is_property = isinstance(obj, property)
    if is_property and hasattr(obj, 'fget') and obj.fget is not None:
        feature_types = getattr(obj.fget, "_feature_types", None)
    else:
        feature_types = getattr(obj, "_feature_types", None)
    
    type_mapping = {
        'pitch': 'Pitch',
        'interval': 'Interval', 
        'contour': 'Contour',
        'rhythm': 'Rhythm',
        'complexity': 'Complexity',
        'tonality': 'Tonality',
        'metre': 'Metre',
        'expectation': 'Expectation',
        'corpus_prevalence': 'Corpus',
        'mtype': 'MType',
        'class_based': 'Class-based',
        'descriptives': 'Descriptives',
    }
    
    if feature_types and len(feature_types) > 0:
        # Map all feature types to their categories
        categories = []
        for feature_type in feature_types:
            mapped = type_mapping.get(feature_type)
            if mapped and mapped not in categories:
                categories.append(mapped)
        
        if categories:
            return ', '.join(categories)
        # Fallback: if no mapping found, use title case of first type
        return feature_types[0].title()
    
    # handle class based features (fallback for features without decorators)
    if hasattr(obj, '__name__'):
        name = obj.__name__
        if name in ['honores_h', 'yules_k', 'simpsons_d', 'sichels_s', 'mean_entropy', 'mean_productivity']:
            return 'Complexity'
        elif name in ['class_label', 'global_variation', 'global_direction', 'local_variation', 'coefficients']:
            return 'Contour'
    
    # get properties
    if isinstance(obj, property):
        if hasattr(obj, 'fget') and obj.fget:
            if hasattr(obj.fget, '__qualname__'):
                qualname = obj.fget.__qualname__
                if 'NGramCounter' in qualname:
                    return 'Complexity'
                elif any(cls in qualname for cls in ['HuronContour', 'StepContour', 'InterpolationContour', 'PolynomialContour']):
                    return 'Contour'
        return 'Other'
    
    return 'Other'


def build_table() -> pd.DataFrame:
    members = inspect.getmembers(features_module)
    
    all_features = []
    
    functions = [(n, o) for n, o in members if inspect.isfunction(o)]
    all_features.extend(functions)
    
    feature_classes = [StepContour, InterpolationContour, PolynomialContour, HuronContour, NGramCounter]
    excluded_properties = {"count_values", "freq_spec", "total_tokens"}
    
    for cls in feature_classes:
        class_name = cls.__name__
        for prop_name, prop_obj in inspect.getmembers(cls):
            if isinstance(prop_obj, property) and prop_name not in excluded_properties:
                all_features.append((f"{class_name}.{prop_name}", prop_obj))
    
    # Add placeholder functions for IDyOM features that are dynamically generated
    @idyom
    @expectation
    @pitch
    def pitch_mean_information_content_stm(_melody):
        """The average information content across all notes in a melody,
        calculated using IDyOM's prediction-by-partial-matching (PPM) algorithm. 
        Information content is perceptually related to surprise, and can be calculated
        for pitches or rhythms.
        
        Citation
        --------
        Pearce, M. (2005)
        """
        # Placeholder function for table generation

    @idyom
    @expectation
    @corpus_prevalence
    @pitch
    def pitch_mean_information_content_ltm(_melody):
        """The average information content across all notes in a melody,
        calculated using IDyOM's long-term model (LTM). Information content is
        perceptually related to surprise, and can be calculated for pitches or rhythms.
        
        Citation
        --------
        Pearce, M. (2005)
        """
        # Placeholder function for table generation

    @idyom
    @expectation
    @rhythm
    def rhythm_mean_information_content_stm(_melody):
        """The average rhythmic information content across all notes in a melody,
        calculated using IDyOM's short-term model (STM). Information content is
        perceptually related to surprise, and can be calculated for pitches or rhythms.
        
        Citation
        --------
        Pearce, M. (2005)
        """
        # Placeholder function for table generation

    @idyom
    @expectation
    @corpus_prevalence
    @rhythm
    def rhythm_mean_information_content_ltm(_melody):
        """The average rhythmic information content across all notes in a melody,
        calculated using IDyOM's long-term model (LTM). Information content is
        perceptually related to surprise, and can be calculated for pitches or rhythms.
        
        Citation
        --------
        Pearce, M. (2005)
        """
        # Placeholder function for table generation
    
    all_features.extend(
        [
            ("pitch_mean_information_content_stm", pitch_mean_information_content_stm),
            ("pitch_mean_information_content_ltm", pitch_mean_information_content_ltm),
            ("rhythm_mean_information_content_stm", rhythm_mean_information_content_stm),
            ("rhythm_mean_information_content_ltm", rhythm_mean_information_content_ltm),
        ]
    )
    
    rows = collect_feature_rows(all_features)
    return to_dataframe(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a feature summary table for Quarto.")
    parser.add_argument("--format", choices=["csv", "qmd"], default="qmd", help="Output format")
    parser.add_argument("--out", required=True, help="Output file path")
    args = parser.parse_args()

    df = build_table()

    if args.format == "csv":
        df.rename(
            columns={
                "name": "Name",
                "implementations": "Pre-existing Implementations",
                "references": "Further References",
                "description": "Description",
                "type_label": "Type",
                "notes": "Notes",
            }
        ).to_csv(args.out, index=False)
    else:
        _ = df.rename(
            columns={
                "name": "Name",
                "implementations": "Pre-existing Implementations",
                "references": "Further References",
                "description": "Description",
                "type_label": "Type",
                "notes": "Notes",
            }
        )
        
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write("title: \"Melody Features Summary\"\n")
            f.write("page-layout: full\n")
            f.write("format:\n")
            f.write("  html:\n")
            f.write("    theme: cosmo\n")
            f.write("    toc: false\n")
            f.write("---\n\n")
            f.write("This table provides a comprehensive overview of all melody features available in this package.\n\n")
            f.write("```{python}\n")
            f.write("#| echo: false\n")
            f.write("import pandas as pd\n")
            f.write("import sys\n")
            f.write("from pathlib import Path\n")
            f.write("script_dir = Path.cwd()\n")
            f.write("src_dir = script_dir / \"src\"\n")
            f.write("docs_dir = script_dir / \"docs\"\n")
            f.write("sys.path.insert(0, str(src_dir))\n")
            f.write("sys.path.insert(0, str(docs_dir))\n")
            f.write("from quarto_table_build import build_table\n\n")
            f.write("df = build_table()\n")
            f.write("df_renamed = df.rename(columns={\n")
            f.write("    'name': 'Name',\n")
            f.write("    'implementations': 'Pre-existing Implementations',\n")
            f.write("    'references': 'Further References',\n")
            f.write("    'description': 'Description',\n")
            f.write("    'type_label': 'Type',\n")
            f.write("    'notes': 'Notes'\n")
            f.write("})\n")
            f.write("```\n\n")
            f.write("```{python}\n")
            f.write("#| echo: false\n")
            f.write("#| output: asis\n")
            f.write("from IPython.display import display, HTML\n")
            f.write("import json\n\n")
            f.write("# Create a simple interactive table with search and sort\n")
            f.write("# Add category data to each row for filtering\n")
            f.write("df_renamed['data-category'] = df_renamed.index.map(lambda i: df.iloc[i]['category'])\n")
            f.write("df_renamed['data-domain'] = df_renamed.index.map(lambda i: df.iloc[i].get('domain', ''))\n")
            f.write("\n")
            f.write("# Create a single table with category data for filtering (exclude category columns from display)\n")
            f.write("df_display = df_renamed.drop(columns=['category', 'domain', 'data-category', 'data-domain', 'sort_name'], errors='ignore')\n")
            f.write("table_html = df_display.to_html(classes='table table-striped table-hover', table_id='features-table', escape=False, index=False)\n")
            f.write("\n")
            f.write("# Add data-category attributes to table rows using a more robust approach\n")
            f.write("import re\n")
            f.write("def add_data_category_attributes(html, df_with_categories):\n")
            f.write("    # Split HTML into lines for easier processing\n")
            f.write("    lines = html.split('\\n')\n")
            f.write("    result_lines = []\n")
            f.write("    data_row_index = 0\n")
            f.write("    \n")
            f.write("    for line in lines:\n")
            f.write("        if '<tr>' in line and 'thead' not in line:\n")
            f.write("            # This is a data row, add the data-category attribute\n")
            f.write("            if data_row_index < len(df_with_categories):\n")
            f.write("                category = df_with_categories.iloc[data_row_index]['category']\n")
            f.write("                impls = df_with_categories.iloc[data_row_index].get('Pre-existing Implementations', '') or ''\n")
            f.write("                ftype = df_with_categories.iloc[data_row_index].get('Type', '') or ''\n")
            f.write("                domain = df_with_categories.iloc[data_row_index].get('domain', '') or ''\n")
            f.write("                line = line.replace('<tr>', f'<tr data-category=\"{category}\" data-impl=\"{impls}\" data-type=\"{ftype}\" data-domain=\"{domain}\">')\n")
            f.write("                data_row_index += 1\n")
            f.write("        result_lines.append(line)\n")
            f.write("    \n")
            f.write("    return '\\n'.join(result_lines)\n")
            f.write("\n")
            f.write("table_html = add_data_category_attributes(table_html, df_renamed)\n\n")
            f.write("# Generate dropdown options for filters\n")
            f.write("# Extract all unique categories from comma-separated values\n")
            f.write("all_categories = set()\n")
            f.write("for cat_str in df['category'].fillna(''):\n")
            f.write("    for cat in [c.strip() for c in str(cat_str).split(',') if c.strip()]:\n")
            f.write("        all_categories.add(cat)\n")
            f.write("categories = sorted(all_categories)\n")
            f.write("category_options = '\\n'.join([f'        <option value=\"{cat}\">{cat}</option>' for cat in categories])\n\n")
            f.write("# Domains (from decorators)\n")
            f.write("domains = ['Pitch', 'Rhythm', 'Both']\n")
            f.write("domain_options = '\\n'.join([f'        <option value=\"{opt}\">{opt}</option>' for opt in domains])\n\n")
            f.write("# Implementations: split comma-separated values and deduplicate\n")
            f.write("impl_tokens = set()\n")
            f.write("for v in df['implementations'].fillna(''):\n")
            f.write("    for token in [t.strip() for t in str(v).split(',') if t.strip()]:\n")
            f.write("        # Normalize tokens to canonical display forms to avoid duplicates\n")
            f.write("        token_low = token.lower()\n")
            f.write("        if token_low in ['fantastic']:\n")
            f.write("            impl_tokens.add('FANTASTIC')\n")
            f.write("        elif token_low in ['jsymbolic']:\n")
            f.write("            impl_tokens.add('jSymbolic')\n")
            f.write("        elif token_low in ['midi toolbox', 'midi_toolbox']:\n")
            f.write("            impl_tokens.add('MIDI Toolbox')\n")
            f.write("        elif token_low in ['simile']:\n")
            f.write("            impl_tokens.add('SIMILE')\n")
            f.write("        elif token_low in ['idyom']:\n")
            f.write("            impl_tokens.add('IDyOM')\n")
            f.write("        else:\n")
            f.write("            impl_tokens.add(token)\n")
            f.write("implementation_options = '\\n'.join([f'        <option value=\"{opt}\">{opt}</option>' for opt in sorted(impl_tokens)])\n\n")
            f.write("# Types (Descriptor/Sequence)\n")
            f.write("type_options = '\\n'.join([f'        <option value=\"{opt}\">{opt}</option>' for opt in sorted(df['type_label'].unique())])\n\n")
            f.write("# Add custom CSS and JavaScript for interactivity\n")
            f.write("interactive_html = '''\n")
            f.write("<style>\n")
            f.write("/* Make Quarto content span full page width */\n")
            f.write(":where(.quarto-container, main.content, .page-content, .content){max-width:100% !important;width:100% !important;padding-left:0 !important;padding-right:0 !important;}\n")
            f.write("body{margin-left:0;margin-right:0;}\n")
            f.write(".filter-container {\n")
            f.write("    display: flex;\n")
            f.write("    gap: 15px;\n")
            f.write("    margin-bottom: 20px;\n")
            f.write("    flex-wrap: wrap;\n")
            f.write("}\n")
            f.write(".search-input {\n")
            f.write("    flex: 1;\n")
            f.write("    min-width: 200px;\n")
            f.write("    padding: 10px;\n")
            f.write("    border: 1px solid #ddd;\n")
            f.write("    border-radius: 4px;\n")
            f.write("    font-size: 16px;\n")
            f.write("}\n")
            f.write(".category-filter {\n")
            f.write("    padding: 10px;\n")
            f.write("    border: 1px solid #ddd;\n")
            f.write("    border-radius: 4px;\n")
            f.write("    font-size: 16px;\n")
            f.write("    background-color: white;\n")
            f.write("    min-width: 150px;\n")
            f.write("}\n")
            f.write(".feature-counter {\n")
            f.write("    padding: 10px 15px;\n")
            f.write("    font-size: 16px;\n")
            f.write("    font-weight: 600;\n")
            f.write("    color: #495057;\n")
            f.write("    background-color: #f8f9fa;\n")
            f.write("    border: 1px solid #dee2e6;\n")
            f.write("    border-radius: 4px;\n")
            f.write("    white-space: nowrap;\n")
            f.write("    align-self: center;\n")
            f.write("}\n")
            f.write("/* Visual grouping within the table */\n")
            f.write(".table tr[data-category] {\n")
            f.write("    border-top: 2px solid #e9ecef;\n")
            f.write("}\n")
            f.write(".table tr[data-category]:first-child {\n")
            f.write("    border-top: 3px solid #007bff;\n")
            f.write("}\n")
            f.write(".sortable {\n")
            f.write("    cursor: pointer;\n")
            f.write("    user-select: none;\n")
            f.write("    position: relative;\n")
            f.write("}\n")
            f.write(".sortable:hover {\n")
            f.write("    background-color: #f8f9fa;\n")
            f.write("}\n")
            f.write(".sortable::after {\n")
            f.write("    content: ' ↕';\n")
            f.write("    opacity: 0.5;\n")
            f.write("}\n")
            f.write(".sortable.asc::after {\n")
            f.write("    content: ' ↑';\n")
            f.write("    opacity: 1;\n")
            f.write("}\n")
            f.write(".sortable.desc::after {\n")
            f.write("    content: ' ↓';\n")
            f.write("    opacity: 1;\n")
            f.write("}\n")
            f.write(".table-container {\n")
            f.write("    width: 100%;\n")
            f.write("    max-width: none;\n")
            f.write("}\n")
            f.write(".table {\n")
            f.write("    width: 100%;\n")
            f.write("    table-layout: auto;\n")
            f.write("    border-collapse: collapse;\n")
            f.write("}\n")
            f.write(".table td, .table th {\n")
            f.write("    word-wrap: break-word;\n")
            f.write("    word-break: normal;\n")
            f.write("    vertical-align: top;\n")
            f.write("    padding: 12px 15px;\n")
            f.write("    border: 1px solid #dee2e6;\n")
            f.write("}\n")
            f.write(".table th {\n")
            f.write("    white-space: normal;\n")
            f.write("    background-color: #f8f9fa;\n")
            f.write("    font-weight: bold;\n")
            f.write("    text-align: left;\n")
            f.write("    hyphens: auto;\n")
            f.write("}\n")
            f.write("/* Feature name column sizing and wrapping */\n")
            f.write("#features-table td:first-child {\n")
            f.write("    min-width: 220px;\n")
            f.write("    width: 26%;\n")
            f.write("}\n")
            f.write("#features-table td:first-child a {\n")
            f.write("    white-space: normal;\n")
            f.write("    word-break: keep-all;\n")
            f.write("    hyphens: auto;\n")
            f.write("}\n")
            f.write("/* Dynamic column sizing - let content determine width */\n")
            f.write("</style>\n")
            f.write("<div class='filter-container'>\n")
            f.write("    <input type='text' class='search-input' id='searchInput' placeholder='Search features...'>\n")
            f.write("    <select class='category-filter' id='domainFilter'>\n")
            f.write("        <option value=''>All Domains</option>\n")
            f.write("        {domain_options}\n")
            f.write("    </select>\n")
            f.write("    <select class='category-filter' id='categoryFilter'>\n")
            f.write("        <option value=''>All Categories</option>\n")
            f.write("        {category_options}\n")
            f.write("    </select>\n")
            f.write("    <select class='category-filter' id='implementationFilter'>\n")
            f.write("        <option value=''>All Implementations</option>\n")
            f.write("        {implementation_options}\n")
            f.write("    </select>\n")
            f.write("    <select class='category-filter' id='typeFilter'>\n")
            f.write("        <option value=''>All Types</option>\n")
            f.write("        {type_options}\n")
            f.write("    </select>\n")
            f.write("    <span class='feature-counter' id='featureCounter'>0 features</span>\n")
            f.write("</div>\n")
            f.write("<div class='table-container'>\n")
            f.write("'''\n\n")
            f.write("# Format the HTML with category options\n")
            f.write("formatted_html = (interactive_html\n")
            f.write("                    .replace('{category_options}', category_options)\n")
            f.write("                    .replace('{implementation_options}', implementation_options)\n")
            f.write("                    .replace('{type_options}', type_options)\n")
            f.write("                    .replace('{domain_options}', domain_options))\n")
            f.write("display(HTML(formatted_html + table_html + '</div>'))\n\n")
            f.write("# Add JavaScript for search and sort functionality\n")
            f.write("display(HTML('''\n")
            f.write("<script>\n")
            f.write("document.addEventListener('DOMContentLoaded', function() {\n")
            f.write("    const table = document.getElementById('features-table');\n")
            f.write("    const searchInput = document.getElementById('searchInput');\n")
            f.write("    const categoryFilter = document.getElementById('categoryFilter');\n")
            f.write("    const implementationFilter = document.getElementById('implementationFilter');\n")
            f.write("    const typeFilter = document.getElementById('typeFilter');\n")
            f.write("    const domainFilter = document.getElementById('domainFilter');\n")
            f.write("    const featureCounter = document.getElementById('featureCounter');\n")
            f.write("    const tbody = table.querySelector('tbody');\n")
            f.write("    const rows = Array.from(tbody.querySelectorAll('tr'));\n")
            f.write("    \n")
            f.write("    // Add sortable class to headers\n")
            f.write("    const headers = table.querySelectorAll('th');\n")
            f.write("    headers.forEach((header, index) => {\n")
            f.write("        header.classList.add('sortable');\n")
            f.write("        header.addEventListener('click', () => sortTable(index));\n")
            f.write("    });\n")
            f.write("    \n")
            f.write("    // Filter functionality\n")
            f.write("    function filterRows() {\n")
            f.write("        const searchTerm = searchInput.value.toLowerCase();\n")
            f.write("        const selectedCategory = categoryFilter.value;\n")
            f.write("        const selectedImplementation = implementationFilter.value;\n")
            f.write("        const selectedType = typeFilter.value;\n")
            f.write("        const selectedDomain = domainFilter.value;\n")
            f.write("        \n")
            f.write("        rows.forEach(row => {\n")
            f.write("            const text = row.textContent.toLowerCase();\n")
            f.write("            const category = row.getAttribute('data-category') || '';\n")
            f.write("            const impl = row.getAttribute('data-impl') || '';\n")
            f.write("            const ftype = row.getAttribute('data-type') || '';\n")
            f.write("            const domain = row.getAttribute('data-domain') || '';\n")
            f.write("            \n")
            f.write("            const matchesSearch = text.includes(searchTerm);\n")
            f.write("            const matchesCategory = !selectedCategory || category.split(',').map(s => s.trim()).includes(selectedCategory);\n")
            f.write("            const matchesImplementation = !selectedImplementation || impl.split(',').map(s => s.trim()).includes(selectedImplementation);\n")
            f.write("            const matchesType = !selectedType || ftype === selectedType;\n")
            f.write("            // Domain filtering: handle special case of 'pitch,rhythm' appearing in both filters\n")
            f.write("            // 'both' domain features only appear when 'Both' is selected\n")
            f.write("            let matchesDomain = true;\n")
            f.write("            if (selectedDomain) {\n")
            f.write("                if (selectedDomain === 'Pitch') {\n")
            f.write("                    matchesDomain = domain === 'pitch' || domain === 'pitch,rhythm';\n")
            f.write("                } else if (selectedDomain === 'Rhythm') {\n")
            f.write("                    matchesDomain = domain === 'rhythm' || domain === 'pitch,rhythm';\n")
            f.write("                } else if (selectedDomain === 'Both') {\n")
            f.write("                    matchesDomain = domain === 'both';\n")
            f.write("                } else {\n")
            f.write("                    matchesDomain = domain === selectedDomain.toLowerCase();\n")
            f.write("                }\n")
            f.write("            }\n")
            f.write("            \n")
            f.write("            row.style.display = (matchesSearch && matchesCategory && matchesImplementation && matchesType && matchesDomain) ? '' : 'none';\n")
            f.write("        });\n")
            f.write("        \n")
            f.write("        // Update feature counter\n")
            f.write("        const visibleCount = rows.filter(row => row.style.display !== 'none').length;\n")
            f.write("        const totalCount = rows.length;\n")
            f.write("        if (visibleCount === totalCount) {\n")
            f.write("            featureCounter.textContent = `${totalCount} feature${totalCount !== 1 ? 's' : ''}`;\n")
            f.write("        } else {\n")
            f.write("            featureCounter.textContent = `${visibleCount} of ${totalCount} feature${totalCount !== 1 ? 's' : ''}`;\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("    \n")
            f.write("    // Add event listeners\n")
            f.write("    searchInput.addEventListener('input', filterRows);\n")
            f.write("    categoryFilter.addEventListener('change', filterRows);\n")
            f.write("    implementationFilter.addEventListener('change', filterRows);\n")
            f.write("    typeFilter.addEventListener('change', filterRows);\n")
            f.write("    domainFilter.addEventListener('change', filterRows);\n")
            f.write("    \n")
            f.write("    // Sort functionality\n")
            f.write("    let sortColumn = -1;\n")
            f.write("    let sortDirection = 'asc';\n")
            f.write("    \n")
            f.write("    function sortTable(columnIndex) {\n")
            f.write("        const isAsc = sortColumn === columnIndex && sortDirection === 'asc';\n")
            f.write("        sortDirection = isAsc ? 'desc' : 'asc';\n")
            f.write("        sortColumn = columnIndex;\n")
            f.write("        \n")
            f.write("        // Update header classes\n")
            f.write("        headers.forEach((header, index) => {\n")
            f.write("            header.classList.remove('asc', 'desc');\n")
            f.write("            if (index === columnIndex) {\n")
            f.write("                header.classList.add(sortDirection);\n")
            f.write("            }\n")
            f.write("        });\n")
            f.write("        \n")
            f.write("        // Sort rows\n")
            f.write("        const sortedRows = rows.sort((a, b) => {\n")
            f.write("            const aText = a.cells[columnIndex].textContent.trim();\n")
            f.write("            const bText = b.cells[columnIndex].textContent.trim();\n")
            f.write("            \n")
            f.write("            // Try to parse as numbers\n")
            f.write("            const aNum = parseFloat(aText);\n")
            f.write("            const bNum = parseFloat(bText);\n")
            f.write("            \n")
            f.write("            if (!isNaN(aNum) && !isNaN(bNum)) {\n")
            f.write("                return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;\n")
            f.write("            }\n")
            f.write("            \n")
            f.write("            // String comparison\n")
            f.write("            return sortDirection === 'asc' ? \n")
            f.write("                aText.localeCompare(bText) : \n")
            f.write("                bText.localeCompare(aText);\n")
            f.write("        });\n")
            f.write("        \n")
            f.write("        // Reorder rows in DOM\n")
            f.write("        sortedRows.forEach(row => tbody.appendChild(row));\n")
            f.write("    }\n")
            f.write("    \n")
            f.write("    // Initialize counter on page load\n")
            f.write("    filterRows();\n")
            f.write("});\n")
            f.write("</script>\n")
            f.write("'''))\n")
            f.write("```\n\n")
            f.write("## Sources\n\n")
            f.write("- **FANTASTIC**: Müllensiefen, D. (2009). Feature ANalysis Technology Accessing STatistics (In a Corpus): Technical Report v1.5\n")
            f.write("- **jSymbolic**: McKay, C., & Fujinaga, I. (2006). jSymbolic: A Feature Extractor for MIDI Files\n")
            f.write("- **IDyOM**: Pearce, M. T. (2005). The construction and evaluation of statistical models of melodic structure in music perception and composition\n")
            f.write("- **MIDI Toolbox**: Eerola, T., & Toiviainen, P. (2004). MIDI Toolbox: MATLAB Tools for Music Research\n")
            f.write("- **Melsim**: Silas, S., & Frieler, K. (n.d.). Melsim: Framework for calculating tons of melodic similarities\n")
            f.write("- **Simile**: Müllensiefen, D., & Frieler, K. (2004). The Simile algorithms documentation 0.3\n")
            f.write("- **Novel**: Custom features introduced in this package\n\n")
            f.write("## Feature Types\n\n")
            f.write("- **Descriptor**: Returns a single scalar value (int, float, bool)\n")
            f.write("- **Sequence**: Returns a collection (list, tuple, dict, etc.)\n")


if __name__ == "__main__":
    main()
