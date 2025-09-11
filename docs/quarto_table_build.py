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
from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path

import pandas as pd

# Add src directory to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from melody_features import features as features_module
from melody_features.step_contour import StepContour
from melody_features.interpolation_contour import InterpolationContour
from melody_features.polynomial_contour import PolynomialContour
from melody_features.huron_contour import HuronContour
from melody_features.ngram_counter import NGramCounter


@dataclass
class FeatureRow:
    name: str
    implementations: str
    references: str
    description: str
    type_label: str
    notes: str


SECTION_RE = re.compile(r"^([A-Za-z ]+)\n[-]+$", re.MULTILINE)


def snake_to_title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def capitalize_ioi(text: str) -> str:
    """Capitalize 'IOI' in text while preserving other formatting."""
    if not text:
        return text
    # Use word boundaries to avoid partial matches
    return re.sub(r'\bioi\b', 'IOI', text, flags=re.IGNORECASE)


def extract_sections_from_docstring(doc: str) -> dict[str, str]:
    """Parse simple NumPy-style sections (Parameters, Returns, Notes, Citation, etc.)."""
    if not doc:
        return {}
    # Normalize newlines and strip trailing spaces
    text = inspect.cleandoc(doc)
    # Find section headings (lines followed by dashes)
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        sections["Preamble"] = text.strip()
        return sections
    # Preamble before first section
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections["Preamble"] = preamble
    # Each section content until next heading
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[title] = body
    return sections


def determine_type_from_return_annotation(obj) -> str:
    # Default to Descriptor if no annotation available
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return "Descriptor"

    if ann is inspect.Signature.empty:
        return "Descriptor"

    # Builtin scalars considered descriptors
    scalar_types = (int, float, bool)
    # Common sequence-like types considered sequences
    sequence_type_names = {"list", "tuple", "dict", "set", "ndarray", "Series", "DataFrame"}

    # Direct type objects
    if isinstance(ann, type):
        return "Descriptor" if issubclass(ann, scalar_types) else "Sequence"

    # String annotations like 'list[int]' or 'float'
    if isinstance(ann, str):
        lowered = ann.lower()
        if any(t in lowered for t in ("int", "float", "bool")) and not any(t in lowered for t in ("list", "tuple", "dict", "set")):
            return "Descriptor"
        if any(t in lowered for t in ("list", "tuple", "dict", "set", "ndarray")):
            return "Sequence"
        return "Descriptor"

    # Typing objects (e.g., list[int])
    name = getattr(getattr(ann, "__origin__", None), "__name__", "") or getattr(ann, "_name", "") or str(ann)
    for seq_name in sequence_type_names:
        if seq_name in str(name):
            return "Sequence"
    return "Descriptor"


def collect_feature_rows(objs: Iterable[tuple[str, object]]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for name, obj in objs:
        # Skip get_ methods (these are wrapper functions)
        if name.startswith("get_"):
            continue
            
        # Include if it has a feature type decorator OR is a property (class feature)
        feature_types = getattr(obj, "_feature_types", None)
        is_property = isinstance(obj, property)
        if not feature_types and not is_property:
            continue

        # Name (handle class.property format)
        if "." in name:
            class_name, prop_name = name.split(".", 1)
            # Class display name adjustments
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
                class_display = ""  # Hide class name
            class_part = f"{class_display} " if class_display else ""
            pretty_name = f"{class_part}{snake_to_title(prop_name)}".strip()
        else:
            pretty_name = snake_to_title(name)

        # Implementations (from source decorators if present). For properties, try mapping by class name
        feature_sources = getattr(obj, "_feature_sources", [])
        if feature_sources:
            implementations = ", ".join(sorted({s.replace("_", " ").title() for s in feature_sources}))
        else:
            implementations = ""
            if is_property and "." in name:
                class_name = name.split(".", 1)[0]
                class_source_map = {
                    "StepContour": "fantastic",
                    "InterpolationContour": "fantastic",
                    "PolynomialContour": "fantastic",
                    "HuronContour": "fantastic",
                    "NGramCounter": "fantastic",
                }
                mapped = class_source_map.get(class_name)
                if mapped:
                    implementations = mapped.title()

        # Docstring sections
        # For properties, read the fget function docstring
        doc_string = inspect.getdoc(obj.fget) if is_property else inspect.getdoc(obj)
        sections = extract_sections_from_docstring(doc_string or "")
        description = capitalize_ioi(" ".join(sections.get("Preamble", "").split()))
        notes = capitalize_ioi(" ".join(sections.get("Note", "").split()))

        # Further References: only from Citation section in docstring (not decorator citations)
        citation_section = sections.get("Citation", "").strip()
        if citation_section:
            # Split by blank lines to avoid huge blocks, then normalize whitespace
            references = capitalize_ioi(" | ".join([" ".join(p.split()) for p in re.split(r"\n\s*\n", citation_section) if p.strip()]))
        else:
            references = ""

        # Type: Descriptor vs Sequence; for known properties, override
        if is_property:
            # Default
            type_label = "Descriptor"
            if "." in name:
                class_name, prop_name = name.split(".", 1)
                if class_name == "PolynomialContour" and prop_name == "coefficients":
                    type_label = "Sequence"  # list[float]
                if class_name == "HuronContour" and prop_name == "huron_contour":
                    type_label = "Descriptor"  # str
        else:
            type_label = determine_type_from_return_annotation(obj)

        rows.append(
            FeatureRow(
                name=pretty_name,
                implementations=implementations,
                references=references,
                description=description,
                type_label=type_label,
                notes=notes,
            )
        )
    return rows


def to_dataframe(rows: list[FeatureRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Sort by Name
    return df.sort_values("name").reset_index(drop=True)


def build_table() -> pd.DataFrame:
    members = inspect.getmembers(features_module)
    
    # Collect functions and class properties with feature type decorators
    all_features = []
    
    # Add functions
    functions = [(n, o) for n, o in members if inspect.isfunction(o)]
    all_features.extend(functions)
    
    # Add class properties
    # Explicitly inspect known feature classes whose properties should be exposed
    feature_classes = [StepContour, InterpolationContour, PolynomialContour, HuronContour, NGramCounter]
    excluded_properties = {"count_values", "freq_spec", "total_tokens"}
    
    for cls in feature_classes:
        class_name = cls.__name__
        for prop_name, prop_obj in inspect.getmembers(cls):
            if isinstance(prop_obj, property) and prop_name not in excluded_properties:
                all_features.append((f"{class_name}.{prop_name}", prop_obj))
    
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
        # Quarto Markdown file with interactive table
        qmd_df = df.rename(
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
            f.write("table_html = df_renamed.to_html(classes='table table-striped table-hover', table_id='features-table', escape=False, index=False)\n\n")
            f.write("# Add custom CSS and JavaScript for interactivity\n")
            f.write("interactive_html = '''\n")
            f.write("<style>\n")
            f.write(".search-container {\n")
            f.write("    margin-bottom: 20px;\n")
            f.write("}\n")
            f.write(".search-input {\n")
            f.write("    width: 100%;\n")
            f.write("    padding: 10px;\n")
            f.write("    border: 1px solid #ddd;\n")
            f.write("    border-radius: 4px;\n")
            f.write("    font-size: 16px;\n")
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
            f.write("/* Dynamic column sizing - let content determine width */\n")
            f.write("</style>\n")
            f.write("<div class='search-container'>\n")
            f.write("    <input type='text' class='search-input' id='searchInput' placeholder='Search features...'>\n")
            f.write("</div>\n")
            f.write("<div class='table-container'>\n")
            f.write("'''\n\n")
            f.write("display(HTML(interactive_html + table_html + '</div>'))\n\n")
            f.write("# Add JavaScript for search and sort functionality\n")
            f.write("display(HTML('''\n")
            f.write("<script>\n")
            f.write("document.addEventListener('DOMContentLoaded', function() {\n")
            f.write("    const table = document.getElementById('features-table');\n")
            f.write("    const searchInput = document.getElementById('searchInput');\n")
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
            f.write("    // Search functionality\n")
            f.write("    searchInput.addEventListener('input', function() {\n")
            f.write("        const searchTerm = this.value.toLowerCase();\n")
            f.write("        rows.forEach(row => {\n")
            f.write("            const text = row.textContent.toLowerCase();\n")
            f.write("            row.style.display = text.includes(searchTerm) ? '' : 'none';\n")
            f.write("        });\n")
            f.write("    });\n")
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


