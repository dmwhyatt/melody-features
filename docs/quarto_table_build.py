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

The metadata collection itself (sources, categories, descriptions, notes,
references, etc.) lives in `melody_features.feature_metadata` so that the
docs table and the package-level `get_feature_metadata()` API share a single
source of truth. This script only adds Quarto/HTML-specific rendering on
top of that shared data.

Usage:
  python docs/quarto_table_build.py --format csv --out /path/to/features_table.csv
  python docs/quarto_table_build.py --format qmd --out /path/to/features_table.qmd
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from melody_features.feature_metadata import (  # noqa: E402
    build_table,
    count_features,
)


def format_implementations_html(implementations: str) -> str:
    """Render implementation sources as compact badges."""
    if not implementations:
        return ""
    tokens = [t.strip() for t in implementations.split(",") if t.strip()]
    badges = []
    for token in tokens:
        css_class = _IMPL_BADGE_CLASSES.get(token, "impl-default")
        badges.append(f'<span class="impl-badge {css_class}">{token}</span>')
    return '<span class="impl-badges">' + " ".join(badges) + "</span>"


_IMPL_BADGE_CLASSES: dict[str, str] = {
    "FANTASTIC": "impl-fantastic",
    "jSymbolic": "impl-jsymbolic",
    "IDyOM": "impl-idyom",
    "MIDI Toolbox": "impl-midi-toolbox",
    "MUST": "impl-must",
    "SIMILE": "impl-simile",
    "Melsim": "impl-melsim",
    "Novel": "impl-novel",
    "Partitura": "impl-partitura",
}


def format_type_badge_html(type_label: str) -> str:
    """Render Descriptor / Sequence as a colored pill."""
    if not type_label:
        return ""
    css_class = "type-descriptor" if type_label == "Descriptor" else "type-sequence"
    return f'<span class="type-badge {css_class}">{type_label}</span>'


def format_references_html(references: str) -> str:
    """Render citation strings as inline text or a compact list."""
    if not references:
        return ""
    parts = [p.strip() for p in references.split(" | ") if p.strip()]
    if len(parts) == 1:
        return f'<span class="citation-inline">{parts[0]}</span>'
    items = "".join(f"<li>{part}</li>" for part in parts)
    return f'<ul class="citation-list">{items}</ul>'


def format_notes_html(notes: str) -> str:
    """Improve readability of notes: code tokens, implementation names, emphasis."""
    if not notes:
        return ""

    text = notes

    def _code_token(match: re.Match[str]) -> str:
        return f"<code>{match.group(1)}</code>"

    # Docstring / alias patterns
    text = re.sub(
        r'\bThis is called\s+([a-z][a-z0-9_]*)\s+in\s+([^.;]+)',
        r'This is called <code>\1</code> in <span class="impl-ref">\2</span>',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\bThis feature is named\s+"([^"]+)"\s+in\s+([^.;]+)',
        r'This feature is named <strong>\1</strong> in <span class="impl-ref">\2</span>',
        text,
    )
    text = re.sub(
        r'\bnamed\s+"([^"]+)"\s+in\s+([^.;]+)',
        r'named <strong>\1</strong> in <span class="impl-ref">\2</span>',
        text,
    )

    # Backtick-style identifiers already in docstrings
    text = re.sub(r"`([^`]+)`", _code_token, text)

    # Bare snake_case identifiers (e.g. variability_of_time_between_attacks)
    text = re.sub(
        r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b",
        _code_token,
        text,
    )

    return f'<span class="feature-notes">{text}</span>'


def format_description_html(description: str) -> str:
    """Wrap description text for consistent table typography."""
    if not description:
        return ""
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", description)
    return f'<span class="feature-description">{text}</span>'


def format_name_html(name: str, source_url: str) -> str:
    """Render the feature name as a link to its source definition, if known."""
    if source_url:
        return (
            f'<a class="feature-name-link" href="{source_url}" target="_blank" '
            f'rel="noopener noreferrer">{name}</a>'
        )
    return f'<span class="feature-name-text">{name}</span>'


def format_table_display_html(df: pd.DataFrame) -> pd.DataFrame:
    """Apply HTML formatting to columns shown in the Quarto feature table."""
    display = df.copy()
    name_col = "Name" if "Name" in display.columns else "name"
    if name_col in display.columns and "source_url" in display.columns:
        display[name_col] = display.apply(
            lambda r: format_name_html(r[name_col], r.get("source_url", "")), axis=1
        )
    if "Pre-existing Implementations" in display.columns:
        display["Pre-existing Implementations"] = display["Pre-existing Implementations"].map(
            lambda v: format_implementations_html(v if isinstance(v, str) else "")
        )
    if "Type" in display.columns:
        display["Type"] = display["Type"].map(
            lambda v: format_type_badge_html(v if isinstance(v, str) else "")
        )
    if "Notes" in display.columns:
        display["Notes"] = display["Notes"].map(
            lambda v: format_notes_html(v if isinstance(v, str) else "")
        )
    if "Further References" in display.columns:
        display["Further References"] = display["Further References"].map(
            lambda v: format_references_html(v if isinstance(v, str) else "")
        )
    if "Description" in display.columns:
        display["Description"] = display["Description"].map(
            lambda v: format_description_html(v if isinstance(v, str) else "")
        )
    return display


def main():
    parser = argparse.ArgumentParser(description="Build a feature summary table for Quarto.")
    parser.add_argument("--format", choices=["csv", "qmd"], default="qmd", help="Output format")
    parser.add_argument("--out", required=True, help="Output file path")
    args = parser.parse_args()

    df = build_table()
    feature_count = len(df)

    if args.format == "csv":
        display_df = df.rename(
            columns={
                "python_name": "Python Name",
                "name": "Name",
                "implementations": "Pre-existing Implementations",
                "references": "Further References",
                "description": "Description",
                "type_label": "Type",
                "notes": "Notes",
            }
        )
        display_df.drop(
            columns=["source_url", "feature_types", "has_corpus_stats_param"], errors="ignore"
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
            f.write(
                f"This table provides a comprehensive overview of all **{feature_count}** melody features "
                "available in this package.\n\n"
            )
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
            f.write("from melody_features.feature_metadata import build_table\n")
            f.write("from quarto_table_build import format_table_display_html\n\n")
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
            f.write("df_display = df_renamed.drop(columns=['category', 'domain', 'data-category', 'data-domain', 'sort_name', 'python_name', 'feature_types', 'has_corpus_stats_param'], errors='ignore')\n")
            f.write("df_display = format_table_display_html(df_display)\n")
            f.write("df_display = df_display.drop(columns=['source_url'], errors='ignore')\n")
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
            f.write("domains = ['Pitch', 'Rhythm', 'Pitch & Rhythm']\n")
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
            f.write("        elif token_low in ['must']:\n")
            f.write("            impl_tokens.add('MUST')\n")
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
            f.write("    width: 22%;\n")
            f.write("}\n")
            f.write("#features-table td:first-child a.feature-name-link {\n")
            f.write("    color: #0d6efd;\n")
            f.write("    font-weight: 600;\n")
            f.write("    text-decoration: none;\n")
            f.write("    white-space: normal;\n")
            f.write("    word-break: keep-all;\n")
            f.write("    hyphens: auto;\n")
            f.write("}\n")
            f.write("#features-table td:first-child a.feature-name-link:hover {\n")
            f.write("    text-decoration: underline;\n")
            f.write("}\n")
            f.write("#features-table td:first-child .feature-name-text {\n")
            f.write("    font-weight: 600;\n")
            f.write("}\n")
            f.write("/* Implementation badges */\n")
            f.write(".impl-badges { display: flex; flex-wrap: wrap; gap: 0.35rem; }\n")
            f.write(".impl-badge {\n")
            f.write("    display: inline-block;\n")
            f.write("    padding: 0.15rem 0.5rem;\n")
            f.write("    border-radius: 999px;\n")
            f.write("    font-size: 0.78rem;\n")
            f.write("    font-weight: 600;\n")
            f.write("    line-height: 1.3;\n")
            f.write("    white-space: nowrap;\n")
            f.write("    border: 1px solid transparent;\n")
            f.write("}\n")
            f.write(".impl-fantastic { background: #e8f4ea; color: #1b5e20; border-color: #c8e6c9; }\n")
            f.write(".impl-jsymbolic { background: #e3f2fd; color: #0d47a1; border-color: #bbdefb; }\n")
            f.write(".impl-idyom { background: #f3e5f5; color: #4a148c; border-color: #e1bee7; }\n")
            f.write(".impl-midi-toolbox { background: #fff3e0; color: #e65100; border-color: #ffe0b2; }\n")
            f.write(".impl-must { background: #e0f2f1; color: #004d40; border-color: #b2dfdb; }\n")
            f.write(".impl-simile { background: #fce4ec; color: #880e4f; border-color: #f8bbd0; }\n")
            f.write(".impl-default { background: #f1f3f5; color: #495057; border-color: #dee2e6; }\n")
            f.write("/* Type pills */\n")
            f.write(".type-badge {\n")
            f.write("    display: inline-block;\n")
            f.write("    padding: 0.2rem 0.55rem;\n")
            f.write("    border-radius: 0.35rem;\n")
            f.write("    font-size: 0.8rem;\n")
            f.write("    font-weight: 600;\n")
            f.write("    letter-spacing: 0.02em;\n")
            f.write("}\n")
            f.write(".type-descriptor { background: #eef2ff; color: #3730a3; }\n")
            f.write(".type-sequence { background: #ecfdf5; color: #065f46; }\n")
            f.write("/* Notes, description, references */\n")
            f.write(".feature-description { color: #212529; line-height: 1.45; }\n")
            f.write(".feature-notes {\n")
            f.write("    display: block;\n")
            f.write("    color: #495057;\n")
            f.write("    font-size: 0.92rem;\n")
            f.write("    line-height: 1.45;\n")
            f.write("}\n")
            f.write(".feature-description code,\n")
            f.write(".feature-notes code {\n")
            f.write("    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;\n")
            f.write("    font-size: 0.85em;\n")
            f.write("    padding: 0.1rem 0.35rem;\n")
            f.write("    border-radius: 0.25rem;\n")
            f.write("    background: #f8f9fa;\n")
            f.write("    border: 1px solid #e9ecef;\n")
            f.write("    color: #c7254e;\n")
            f.write("    word-break: break-word;\n")
            f.write("}\n")
            f.write(".feature-notes .impl-ref { font-weight: 600; color: #343a40; }\n")
            f.write(".citation-inline {\n")
            f.write("    display: inline;\n")
            f.write("    font-size: 0.9rem;\n")
            f.write("    color: #495057;\n")
            f.write("    line-height: 1.35;\n")
            f.write("    white-space: normal;\n")
            f.write("}\n")
            f.write(".citation-list {\n")
            f.write("    margin: 0;\n")
            f.write("    padding-left: 1.1rem;\n")
            f.write("    font-size: 0.88rem;\n")
            f.write("    color: #6c757d;\n")
            f.write("    line-height: 1.4;\n")
            f.write("}\n")
            f.write(".citation-list li { margin-bottom: 0.2rem; }\n")
            f.write("/* Column widths */\n")
            f.write("#features-table th:nth-child(2), #features-table td:nth-child(2) { min-width: 11rem; }\n")
            f.write("#features-table th:nth-child(3), #features-table td:nth-child(3) { min-width: 9.5rem; }\n")
            f.write("#features-table th:nth-child(4), #features-table td:nth-child(4) { min-width: 16rem; }\n")
            f.write("#features-table th:nth-child(6), #features-table td:nth-child(6) { min-width: 18rem; }\n")
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
            f.write("            // 'both' domain features only appear when 'Pitch & Rhythm' is selected\n")
            f.write("            let matchesDomain = true;\n")
            f.write("            if (selectedDomain) {\n")
            f.write("                if (selectedDomain === 'Pitch') {\n")
            f.write("                    matchesDomain = domain === 'pitch' || domain === 'pitch,rhythm';\n")
            f.write("                } else if (selectedDomain === 'Rhythm') {\n")
            f.write("                    matchesDomain = domain === 'rhythm' || domain === 'pitch,rhythm';\n")
            f.write("                } else if (selectedDomain === 'Pitch & Rhythm') {\n")
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
            f.write("- **MUST**: Clemente, A., Vila-Vidal, M., Pearce, M. T., et al. (2020). A Set of 200 Musical Stimuli Varying in Balance, Contour, Symmetry, and Complexity\n")
            f.write("- **Melsim**: Silas, S., & Frieler, K. (n.d.). Melsim: Framework for calculating tons of melodic similarities\n")
            f.write("- **Simile**: Müllensiefen, D., & Frieler, K. (2004). The Simile algorithms documentation 0.3\n")
            f.write("- **Novel**: Custom features introduced in this package\n\n")
            f.write("## Feature Types\n\n")
            f.write("- **Descriptor**: Returns a single scalar value (int, float, bool)\n")
            f.write("- **Sequence**: Returns a collection (list, tuple, dict, etc.)\n")

    print(f"Built table with {feature_count} features -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
