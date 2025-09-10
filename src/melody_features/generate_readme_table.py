#!/usr/bin/env python3
"""
Generate an HTML table for the README.md file that displays all melody features.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent  # Go up one level from melody_features to src
sys.path.insert(0, str(src_dir))

from melody_features.quarto_table_build import build_table

def generate_readme_table():
    """Generate HTML table content for README.md."""
    
    # Build the table data
    df = build_table()
    
    # Rename columns for display
    df_renamed = df.rename(columns={
        'name': 'Name',
        'implementations': 'Pre-existing Implementations',
        'references': 'Further References',
        'description': 'Description',
        'type_label': 'Type',
        'notes': 'Notes'
    })
    
    # Generate HTML table with inline CSS for GitHub compatibility
    html_table = df_renamed.to_html(
        classes='features-table',
        table_id='features-table',
        escape=False,
        index=False,
        border=1
    )
    
    # Add inline CSS for styling (GitHub supports basic CSS)
    styled_html = f"""
<div align="center">

## 📊 Melody Features Summary

This table provides a comprehensive overview of all {len(df)} melody features available in this package.

</div>

<div style="overflow-x: auto;">

{html_table}

</div>

<div align="center">

### Feature Types
- **Descriptor**: Returns a single scalar value (int, float, bool)
- **Sequence**: Returns a collection (list, tuple, dict, etc.)

### Sources
- **FANTASTIC**: Müllensiefen, D. (2009). Feature ANalysis Technology Accessing STatistics
- **jSymbolic**: McKay, C., & Fujinaga, I. (2006). jSymbolic: A Feature Extractor for MIDI Files
- **IDyOM**: Pearce, M. T. (2005). The construction and evaluation of statistical models of melodic structure
- **MIDI Toolbox**: Eerola, T., & Toiviainen, P. (2004). MIDI Toolbox: MATLAB Tools for Music Research
- **Melsim**: Silas, S., & Frieler, K. (n.d.). Melsim: Framework for calculating tons of melodic similarities
- **Simile**: Müllensiefen, D., & Frieler, K. (2004). The Simile algorithms documentation
- **Novel**: Custom features introduced in this package

</div>
"""
    
    return styled_html

def update_readme_with_table():
    """Update README.md with the features table."""
    
    # Read current README
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("README.md not found!")
        return False
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Generate table HTML
    table_html = generate_readme_table()
    
    # Find the section to replace (look for existing table or add after Overview)
    if "<!-- FEATURES_TABLE_START -->" in readme_content and "<!-- FEATURES_TABLE_END -->" in readme_content:
        # Replace existing table
        start_marker = "<!-- FEATURES_TABLE_START -->"
        end_marker = "<!-- FEATURES_TABLE_END -->"
        start_idx = readme_content.find(start_marker)
        end_idx = readme_content.find(end_marker) + len(end_marker)
        
        new_content = (
            readme_content[:start_idx] + 
            start_marker + "\n" + 
            table_html + "\n" + 
            end_marker + 
            readme_content[end_idx:]
        )
    else:
        # Add table after Overview section
        overview_end = readme_content.find("Included in the package are contributions from:")
        if overview_end == -1:
            # Fallback: add at the end
            new_content = readme_content + "\n\n<!-- FEATURES_TABLE_START -->\n" + table_html + "\n<!-- FEATURES_TABLE_END -->"
        else:
            new_content = (
                readme_content[:overview_end] + 
                "\n\n<!-- FEATURES_TABLE_START -->\n" + 
                table_html + 
                "\n<!-- FEATURES_TABLE_END -->\n\n" + 
                readme_content[overview_end:]
            )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Get the number of features for the print statement
    df = build_table()
    print(f"Updated README.md with features table ({len(df)} features)")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate features table for README.md")
    parser.add_argument("--update", action="store_true", help="Update README.md with the table")
    args = parser.parse_args()
    
    if args.update:
        success = update_readme_with_table()
        if not success:
            sys.exit(1)
    else:
        # Just print the table HTML
        print(generate_readme_table())
