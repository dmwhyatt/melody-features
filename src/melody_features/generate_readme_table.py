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

## Melody Features Summary

This table provides an overview of all {len(df)} melody features available in this package.

</div>

<div style="overflow-x: auto;">

{html_table}

</div>

### Feature Types
- **Descriptor**: Returns a single scalar value (int, float, bool)
- **Sequence**: Returns a collection (list, tuple, dict, etc.)
"""
    
    return styled_html

def update_readme_with_table():
    """Update README.md with the features table only if content has changed.
    
    Returns:
        tuple: (success: bool, changed: bool, message: str)
        - success: True if operation completed successfully, False if there was an error
        - changed: True if file was updated, False if no changes were needed
        - message: Description of what happened
    """
    
    try:
        # Read current README
        readme_path = Path("README.md")
        if not readme_path.exists():
            return False, False, "README.md not found!"
        
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
            
            # Extract current table content
            current_table_content = readme_content[start_idx + len(start_marker):end_idx - len(end_marker)].strip()
            
            new_content = (
                readme_content[:start_idx] + 
                start_marker + "\n" + 
                table_html + "\n" + 
                end_marker + 
                readme_content[end_idx:]
            )
            
            # Check if content has actually changed
            if current_table_content == table_html.strip():
                return True, False, "Features table is already up to date - no changes needed"
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
        return True, True, f"Updated README.md with features table ({len(df)} features)"
        
    except Exception as e:
        return False, False, f"Error updating README.md: {str(e)}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate features table for README.md")
    parser.add_argument("--update", action="store_true", help="Update README.md with the table")
    args = parser.parse_args()
    
    if args.update:
        success, changed, message = update_readme_with_table()
        print(message)
        
        if success:
            # Operation completed successfully (whether changes were made or not)
            sys.exit(0)
        else:
            # There was an error during the operation
            sys.exit(1)
    else:
        # Just print the table HTML
        print(generate_readme_table())
