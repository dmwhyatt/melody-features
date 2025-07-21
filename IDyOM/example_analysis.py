#!/usr/bin/env python3
"""
Example usage of the IDyOM standalone analyzer
"""

from idyom_analyzer import IDyOMAnalyzer
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO)

def main():
    """Example analysis workflows"""

    # Example 1: Basic melody analysis
    print("=== Example 1: Basic Melody Analysis ===")
    analyzer = IDyOMAnalyzer(
        test_dataset_path="/Users/davidwhyatt/Downloads/fixed_mid/",
        pretrain_dataset_path="/Users/davidwhyatt/Downloads/fixed_mid/",
        output_dir="./results/melody_analysis",
        experiment_name="fixed_mid_example"
    )

    # Run melody analysis with default parameters
    results_path = analyzer.analyze_melody(
        target_viewpoints=['cpitch', 'onset'],
        source_viewpoints=['cpitch', 'onset', 'cpint'],
        models=':both',
        k=1,
        detail=3  # Use detail level 3 (level 1 is not implemented)
    )
    
    # Export results to CSV
    analyzer.export_results(results_path, format='csv')
    print(f"Melody analysis complete: {results_path}")
    
    # Example 2: Get experiment information
    print("\n=== Example 2: Experiment Information ===")
    
    exp_info = analyzer.get_experiment_info(results_path)
    print(f"Experiment has {len(exp_info.melodies_dict)} melodies")
    
    # List available output keywords
    if exp_info.melodies_dict:
        first_melody = next(iter(exp_info.melodies_dict.values()))
        available_keywords = first_melody.get_idyom_output_keyword_list()
        print(f"Available output keywords: {available_keywords[:10]}...")  # Show first 10
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 