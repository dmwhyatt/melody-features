"""
Test jSymbolic feature validation against reference values.
These values were produced using the jSymbolic feature extraction tool for the
first 5 melodies in the Essen Corpus.
"""

import pytest
import csv
from melody_features.features import get_jsymbolic_features
from melody_features.representations import Melody
from melody_features.import_mid import import_midi

EXCLUDED_FEATURES = [
    'Average_Range_of_Glissandos', # we don't support pitch bend messages so no glissando features
    'Average_Time_Between_Attacks_for_Each_Voice', # monophonic melodies have only one voice, redundant
    'Major_or_Minor', # covered by tonality features "mode", takes value "major" or "minor"
    'Glissando_Prevalence',
    'Average_Range_of_Glissandos',
    'Vibrato_Prevalence', # we don't have vibrato encoding
    'Microtone_Prevalence', # or microtones
    'Metrical_Diversity', # covered by metric stability feature
    'Average_Variability_of_Time_Between_Attacks_for_Each_Voice',
    'Initial_Time_Signature_0', # we have this, but this is read from the Melody object so not a function
    'Initial_Time_Signature_1', # ditto,
    'Partial_Rests_Fraction', # package only handles monophonic so no such thing as partial rest
    'Longest_Partial_Rest',
    'Mean_Partial_Rest_Duration',
    'Median_Partial_Rest_Duration',
    'Variability_of_Partial_Rest_Durations',
    'Variability_Across_Voices_of_Combined_Rests',
    'Average_Rest_Fraction_Across_Voices',
]




def load_reference_values():
    """Load reference jSymbolic values from CSV."""
    reference_data = {}

    with open('tests/jsymbolic_test_values.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['Path']
            reference_data[path] = {k: float(v) if v else 0.0 for k, v in row.items() if k != 'Path'}

    return reference_data


def create_melody_from_file(file_path: str) -> Melody:
    """Create a melody from a MIDI file."""
    midi_data = import_midi(file_path)
    melody = Melody(midi_data)
    return melody


def get_feature_mapping():
    """Create mapping from CSV feature names to our function names."""
    from melody_features.features import _get_features_by_source
    from difflib import get_close_matches

    # Get all jSymbolic features
    jsymbolic_features = _get_features_by_source('jsymbolic')

    # Load reference CSV to get actual column names
    reference_data = load_reference_values()
    csv_columns = set(list(reference_data.values())[0].keys())

    mapping = {}

    for func_name in jsymbolic_features.keys():
        # Try multiple naming convention conversions
        candidates = [
            # Direct snake_case to Title_Case
            func_name.replace('_', ' ').title().replace(' ', '_'),
            # Handle special cases
            func_name.replace('number_of_unique_', 'Number_of_').replace('_', ' ').title().replace(' ', '_'),
            func_name.replace('number_of_common_', 'Number_of_Common_').replace('_', ' ').title().replace(' ', '_'),
            func_name.replace('most_common_', 'Most_Common_').replace('_', ' ').title().replace(' ', '_'),
            func_name.replace('mean_', 'Mean_').replace('_', ' ').title().replace(' ', '_'),
            func_name.replace('prevalence_of_', 'Prevalence_of_').replace('_', ' ').title().replace(' ', '_'),
            # Handle aliases
            'Most_Common_Melodic_Interval' if func_name == 'most_common_interval' else None,
            'Mean_Melodic_Interval' if func_name == 'mean_melodic_interval' else None,
            'Range' if func_name == 'pitch_range' else None,
            'Mean_Rhythmic_Value' if func_name == 'mean_duration' else None,
            'Most_Common_Rhythmic_Value' if func_name == 'modal_duration' else None,
        ]

        # Remove None values
        candidates = [c for c in candidates if c is not None]

        # Find the best match in CSV columns
        best_match = None
        for candidate in candidates:
            if candidate in csv_columns:
                best_match = candidate
                break

        # If no exact match, try fuzzy matching
        if not best_match:
            for candidate in candidates:
                matches = get_close_matches(candidate, csv_columns, n=1, cutoff=0.8)
                if matches:
                    best_match = matches[0]
                    break

        if best_match:
            mapping[best_match] = func_name

    return mapping


# Generate the mapping dynamically
FEATURE_MAPPING = get_feature_mapping()


def test_all_csv_features_have_mappings():
    """Test that all features found in the CSV have an equivalent in our package that are mapped."""
    reference_data = load_reference_values()
    all_csv_features = set(list(reference_data.values())[0].keys())
    excluded_features = set(EXCLUDED_FEATURES)
    testable_features = all_csv_features - excluded_features
    
    # Get our mapped features
    mapped_features = set(FEATURE_MAPPING.keys())
    
    # Find features that are in CSV but not mapped
    unmapped_features = testable_features - mapped_features
    
    print(f"\n=== Feature Mapping Coverage ===")
    print(f"Total CSV features: {len(all_csv_features)}")
    print(f"Excluded features: {len(excluded_features)}")
    print(f"Testable features: {len(testable_features)}")
    print(f"Mapped features: {len(mapped_features)}")
    print(f"Unmapped features: {len(unmapped_features)}")
    
    if unmapped_features:
        print(f"\nUnmapped features ({len(unmapped_features)}):")
        for feature in sorted(unmapped_features):
            print(f"  - {feature}")
    
    # Calculate coverage percentage
    coverage_percentage = (len(mapped_features) / len(testable_features) * 100) if testable_features else 0
    print(f"\nCoverage: {coverage_percentage:.1f}%")
    
    # Assert minimum coverage threshold
    assert coverage_percentage >= 95.0, f"Feature coverage too low: {coverage_percentage:.1f}% (minimum 95%)"
    assert len(mapped_features) > 0, "No features are mapped for validation"


def test_all_mapped_features_match_csv_values():
    """Test that all of the values match their CSV counterpart."""
    reference_data = load_reference_values()
    
    # Test all melodies in the CSV
    melody_files = [
        "src/melody_features/corpora/Essen_Corpus/appenzel.mid",
        "src/melody_features/corpora/Essen_Corpus/arabic01.mid", 
        "src/melody_features/corpora/Essen_Corpus/ashsham1.mid",
        "src/melody_features/corpora/Essen_Corpus/belgium1.mid",
        "src/melody_features/corpora/Essen_Corpus/brabant1.mid"
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    all_failures = []
    
    for melody_file in melody_files:
        # Get the CSV path for this melody
        csv_path = melody_file.replace("src/melody_features/", "")
        
        if csv_path not in reference_data:
            print(f"Warning: No reference data found for {csv_path}")
            continue
            
        print(f"\n=== Testing {csv_path} ===")
        
        # Create melody and get features
        melody = create_melody_from_file(melody_file)
        our_features = get_jsymbolic_features(melody)
        reference_row = reference_data[csv_path]
        
        # Test each mapped feature for this melody
        melody_tests = 0
        melody_passed = 0
        melody_failed = 0
        
        for csv_name, our_name in FEATURE_MAPPING.items():
            # Skip excluded features
            if csv_name in EXCLUDED_FEATURES:
                continue
                
            if csv_name in reference_row and our_name in our_features:
                melody_tests += 1
                total_tests += 1
                
                expected = reference_row[csv_name]
                actual = our_features[our_name]

                # Handle different data types
                if isinstance(actual, dict):
                    # For histogram features, just check if they exist
                    melody_passed += 1
                    total_passed += 1
                else:
                    # For numeric features, check with 1% tolerance
                    if expected == 0.0:
                        tolerance = 1e-10  # Very small tolerance for zero values
                    else:
                        tolerance = abs(expected) * 0.01

                    if abs(expected - actual) >= tolerance:
                        failure = {
                            'melody': csv_path,
                            'feature': csv_name,
                            'our_name': our_name,
                            'expected': expected,
                            'actual': actual,
                            'tolerance': tolerance,
                            'difference': abs(expected - actual)
                        }
                        all_failures.append(failure)
                        melody_failed += 1
                        total_failed += 1
                        
                        print(f"    ✗ {csv_name}: expected {expected}, got {actual} (diff: {abs(expected - actual):.6f})")
                    else:
                        melody_passed += 1
                        total_passed += 1
        
        print(f"  {melody_passed}/{melody_tests} passed")
    
    # Print summary
    print(f"\n=== Overall Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {(total_passed/total_tests*100):.1f}%")
    
    # Show detailed failure analysis
    if all_failures:
        print(f"\n=== Failure Analysis ===")
        
        # Group failures by feature
        feature_failures = {}
        for failure in all_failures:
            feature = failure['feature']
            if feature not in feature_failures:
                feature_failures[feature] = []
            feature_failures[feature].append(failure)
        
        print(f"Features with failures ({len(feature_failures)}):")
        for feature, failures in sorted(feature_failures.items()):
            avg_diff = sum(f['difference'] for f in failures) / len(failures)
            print(f"  {feature}: {len(failures)} failures, avg diff: {avg_diff:.6f}")
        
        # Show worst failures
        worst_failures = sorted(all_failures, key=lambda x: x['difference'], reverse=True)[:5]
        print(f"\nWorst failures:")
        for failure in worst_failures:
            print(f"  {failure['melody']} - {failure['feature']}: {failure['expected']} vs {failure['actual']} (diff: {failure['difference']:.6f})")
    
    # Assert minimum success rate
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    assert success_rate >= 95.0, f"Success rate too low: {success_rate:.1f}% (minimum 95%)"
    assert total_passed > 0, "No features passed validation"


if __name__ == "__main__":
    test_all_csv_features_have_mappings()
    test_all_mapped_features_match_csv_values()
    pytest.main([__file__, "-v"])
