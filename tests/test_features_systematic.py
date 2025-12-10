"""
Systematic test suite for features.py that validates every feature output 
against its function signature return type.

This test validates that get_all_features returns the correct types for all
features as specified in their function signatures.
"""

import pytest
import tempfile
import os
import numpy as np
import inspect
from typing import get_type_hints, get_origin, get_args

from melody_features.features import get_all_features, Config, IDyOMConfig, FantasticConfig
from melody_features.feature_decorators import FeatureType, FeatureDomain


def create_test_midi_file(pitches, starts, ends, tempo=120):
    """Create a temporary MIDI file for testing."""
    import mido

    # Create a new MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

    ticks_per_second = 480 * (tempo / 60)

    current_time = 0
    for i, (pitch, start, end) in enumerate(zip(pitches, starts, ends)):
        start_ticks = int(start * ticks_per_second)
        delta_time = start_ticks - current_time

        track.append(mido.Message('note_on', channel=0, note=pitch, velocity=64, time=delta_time))

        duration_ticks = int((end - start) * ticks_per_second)
        track.append(mido.Message('note_off', channel=0, note=pitch, velocity=64, time=duration_ticks))

        current_time = start_ticks + duration_ticks

    return mid


def _discover_feature_functions():
    """Dynamically discover all feature functions and their expected return types."""
    import sys
    import melody_features.features as features_module
    
    feature_functions = {}
    
    for name, obj in inspect.getmembers(features_module):
        if (inspect.isfunction(obj) and 
            hasattr(obj, '_feature_types') and 
            obj._feature_types):
            
            # Get return type annotation
            sig = inspect.signature(obj)
            return_annotation = sig.return_annotation
            
            # Convert type annotation to expected types
            expected_types = _convert_annotation_to_types(return_annotation)
            
            feature_functions[name] = {
                'function': obj,
                'expected_types': expected_types,
                'feature_types': obj._feature_types,
                'feature_domain': getattr(obj, '_feature_domain', None)
            }
    
    return feature_functions


def _convert_annotation_to_types(annotation):
    """Convert type annotation to tuple of expected types for validation."""
    if annotation == inspect.Parameter.empty:
        return (int, float, dict, list, str, np.integer, np.floating, np.ndarray)
    
    # Handle Union types
    if get_origin(annotation) is type(None) or get_origin(annotation) is type(None):
        # Handle Optional types
        args = get_args(annotation)
        if args:
            return _convert_annotation_to_types(args[0])
    
    # Handle basic types with numpy equivalents
    if annotation == int:
        return (int, np.integer, np.int64, np.int32, np.int16, np.int8)
    elif annotation == float:
        return (float, np.floating, np.float64, np.float32, np.float16)
    elif annotation == str:
        return (str,)
    elif annotation == dict:
        return (dict,)
    elif annotation == list:
        return (list, np.ndarray)
    elif annotation == bool:
        return (bool, np.bool_)
    
    # Handle generic types like list[float], dict[str, int], etc.
    if hasattr(annotation, '__origin__'):
        origin = annotation.__origin__
        if origin is list:
            return (list, np.ndarray)
        elif origin is dict:
            return (dict,)
        elif origin is tuple:
            return (tuple,)
    
    # Handle numpy types
    if hasattr(np, annotation.__name__):
        return (annotation,)
    
    # Default fallback - be more permissive
    return (int, float, dict, list, str, np.integer, np.floating, np.ndarray, 
            np.int64, np.int32, np.int16, np.int8, np.float64, np.float32, np.float16)


def _get_feature_category_mapping():
    """Get mapping from domains and types to category names."""
    return {
        # Domain-based mappings
        (FeatureDomain.PITCH, FeatureType.ABSOLUTE): 'pitch_features',
        (FeatureDomain.RHYTHM, FeatureType.TIMING): 'rhythm_features',
        (FeatureDomain.RHYTHM, FeatureType.INTERVAL): 'rhythm_features',
        # Type-based mappings
        FeatureType.INTERVAL: 'interval_features', 
        FeatureType.CONTOUR: 'contour_features',
        FeatureType.TONALITY: 'tonality_features',
        FeatureType.METRE: 'metre_features',
        FeatureType.EXPECTATION: 'expectation_features',
        FeatureType.COMPLEXITY: 'complexity_features',
        # back compatible mappings for corpus
        'corpus': 'corpus_features',
    }

def _internal_to_display_category(internal_name):
    """Map internal category names to display names (lowercase with underscores)."""
    mapping = {
        'pitch_features': 'absolute_pitch',
        'pitch_class_features': 'pitch_class',
        'interval_features': 'interval',
        'contour_features': 'contour',
        'rhythm_features': 'timing',  # ioi features map to 'inter_onset_interval'
        'tonality_features': 'tonality',
        'metre_features': 'metre',
        'expectation_features': 'expectation',
        'complexity_features': 'complexity',  # mtype features map to 'lexical_diversity'
        'corpus_features': 'corpus',
    }
    return mapping.get(internal_name, internal_name.lower().replace('_features', ''))

# Some features are actually allowed to be NaN
NAN_ALLOWED_FEATURES = {
    'lexical_diversity.yules_k',  
    'lexical_diversity.simpsons_d',  
    'lexical_diversity.sichels_s',  
    'lexical_diversity.honores_h',  
    'lexical_diversity.mean_entropy',  
    'lexical_diversity.mean_productivity',  
    'pitch_class.pitch_class_kurtosis_after_folding',  
    'pitch_class.pitch_class_skewness_after_folding',  
    'pitch_class.pitch_class_variability_after_folding',  
    'interval.standard_deviation_absolute_interval',
    'inter_onset_interval.ioi_standard_deviation',
    'interval.minor_major_third_ratio',
    'timing.variability_of_time_between_attacks',
}

PROPORTION_FEATURES = {
    'absolute_pitch.stepwise_motion',
    'absolute_pitch.repeated_notes',
    'interval.chromatic_motion',
    'interval.amount_of_arpeggiation',
    'expectation.melodic_embellishment',
    'timing.metric_stability',
    'tonality.tonalness',
    'interval.melodic_large_intervals',
    'tonality.proportion_conjunct_scalar',
    'tonality.proportion_scalar',
    'interval.prevalence_of_most_common_melodic_interval',
    'timing.equal_duration_transitions',
    'timing.dotted_duration_transitions',
    'timing.half_duration_transitions',
}


class TestFeatureTypeValidation:
    """Systematic validation of all feature types against function signatures."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = Config(
            idyom={
                "test": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=["cpint"],
                    ppm_order=1,
                    models=":stm"
                )
            },
            fantastic=FantasticConfig(max_ngram_order=3, phrase_gap=1.5),
            corpus=None
        )
        
        # Dynamically discover all feature functions
        self.feature_functions = _discover_feature_functions()
        self.category_mapping = _get_feature_category_mapping()
    
    def test_normal_melody_feature_types(self):
        """Test all feature types with a normal melody."""
        # Create a normal C major scale melody
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        ends = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
        
        midi_data = create_test_midi_file(pitches, starts, ends)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            midi_data.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            df = get_all_features(temp_path, config=self.config, skip_idyom=True)
            row = df.iloc[0]
            
            self._validate_feature_categories(row)
            
            self._validate_specific_feature_types(row, "normal melody")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_edge_case_two_notes_feature_types(self):
        """Test all feature types with minimal melody (two notes)."""
        pitches = [60, 67]
        starts = [0.0, 1.0]
        ends = [0.8, 1.8]
        
        midi_data = create_test_midi_file(pitches, starts, ends)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            midi_data.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            df = get_all_features(temp_path, config=self.config, skip_idyom=True)
            row = df.iloc[0]
            
            self._validate_feature_categories(row)
            
            self._validate_specific_feature_types(row, "two-note melody", allow_more_nans=True)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_edge_case_repeated_notes_feature_types(self):
        """Test all feature types with repeated notes."""
        pitches = [60, 60, 60, 60, 60]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0]
        ends = [0.4, 0.9, 1.4, 1.9, 2.4]
        
        midi_data = create_test_midi_file(pitches, starts, ends)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            midi_data.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            df = get_all_features(temp_path, config=self.config, skip_idyom=True)
            row = df.iloc[0]
            
            self._validate_feature_categories(row)
            
            self._validate_specific_feature_types(row, "repeated notes melody", allow_more_nans=True)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_edge_case_large_intervals_feature_types(self):
        """Test all feature types with large melodic intervals."""
        pitches = [36, 84, 24, 96, 48]
        starts = [0.0, 1.0, 2.0, 3.0, 4.0]
        ends = [0.8, 1.8, 2.8, 3.8, 4.8]
        
        midi_data = create_test_midi_file(pitches, starts, ends)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
            midi_data.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            df = get_all_features(temp_path, config=self.config, skip_idyom=True)
            row = df.iloc[0]
            
            self._validate_feature_categories(row)
            
            self._validate_specific_feature_types(row, "large intervals melody")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _validate_feature_categories(self, row):
        """Validate that feature categories match their expected types."""
        # Group features by category
        category_features = {}
        for col in row.index:
            if '.' in col and not col.startswith('idyom'):
                category = col.split('.')[0]
                if category not in category_features:
                    category_features[category] = {}
                category_features[category][col] = row[col]
        
        # Validate each category
        for category, features in category_features.items():
            for feature_name, value in features.items():
                # Skip None check for features that are allowed to be None
                if feature_name in NAN_ALLOWED_FEATURES:
                    continue
                assert value is not None, f"Feature {feature_name} should not be None"
    
    def _validate_specific_feature_types(self, row, test_case_name, allow_more_nans=False):
        """Validate specific features against their expected types."""
        for col in row.index:
            if col in ['melody_num', 'melody_id']:
                continue
            if col.startswith('idyom'):
                continue
                
            value = row[col]
            
            # Get expected types from function signature if available
            feature_name = col.split('.')[-1] if '.' in col else col
            expected_types = None
            
            if feature_name in self.feature_functions:
                expected_types = self.feature_functions[feature_name]['expected_types']
            
            # Skip None check for features that are allowed to be None
            if col not in NAN_ALLOWED_FEATURES:
                assert value is not None, f"Feature {col} should not be None in {test_case_name}"
                
                # Use expected types from function signature, or fallback to general types
                if expected_types:
                    # For numeric types, be more flexible - allow int/float cross-compatibility
                    if any(issubclass(t, (int, float, np.integer, np.floating)) for t in expected_types):
                        # If expecting numeric, allow any numeric type
                        if not isinstance(value, (int, float, np.integer, np.floating)):
                            assert False, f"Feature {col} has invalid type: {type(value)} in {test_case_name}, expected numeric type"
                    else:
                        assert isinstance(value, expected_types), \
                            f"Feature {col} has invalid type: {type(value)} in {test_case_name}, expected {expected_types}"
                else:
                    valid_types = (int, float, dict, list, str, np.integer, np.floating, np.ndarray)
                    assert isinstance(value, valid_types), \
                        f"Feature {col} has invalid type: {type(value)} in {test_case_name}"
            
            # Check for NaN/Inf in numeric features
            if isinstance(value, (int, float, np.integer, np.floating)):
                if col in NAN_ALLOWED_FEATURES:
                    # These features can legitimately be NaN
                    continue
                elif allow_more_nans and any(keyword in col for keyword in ['entropy', 'yules', 'simpsons', 'sichels', 'honores', 'std', 'deviation', 'variability', 'gradient']):
                    # Allow NaN for certain features in edge cases (including standard_deviation and variability)
                    continue
                else:
                    assert not (np.isnan(value) or np.isinf(value)), \
                        f"Feature {col} should not be NaN/Inf in {test_case_name}: {value}"
            
            if col in PROPORTION_FEATURES and isinstance(value, (int, float, np.integer, np.floating)):
                if not (np.isnan(value) or np.isinf(value)):
                    assert 0.0 <= value <= 1.0, \
                        f"Proportion feature {col} should be in [0,1] in {test_case_name}: {value}"
            
            if isinstance(value, dict):
                self._validate_dict_feature(col, value, test_case_name)
            
            if isinstance(value, (list, np.ndarray)):
                assert len(value) >= 0, f"List feature {col} should have non-negative length in {test_case_name}"
                
            if isinstance(value, str):
                assert len(value) > 0, f"String feature {col} should not be empty in {test_case_name}"

    
    def _validate_dict_feature(self, feature_name, feature_dict, test_case_name):
        """Validate dictionary-type features."""
        assert len(feature_dict) >= 0, f"Dict feature {feature_name} should have non-negative length in {test_case_name}"
        
        for key, value in feature_dict.items():
            assert value is not None, f"Dict feature {feature_name} should not have None values in {test_case_name}"

def test_all_features_comprehensive_validation():
    """Comprehensive test that validates all features systematically."""
    test_validator = TestFeatureTypeValidation()
    test_validator.setup_method()
    
    test_validator.test_normal_melody_feature_types()
    
    test_validator.test_edge_case_two_notes_feature_types()
    test_validator.test_edge_case_repeated_notes_feature_types()
    test_validator.test_edge_case_large_intervals_feature_types()


def test_feature_completeness():
    """Test that we're getting all expected features."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    ends = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
    
    midi_data = create_test_midi_file(pitches, starts, ends)
    
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
        midi_data.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        config = Config(
            idyom={
                "test": IDyOMConfig(
                    target_viewpoints=["cpitch"],
                    source_viewpoints=["cpint"],
                    ppm_order=1,
                    models=":stm"
                )
            },
            fantastic=FantasticConfig(max_ngram_order=3, phrase_gap=1.5),
            corpus=None
        )
        
        df = get_all_features(temp_path, config=config, skip_idyom=True)
        
        # Dynamically discover expected categories from feature functions
        feature_functions = _discover_feature_functions()
        category_mapping = _get_feature_category_mapping()
        
        # Get expected categories from discovered feature functions
        expected_categories = set()
        for func_info in feature_functions.values():
            domain = func_info.get('feature_domain')
            for feature_type in func_info['feature_types']:
                # Try domain+type combination first
                if domain and (domain, feature_type) in category_mapping:
                    expected_categories.add(category_mapping[(domain, feature_type)])
                # Fall back to just type
                elif feature_type in category_mapping:
                    expected_categories.add(category_mapping[feature_type])
                # Check for special string mappings
                elif feature_type in ['mtype', 'corpus']:
                    expected_categories.add(category_mapping.get(feature_type))
        
        # Remove corpus features from expected categories since no corpus is provided
        if config.corpus is None:
            expected_categories.discard('corpus_features')
        
        # Map internal category names to display names (lowercase with underscores)
        expected_display_categories = set()
        for internal_cat in expected_categories:
            display_cat = _internal_to_display_category(internal_cat)
            expected_display_categories.add(display_cat)
        
        # Also add special categories that may exist in the output
        # (lexical_diversity for mtype features, inter_onset_interval for IOI features)
        if 'complexity_features' in expected_categories:
            expected_display_categories.add('lexical_diversity')
        if 'rhythm_features' in expected_categories:
            expected_display_categories.add('inter_onset_interval')
        
        # Check that we have all expected feature categories
        feature_categories = set()
        for col in df.columns:
            if '.' in col and not col.startswith('idyom'):
                category = col.split('.')[0]
                feature_categories.add(category)
        
        missing_categories = expected_display_categories - feature_categories
        assert not missing_categories, f"Missing feature categories: {missing_categories}"
        
        # Check that we have a reasonable number of features
        feature_count = len([col for col in df.columns if '.' in col and not col.startswith('idyom')])
        expected_min_features = len(feature_functions) // 2  # At least half of discovered functions should be present
        assert feature_count >= expected_min_features, f"Expected at least {expected_min_features} features, got {feature_count}"
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])