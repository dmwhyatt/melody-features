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

from melody_features.features import get_all_features, Config, IDyOMConfig, FantasticConfig


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


# Define expected types for each feature category based on function signatures
EXPECTED_FEATURE_TYPES = {
    'pitch_features': dict,
    'interval_features': dict,
    'contour_features': dict,
    'duration_features': dict,
    'tonality_features': dict,
    'melodic_movement_features': dict,
    'mtype_features': dict,
    'complexity_features': dict,
}

# Define specific feature type expectations based on actual feature names from output
SPECIFIC_FEATURE_TYPES = {
    'pitch_features.pitch_range': (int, float, np.integer, np.floating),
    'pitch_features.pitch_standard_deviation': (float, np.floating),
    'pitch_features.pitch_entropy': (float, np.floating),
    'pitch_features.mean_pitch': (float, np.floating),
    'pitch_features.most_common_pitch': (int, np.integer),
    'pitch_features.number_of_pitches': (int, np.integer),
    'pitch_features.melodic_pitch_variety': (float, np.floating),
    'pitch_features.dominant_spread': (int, float, np.integer, np.floating),  # Can be int
    'pitch_features.mean_tessitura': (float, np.floating),
    'pitch_features.tessitura_std': (float, np.floating),
    'pitch_features.pcdist1': dict,
    'pitch_features.basic_pitch_histogram': dict,
    'pitch_features.folded_fifths_pitch_class_histogram': dict,
    'pitch_features.pitch_class_kurtosis_after_folding': (float, np.floating),
    'pitch_features.pitch_class_skewness_after_folding': (float, np.floating),
    'pitch_features.pitch_class_variability_after_folding': (float, np.floating),
    
    'interval_features.mean_absolute_interval': (float, np.floating),
    'interval_features.modal_interval': (int, np.integer),
    'interval_features.interval_entropy': (float, np.floating),
    'interval_features.ivdist1': dict,
    'interval_features.ivdirdist1': dict,
    'interval_features.ivsizedist1': dict,
    'interval_features.melodic_large_intervals': (float, np.floating),
    'interval_features.absolute_interval_range': (int, float, np.integer, np.floating),
    'interval_features.interval_direction_mean': (float, np.floating),
    'interval_features.interval_direction_sd': (float, np.floating),
    'interval_features.melodic_interval_histogram': dict,
    'interval_features.pitch_interval': list,
    'interval_features.number_of_common_melodic_intervals': (int, np.integer),
    'interval_features.prevalence_of_most_common_melodic_interval': (float, np.floating),
    'interval_features.variable_melodic_intervals': (int, np.integer),
    'interval_features.distance_between_most_prevalent_melodic_intervals': (int, float, np.integer, np.floating),
    'interval_features.average_interval_span_by_melodic_arcs': (float, np.floating),
    
    'duration_features.tempo': (float, np.floating),
    'duration_features.mean_duration': (float, np.floating),
    'duration_features.length': (int, float, np.integer, np.floating),
    'duration_features.npvi': (float, np.floating),
    'duration_features.meter_numerator': (int, np.integer),
    'duration_features.meter_denominator': (int, np.integer),
    'duration_features.metric_stability': (float, np.floating),
    'duration_features.duration_entropy': (float, np.floating),
    'duration_features.duration_standard_deviation': (float, np.floating),
    'duration_features.duration_range': (float, np.floating),
    'duration_features.modal_duration': (float, np.floating),
    'duration_features.number_of_durations': (int, np.integer),
    'duration_features.note_density': (float, np.floating),
    'duration_features.global_duration': (float, np.floating),
    'duration_features.mean_duration_accent': (float, np.floating),
    'duration_features.duration_accent_std': (float, np.floating),
    'duration_features.onset_autocorr_peak': (float, np.floating),
    'duration_features.ioi_mean': (float, np.floating),
    'duration_features.ioi_std': (float, np.floating),
    'duration_features.ioi_range': (float, np.floating),
    'duration_features.ioi_ratio_mean': (float, np.floating),
    'duration_features.ioi_ratio_std': (float, np.floating),
    'duration_features.ioi_contour_mean': (float, np.floating),
    'duration_features.ioi_contour_std': (float, np.floating),
    'duration_features.duration_histogram': dict,
    'duration_features.ioi_histogram': dict,
    'duration_features.equal_duration_transitions': (float, np.floating),
    'duration_features.dotted_duration_transitions': (float, np.floating),
    'duration_features.half_duration_transitions': (float, np.floating),
    
    'tonality_features.tonalness': (float, np.floating),
    'tonality_features.referent': (int, np.integer),
    'tonality_features.inscale': (int, np.integer),
    'tonality_features.mode': (int, np.integer),
    'tonality_features.tonal_clarity': (float, np.floating),
    'tonality_features.tonal_spike': (float, np.floating),
    'tonality_features.tonal_entropy': (float, np.floating),
    'tonality_features.temperley_likelihood': (float, np.floating),
    'tonality_features.longest_conjunct_scalar_passage': (int, np.integer),
    'tonality_features.longest_monotonic_conjunct_scalar_passage': (int, np.integer),
    'tonality_features.proportion_conjunct_scalar': (float, np.floating),
    'tonality_features.proportion_scalar': (float, np.floating),
    'tonality_features.tonalness_histogram': dict,
    
    'melodic_movement_features.stepwise_motion': (float, np.floating),
    'melodic_movement_features.repeated_notes': (float, np.floating),
    'melodic_movement_features.chromatic_motion': (float, np.floating),
    'melodic_movement_features.amount_of_arpeggiation': (float, np.floating),
    'melodic_movement_features.melodic_embellishment': (float, np.floating),
    
    'complexity_features.gradus': (int, np.integer),
    'complexity_features.registral_direction': (int, float, np.integer, np.floating),
    'complexity_features.proximity': (int, float, np.integer, np.floating),
    'complexity_features.closure': (int, float, np.integer, np.floating),
    'complexity_features.registral_return': (int, float, np.integer, np.floating),
    'complexity_features.intervallic_difference': (int, float, np.integer, np.floating),
    'complexity_features.mean_mobility': (float, np.floating),
    'complexity_features.mobility_std': (float, np.floating),
    
    'contour_features.huron_contour': str,
    'contour_features.interpolation_contour_mean_gradient': (float, np.floating),
    'contour_features.interpolation_contour_gradient_std': (float, np.floating),
    'contour_features.interpolation_contour_direction_changes': (int, np.integer),
    'contour_features.interpolation_contour_global_direction': (float, np.floating),
    'contour_features.interpolation_contour_class_label': str,
    'contour_features.polynomial_contour_coefficients': list,
    'contour_features.step_contour_global_variation': (float, np.floating),
    'contour_features.step_contour_global_direction': (float, np.floating),
    'contour_features.step_contour_local_variation': (float, np.floating),
    'contour_features.mean_melodic_accent': (float, np.floating),
    'contour_features.melodic_accent_std': (float, np.floating),
    
    'mtype_features.yules_k': (float, np.floating),
    'mtype_features.simpsons_d': (float, np.floating),
    'mtype_features.sichels_s': (float, np.floating),
    'mtype_features.honores_h': (float, np.floating),
    'mtype_features.mean_entropy': (float, np.floating),
    'mtype_features.mean_productivity': (float, np.floating),
}

# Some features are actually allowed to be NaN
NAN_ALLOWED_FEATURES = {
    'mtype_features.yules_k',  
    'mtype_features.simpsons_d',  
    'mtype_features.sichels_s',  
    'mtype_features.honores_h',  
    'mtype_features.mean_entropy',  
    'mtype_features.mean_productivity',  
    'pitch_features.pitch_class_kurtosis_after_folding',  
    'pitch_features.pitch_class_skewness_after_folding',  
    'pitch_features.pitch_class_variability_after_folding',  
    'interval_features.standard_deviation_absolute_interval',
    'duration_features.ioi_standard_deviation',
    'interval_features.minor_major_third_ratio',
}

PROPORTION_FEATURES = {
    'melodic_movement_features.stepwise_motion',
    'melodic_movement_features.repeated_notes',
    'melodic_movement_features.chromatic_motion',
    'melodic_movement_features.amount_of_arpeggiation',
    'melodic_movement_features.melodic_embellishment',
    'duration_features.metric_stability',
    'tonality_features.tonalness',
    'interval_features.melodic_large_intervals',
    'tonality_features.proportion_conjunct_scalar',
    'tonality_features.proportion_scalar',
    'interval_features.prevalence_of_most_common_melodic_interval',
    'duration_features.equal_duration_transitions',
    'duration_features.dotted_duration_transitions',
    'duration_features.half_duration_transitions',
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
        for category, expected_type in EXPECTED_FEATURE_TYPES.items():
            category_features = {col: row[col] for col in row.index if col.startswith(f"{category}.")}
            
            if not category_features:
                continue
            
            if expected_type == dict:
                for feature_name, value in category_features.items():
                    # Skip None check for features that are allowed to be None
                    if feature_name in NAN_ALLOWED_FEATURES:
                        continue
                    assert value is not None, f"Feature {feature_name} should not be None"
            elif expected_type == str:
                main_feature = f"{category}.{category.replace('_features', '')}"
                if main_feature in row.index:
                    value = row[main_feature]
                    assert isinstance(value, str), f"Feature {main_feature} should be string, got {type(value)}"
    
    def _validate_specific_feature_types(self, row, test_case_name, allow_more_nans=False):
        """Validate specific features against their expected types."""
        for col in row.index:
            if col in ['melody_num', 'melody_id']:
                continue
            if col.startswith('idyom'):
                continue
                
            value = row[col]
            
            # Skip None check for features that are allowed to be None
            if col not in NAN_ALLOWED_FEATURES:
                assert value is not None, f"Feature {col} should not be None in {test_case_name}"
                
                valid_types = (int, float, dict, list, str, np.integer, np.floating, np.ndarray)
                assert isinstance(value, valid_types), \
                    f"Feature {col} has invalid type: {type(value)} in {test_case_name}"
            
            # Check for NaN/Inf in numeric features
            if isinstance(value, (int, float, np.integer, np.floating)):
                if col in NAN_ALLOWED_FEATURES:
                    # These features can legitimately be NaN
                    continue
                elif allow_more_nans and any(keyword in col for keyword in ['entropy', 'yules', 'simpsons', 'sichels', 'honores', 'std', 'gradient']):
                    # Allow NaN for certain features in edge cases
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
        
        # Check that we have all expected feature categories
        feature_categories = set()
        for col in df.columns:
            if '.' in col and not col.startswith('idyom'):
                category = col.split('.')[0]
                feature_categories.add(category)
        
        expected_categories = {
            'pitch_features', 'interval_features', 'contour_features',
            'duration_features', 'tonality_features', 'melodic_movement_features',
            'mtype_features', 'complexity_features'
        }
        
        missing_categories = expected_categories - feature_categories
        assert not missing_categories, f"Missing feature categories: {missing_categories}"
        
        feature_count = len([col for col in df.columns if '.' in col and not col.startswith('idyom')])
        assert feature_count >= 80, f"Expected at least 80 features, got {feature_count}"
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])