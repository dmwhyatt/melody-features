"""
Pytest test suite for meter estimation functionality.

Tests the meter estimation functions from meter_estimation.py
using various synthetic and real MIDI examples.
"""

# I am not entirely convinved by the meter estimation employed in the MIDI toolbox,
# so this test suite is more focused on making sure that things work as expected
# than validating the meter estimation itself.

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from melody_features.meter_estimation import (
    estimate_meter,
    estimate_meter_simple,
    estimate_meter_optimal,
    meter_to_time_signature,
    duration_accent,
    melodic_accent,
    onset_autocorrelation_with_accents
)


def test_meter_estimation_returns_valid_values():
    """Test that meter estimation functions return valid meter values."""
    # Create a simple pattern
    starts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ends = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    pitches = [60, 64, 67, 72, 60, 64, 67, 72]  # C major arp
    
    # Test that all estimation methods return valid meter values (2 or 3)
    simple_meter = estimate_meter_simple(starts, ends)
    assert simple_meter in [2, 3], f"Simple estimation should return 2 or 3, got {simple_meter}"
    
    optimal_meter = estimate_meter_optimal(starts, ends, pitches)
    assert optimal_meter in [2, 3], f"Optimal estimation should return 2 or 3, got {optimal_meter}"
    
    main_meter = estimate_meter(starts, ends, pitches, use_optimal=True)
    assert main_meter in [2, 3], f"Main function should return 2 or 3, got {main_meter}"
    
    # Test time signature conversion works for both valid meter types
    time_sig = meter_to_time_signature(main_meter)
    assert time_sig in [(4, 4), (3, 4)], f"Time signature should be (4,4) or (3,4), got {time_sig}"


def test_accent_functions():
    """Test duration and melodic accent calculations."""
    # Test data with varying durations
    starts = [0.0, 1.0, 2.0, 3.0]
    ends = [0.5, 1.2, 2.8, 3.3]  # Varying durations: 0.5, 0.2, 0.8, 0.3
    pitches = [60, 67, 64, 60]  # Up-down-down pattern (peak at index 1)
    
    # Test duration accents
    dur_accents = duration_accent(starts, ends)
    assert len(dur_accents) == 4, f"Should have 4 duration accents, got {len(dur_accents)}"
    assert all(0.0 <= accent <= 1.0 for accent in dur_accents), "All duration accents should be between 0 and 1"
    
    # Longer durations should generally have higher accents (note at index 2 has longest duration)
    longest_duration_idx = 2
    assert dur_accents[longest_duration_idx] > 0, "Longest duration should have positive accent"
    
    mel_accents = melodic_accent(pitches)
    assert len(mel_accents) == 4, f"Should have 4 melodic accents, got {len(mel_accents)}"
    assert all(0.0 <= accent <= 1.0 for accent in mel_accents), "All melodic accents should be between 0 and 1"
    
    assert mel_accents[0] == 1.0, f"First note should have accent 1.0, got {mel_accents[0]}"
    
    # Peak note (index 1: 60->67->64) should have high accent
    assert mel_accents[1] > 0.5, f"Peak note should have high accent, got {mel_accents[1]}"


def test_autocorrelation():
    """Test onset autocorrelation with different accent types."""
    # Create regular pattern for clear autocorrelation
    starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # Eighth notes
    ends = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    
    # Test with duration accents
    autocorr_dur = onset_autocorrelation_with_accents(
        starts, ends, pitches, accent_type='duration', max_lag_quarters=2
    )
    assert len(autocorr_dur) == 9, f"Should have 9 autocorr values (0-8 divisions), got {len(autocorr_dur)}"
    assert autocorr_dur[0] == 1.0, f"Lag 0 should be 1.0 (normalized), got {autocorr_dur[0]}"
    assert all(-1.0 <= val <= 1.0 for val in autocorr_dur), "Autocorrelation values should be between -1 and 1"
    
    # Test with melodic accents
    autocorr_mel = onset_autocorrelation_with_accents(
        starts, ends, pitches, accent_type='melodic', max_lag_quarters=2
    )
    assert len(autocorr_mel) == 9, f"Should have 9 autocorr values (0-8 divisions), got {len(autocorr_mel)}"
    assert autocorr_mel[0] == 1.0, f"Lag 0 should be 1.0 (normalized), got {autocorr_mel[0]}"


def test_edge_cases():
    """Test edge cases and error handling."""
    # Empty inputs
    empty_meter = estimate_meter([], [], [])
    assert empty_meter == 2, f"Empty input should default to 2 (duple), got {empty_meter}"
    
    # Single note
    single_meter = estimate_meter([0.0], [1.0], [60])
    assert single_meter == 2, f"Single note should default to 2 (duple), got {single_meter}"

    duple_sig = meter_to_time_signature(2)
    assert duple_sig == (4, 4), f"Meter 2 should convert to (4, 4), got {duple_sig}"
    
    triple_sig = meter_to_time_signature(3)
    assert triple_sig == (3, 4), f"Meter 3 should convert to (3, 4), got {triple_sig}"
    
    # Test invalid meter value (should raise ValueError)
    with pytest.raises(ValueError, match="Invalid meter value: 999"):
        meter_to_time_signature(999)


def test_meter_estimation_robustness():
    """Test that meter estimation handles various realistic patterns without crashing."""
    # March-like pattern
    march_starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    march_ends = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9, 4.4]
    march_pitches = [60, 60, 64, 64, 67, 67, 72, 72, 60]
    
    march_meter = estimate_meter(march_starts, march_ends, march_pitches, use_optimal=True)
    assert march_meter in [2, 3], f"March pattern should return valid meter, got {march_meter}"
    
    # Waltz-like pattern
    waltz_starts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    waltz_ends = [0.9, 1.4, 2.4, 3.9, 4.4, 5.4, 6.9, 7.4, 8.4]
    waltz_pitches = [48, 60, 64, 48, 60, 64, 48, 60, 64]
    
    waltz_meter = estimate_meter(waltz_starts, waltz_ends, waltz_pitches, use_optimal=True)
    assert waltz_meter in [2, 3], f"Waltz pattern should return valid meter, got {waltz_meter}"
    
    # Irregular pattern (should still return something reasonable)
    irregular_starts = [0.0, 0.33, 1.0, 1.75, 2.5, 3.2]
    irregular_ends = [0.25, 0.8, 1.6, 2.1, 3.0, 3.9]
    irregular_pitches = [60, 72, 55, 67, 62, 69]
    
    irregular_meter = estimate_meter(irregular_starts, irregular_ends, irregular_pitches, use_optimal=True)
    assert irregular_meter in [2, 3], f"Irregular pattern should return valid meter, got {irregular_meter}"


def test_function_interfaces():
    """Test that all meter estimation functions have consistent interfaces."""
    starts = [0.0, 1.0, 2.0]
    ends = [0.5, 1.5, 2.5]
    pitches = [60, 62, 64]
    
    # Test that functions accept expected input types and return numbers
    simple_result = estimate_meter_simple(starts, ends)
    assert isinstance(simple_result, (int, float)), f"Simple method should return number, got {type(simple_result)}"
    assert simple_result in [2, 3], f"Simple method should return 2 or 3, got {simple_result}"
    
    optimal_result = estimate_meter_optimal(starts, ends, pitches)
    assert isinstance(optimal_result, (int, float)), f"Optimal method should return number, got {type(optimal_result)}"
    assert optimal_result in [2, 3], f"Optimal method should return 2 or 3, got {optimal_result}"
    
    # Test main function fallback behavior
    meter_no_pitches = estimate_meter(starts, ends, use_optimal=True)
    assert meter_no_pitches in [2, 3], f"Should return valid meter without pitches, got {meter_no_pitches}"
    
    meter_with_pitches = estimate_meter(starts, ends, pitches, use_optimal=True)
    assert meter_with_pitches in [2, 3], f"Should return valid meter with pitches, got {meter_with_pitches}"


def test_accent_edge_cases():
    """Test accent functions with edge cases."""
    # Empty inputs
    empty_dur_accents = duration_accent([], [])
    assert empty_dur_accents == [], "Empty inputs should return empty list"
    
    empty_mel_accents = melodic_accent([])
    assert empty_mel_accents == [], "Empty pitches should return empty list"
    
    # Single note
    single_mel_accents = melodic_accent([60])
    assert single_mel_accents == [1.0], f"Single note should have accent [1.0], got {single_mel_accents}"
    
    # Two notes
    two_mel_accents = melodic_accent([60, 62])
    assert two_mel_accents == [1.0, 0.0], f"Two notes should have accents [1.0, 0.0], got {two_mel_accents}"


def test_autocorr_edge_cases():
    """Test autocorrelation function with edge cases."""
    # Empty inputs
    empty_autocorr = onset_autocorrelation_with_accents([], [], [], max_lag_quarters=2)
    assert len(empty_autocorr) == 9, f"Empty input should return 9 zeros, got {len(empty_autocorr)}"
    assert all(val == 0.0 for val in empty_autocorr), "Empty input should return all zeros"
    
    # Mismatched lengths
    mismatch_autocorr = onset_autocorrelation_with_accents([0.0], [0.5, 1.0], [60], max_lag_quarters=1)
    assert len(mismatch_autocorr) == 5, f"Mismatched input should return 5 zeros, got {len(mismatch_autocorr)}"
    assert all(val == 0.0 for val in mismatch_autocorr), "Mismatched input should return all zeros"


def test_meter_to_time_signature_conversion():
    """Test conversion from estimated meter to time signature."""
    assert meter_to_time_signature(2) == (4, 4)
    assert meter_to_time_signature(3) == (3, 4)
    
    # Test that invalid inputs raise ValueError
    with pytest.raises(ValueError, match="Invalid meter value: 999"):
        meter_to_time_signature(999)
    
    with pytest.raises(ValueError, match="Invalid meter value: 0"):
        meter_to_time_signature(0)
        
    with pytest.raises(ValueError, match="Invalid meter value: 1"):
        meter_to_time_signature(1)


def test_estimate_meter_consistency():
    """Test that meter estimation methods return consistent types and ranges."""
    starts = [0.0, 1.0, 2.0, 3.0]
    ends = [0.5, 1.5, 2.5, 3.5]
    pitches = [60, 62, 64, 65]
    
    # Test that simple and main function (simple mode) are consistent
    simple_result = estimate_meter_simple(starts, ends)
    main_result_simple = estimate_meter(starts, ends, pitches, use_optimal=False)
    
    assert simple_result == main_result_simple, "Simple and main function (simple mode) should give same result"
    assert simple_result in [2, 3], f"Should return valid meter value, got {simple_result}"
    
    # Test that optimal method doesn't crash and returns valid values
    optimal_result = estimate_meter_optimal(starts, ends, pitches)
    assert optimal_result in [2, 3], f"Optimal method should return valid meter, got {optimal_result}"


def test_end_to_end_workflow():
    """Test complete workflow from note data to time signature."""
    # Realistic note sequence
    starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    ends = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    
    # Complete workflow
    estimated_meter = estimate_meter(starts, ends, pitches, use_optimal=True)
    time_signature = meter_to_time_signature(estimated_meter)
    
    # Verify output format and validity
    assert isinstance(estimated_meter, (int, float)), "Meter should be numeric"
    assert estimated_meter in [2, 3], "Meter should be 2 or 3"
    assert isinstance(time_signature, tuple), "Time signature should be tuple"
    assert len(time_signature) == 2, "Time signature should have 2 elements"
    assert time_signature in [(4, 4), (3, 4)], "Should be valid time signature"
