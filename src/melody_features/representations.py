"""
This module contains classes and functions to represent melodies and extract information
from MIDI sequence data.
"""

__author__ = "David Whyatt"
import json


class Melody:
    """Class to represent a melody from a MIDI sequence. This class is used to extract
    information from a json file containing MIDI sequence data, formatted accordingly:
    A note is represented as a string in the format:
    'Note(start=0.0, end=0.25, pitch=60, velocity=100)'
    We don't need the velocity, so we can ignore it here.

    Attributes:
        pitches (list[int]): List of MIDI note pitches in order of appearance
        starts (list[float]): List of note start times in order of appearance
        ends (list[float]): List of note end times in order of appearance
    """

    def __init__(self, midi_data: dict, tempo: float = 100.00):
        """Initialize a Melody object from MIDI sequence data.

        Args:
            midi_data (dict): Dictionary containing MIDI sequence data
        """
        self._midi_data = midi_data
        # Split on 'Note(' and remove the first empty string
        self._midi_sequence = midi_data["MIDI Sequence"].split("Note(")[1:]
        self._tempo = tempo

    @property
    def pitches(self) -> list[int]:
        """Extract pitch values from MIDI sequence.

        Returns:
            list[int]: List of MIDI pitch values in order of appearance
        """
        pitches = []
        for note in self._midi_sequence:
            # Find pitch value between 'pitch=' and the next comma or closing parenthesis
            pitch_start = note.find("pitch=") + 6
            pitch_end = note.find(",", pitch_start)
            if pitch_end == -1:  # Handle the last note which ends with ')'
                pitch_end = note.find(")", pitch_start)
            if pitch_end != -1:  # Only process if we found a valid pitch
                pitch = int(note[pitch_start:pitch_end])
                pitches.append(pitch)
        return pitches

    @property
    def starts(self) -> list[float]:
        """Extract start times from MIDI sequence.

        Returns:
            list[float]: List of MIDI note start times in order of appearance
        """
        starts = []
        for note in self._midi_sequence:
            # Find start time between 'start=' and the next comma
            start_start = note.find("start=") + 6
            start_end = note.find(",", start_start)
            if start_end != -1:  # Only process if we found a valid start time
                start = float(note[start_start:start_end])
                starts.append(start)
        return starts

    @property
    def ends(self) -> list[float]:
        """Extract end times from MIDI sequence.

        Returns:
            list[float]: List of MIDI note end times in order of appearance
        """
        ends = []
        for note in self._midi_sequence:
            # Find end time between 'end=' and the next comma or closing parenthesis
            end_start = note.find("end=") + 4
            end_end = note.find(",", end_start)
            if end_end == -1:  # Handle the last note which ends with ')'
                end_end = note.find(")", end_start)
            if end_end != -1:  # Only process if we found a valid end time
                end = float(note[end_start:end_end])
                ends.append(end)
        return ends

    @property
    def tempo(self) -> float:
        """Extract tempo from Class input.

        Returns:
            float: Tempo of the melody in beats per minute
        """
        return self._tempo


def read_midijson(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
