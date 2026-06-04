"""
This module contains classes and functions to represent melodies and extract information
from MIDI sequence data.
"""

__author__ = "David Whyatt"
import json


def build_midi_sequence_string(
    pitches: list[int],
    starts: list[float],
    ends: list[float],
) -> str:
    """Build the legacy ``MIDI Sequence`` string from parallel note lists."""
    return "Note(" + "Note(".join(
        f"pitch={p}, start={s}, end={e})"
        for p, s, e in zip(pitches, starts, ends)
    )


class Melody:
    """Melody representation backed by parallel note lists.

    Preferred input is a ``midi_data`` dict with ``pitches``, ``starts``, and
    ``ends`` (as produced by :func:`melody_features.io.midi.import_midi`).
    Legacy dicts that only provide ``MIDI Sequence`` are still supported via
    string parsing.
    """

    def __init__(self, midi_data: dict, tempo: float = None):
        """Initialize a Melody object from MIDI sequence data.

        Args:
            midi_data (dict): Dictionary containing MIDI sequence data
            tempo (float, optional): Tempo in BPM. If None, uses tempo from midi_data if available.
        """
        self._midi_data = midi_data
        midi_sequence = midi_data.get("MIDI Sequence", "")
        self._midi_sequence = (
            midi_sequence.split("Note(")[1:] if midi_sequence else []
        )

        if tempo is not None:
            self._tempo = tempo
        elif "tempo" in midi_data:
            self._tempo = midi_data["tempo"]
        else:
            self._tempo = 100.00

        self._tempo_changes = midi_data.get("tempo_changes", [(0.0, self._tempo)])

        self._pitches, self._starts, self._ends = self._load_note_lists(midi_data)

    @staticmethod
    def _has_structured_note_lists(midi_data: dict) -> bool:
        """Return whether ``midi_data`` has aligned pitches/starts/ends lists."""
        for key in ("pitches", "starts", "ends"):
            values = midi_data.get(key)
            if not isinstance(values, list):
                return False
        note_count = len(midi_data["pitches"])
        return (
            len(midi_data["starts"]) == note_count
            and len(midi_data["ends"]) == note_count
        )

    @classmethod
    def _load_note_lists(
        cls, midi_data: dict
    ) -> tuple[list[int], list[float], list[float]]:
        """Load note lists from structured fields or legacy MIDI Sequence text."""
        if cls._has_structured_note_lists(midi_data):
            return (
                [int(p) for p in midi_data["pitches"]],
                [float(s) for s in midi_data["starts"]],
                [float(e) for e in midi_data["ends"]],
            )

        midi_sequence = midi_data.get("MIDI Sequence", "")
        if not midi_sequence:
            return [], [], []

        fragments = midi_sequence.split("Note(")[1:]
        return cls._parse_midi_sequence(fragments)

    @staticmethod
    def _parse_midi_sequence(
        midi_sequence: list[str],
    ) -> tuple[list[int], list[float], list[float]]:
        """Parse pitch, start, and end times from note string fragments."""
        pitches: list[int] = []
        starts: list[float] = []
        ends: list[float] = []
        for note_fragment in midi_sequence:
            pitch_text = Melody._read_note_field(note_fragment, "pitch")
            if pitch_text is not None:
                pitches.append(int(pitch_text))

            start_text = Melody._read_note_field(note_fragment, "start")
            if start_text is not None:
                starts.append(float(start_text))

            end_text = Melody._read_note_field(note_fragment, "end")
            if end_text is not None:
                ends.append(float(end_text))
        return pitches, starts, ends

    @staticmethod
    def _read_note_field(note_fragment: str, field_name: str) -> str | None:
        """Return the raw text of one ``Note(...)`` field (e.g. pitch, start, end)."""
        marker = f"{field_name}="
        marker_at = note_fragment.find(marker)
        if marker_at == -1:
            return None

        value_begin = marker_at + len(marker)
        comma_at = note_fragment.find(",", value_begin)
        paren_at = note_fragment.find(")", value_begin)

        if field_name == "start":
            if comma_at == -1:
                return None
            value_stop = comma_at
        elif field_name == "pitch":
            if comma_at == -1:
                value_stop = paren_at
            elif (
                comma_at + 1 < len(note_fragment)
                and note_fragment[comma_at + 1] == ")"
            ):
                value_stop = paren_at
            else:
                value_stop = comma_at
        else:
            value_stop = comma_at if comma_at != -1 else paren_at

        if value_stop == -1:
            return None
        return note_fragment[value_begin:value_stop].rstrip(",)")

    @property
    def id(self) -> str:
        """Get the ID (file path) of the MIDI file.

        Returns:
            str: File path or ID of the MIDI file
        """
        return self._midi_data.get("ID", "")
    
    @property
    def pitches(self) -> list[int]:
        """List of MIDI note pitches in order of appearance."""
        return self._pitches

    @property
    def starts(self) -> list[float]:
        """List of MIDI note start times in order of appearance."""
        return self._starts

    @property
    def ends(self) -> list[float]:
        """List of MIDI note end times in order of appearance."""
        return self._ends

    @property
    def tempo(self) -> float:
        """Extract tempo from Class input.

        Returns:
            float: Tempo of the melody in beats per minute
        """
        return self._tempo
    
    @property
    def tempo_changes(self) -> list[tuple[float, float]]:
        """Get tempo changes from the melody.

        Returns:
            list[tuple[float, float]]: List of (time_in_seconds, tempo_in_bpm) tuples
        """
        return self._tempo_changes
    

    @property
    def meter(self) -> tuple[int, int]:
        """Extract the first time signature from the melody.
        
        Returns:
            tuple[int, int]: First time signature as (numerator, denominator)
                           Defaults to (4, 4) if no time signature information available
        """
        time_sig_info = self._midi_data.get("time_signature_info")
        if time_sig_info and "first_time_signature" in time_sig_info:
            return time_sig_info["first_time_signature"]
        return (4, 4)  # Default to 4/4 if no information available

    @property
    def time_signatures(self) -> list[tuple[float, int, int]]:
        """Get all time signatures present in the melody.
        
        Returns:
            list[tuple[float, int, int]]: List of tuples (time_in_seconds, numerator, denominator).
                If none are available, falls back to a single entry using the first meter at time 0.0.
        """
        time_sig_info = self._midi_data.get("time_signature_info")
        if time_sig_info and "all_time_signatures" in time_sig_info:
            return time_sig_info["all_time_signatures"]
        num, den = self.meter
        return [(0.0, num, den)]

    @property  
    def proportion_of_time_in_first_meter(self) -> float:
        """Calculate the proportion of time spent in the first time signature.
        
        Returns:
            float: Proportion (0.0 to 1.0) that the first time signature comprises 
                   of the total melody duration. 1.0 means the melody uses only 
                   one time signature throughout.
        """
        time_sig_info = self._midi_data.get("time_signature_info")
        if time_sig_info and "proportion_of_time_in_first_meter" in time_sig_info:
            return time_sig_info["proportion_of_time_in_first_meter"]
        return 0.0

    @property
    def total_duration(self) -> float:
        """Get the total duration of the MIDI sequence in seconds.
        
        Returns:
            float: Total duration of the MIDI sequence in seconds, including any 
                   leading or trailing silence. This matches jSymbolic's 
                   DurationInSecondsFeature implementation.
        """
        return self._midi_data.get("total_duration", 0.0)
    
    @property
    def key_signature(self) -> tuple:
        """Get the first key signature in the melody.
        
        Returns:
            tuple or None: (key_name, mode) where key_name is a string like 'C', 'Am', 'F#'
                          and mode is either 'major' or 'minor'. Returns None if no key signature found.
        """
        key_sig_info = self._midi_data.get("key_signature_info")
        if key_sig_info:
            return key_sig_info.get("first_key_signature")
        return None
    
    @property
    def key_signatures(self) -> list:
        """Get all key signatures present in the melody.
        
        Returns:
            list[tuple]: List of tuples (key_name, mode) for all key signatures.
                        Empty list if no key signatures are found.
        """
        key_sig_info = self._midi_data.get("key_signature_info")
        if key_sig_info:
            return key_sig_info.get("all_key_signatures", [])
        return []
    
    @property
    def has_key_signature(self) -> bool:
        """Check if the MIDI file contains any key signature information.
        
        Returns:
            bool: True if at least one key signature was found, False otherwise.
        """
        key_sig_info = self._midi_data.get("key_signature_info")
        if key_sig_info:
            return key_sig_info.get("has_key_signature", False)
        return False
    
    @property
    def key_fifths(self) -> int:
        """Get the circle of fifths position of the first key signature.
        
        Returns:
            int or None: Position on circle of fifths (-7 to 7) where:
                        0 = C major / A minor
                        Positive = sharps (G=1, D=2, A=3, E=4, B=5, F#=6, C#=7)
                        Negative = flats (F=-1, Bb=-2, Eb=-3, Ab=-4, Db=-5, Gb=-6, Cb=-7)
                        Returns None if no key signature found.
        """
        key_sig_info = self._midi_data.get("key_signature_info")
        if key_sig_info:
            return key_sig_info.get("fifths")
        return None
    
    @property
    def key_mode(self) -> int:
        """Get the mode of the first key signature as an integer.
        
        Returns:
            int or None: 1 for major, -1 for minor. Returns None if no key signature found.
        """
        key_sig_info = self._midi_data.get("key_signature_info")
        if key_sig_info:
            return key_sig_info.get("mode")
        return None

    

def read_midijson(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
