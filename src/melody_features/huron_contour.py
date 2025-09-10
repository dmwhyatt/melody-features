__author__ = "David Whyatt"

# I didn't implement this before, but Mark has a good implementation in
# AMADS, see https://github.com/music-computing/amads

from melody_features.representations import Melody


class HuronContour:
    """A class for computing the Huron Contour of a melody, as described in the FANTASTIC toolbox.
    Huron Contour classifies melodies based on the shape of the contour between the first pitch, the 
    mean pitch, and the last pitch.
    
    Attributes
    ----------
    melody : Melody
        The melody object containing the melody to analyze.
    huron_contour : str
        The Huron contour classification for the melody.
    """
    def __init__(self, melody: Melody):
        """Initialize the Huron Contour using a Melody object and calculate
        the Huron Contour classification.
        
        Parameters
        ----------
        melody : Melody
            The melody object containing the melody to analyze.
        """
        self.melody = melody
        contour_points = self.get_contour_points(melody)
        self.huron_contour = self.get_contour_class(contour_points)

    def get_contour_points(self, melody: Melody) -> list[float]:
        """Get the contour points of a melody.
        
        Calculates the first pitch, weighted mean pitch, and last pitch
        of the melody.
        
        Parameters
        ----------
        melody : Melody
            The melody object to analyze
            
        Returns
        -------
        list[float]
            The three relevant contour points: [first_pitch, mean_pitch, last_pitch]
        """
        if not melody.pitches or len(melody.pitches) == 0:
            return [0.0, 0.0, 0.0]
            
        if len(melody.pitches) == 1:
            single_pitch = float(melody.pitches[0])
            return [single_pitch, single_pitch, single_pitch]
        
        first_pitch = float(melody.pitches[0])
        last_pitch = float(melody.pitches[-1])
        
        # Calculate duration-weighted mean pitch by giving more weight to longer notes 
        # to determine the overall "center" of the melody
        if len(melody.starts) == len(melody.pitches) and len(melody.ends) == len(melody.pitches):
            durations = [end - start for start, end in zip(melody.starts, melody.ends)]
            total_duration = sum(durations)
            if total_duration > 0:
                weighted_sum = sum(pitch * duration for pitch, duration in zip(melody.pitches, durations))
                mean_pitch = weighted_sum / total_duration
            else:
                # Fallback to simple mean if durations are problematic
                mean_pitch = sum(melody.pitches) / len(melody.pitches)
        else:
            # Fallback to simple arithmetic mean if timing data is inconsistent
            mean_pitch = sum(melody.pitches) / len(melody.pitches)
        
        # Round mean pitch to nearest integer (MIDI pitch value)
        mean_pitch = round(mean_pitch)
        
        return [first_pitch, float(mean_pitch), last_pitch]

    def get_contour_class(self, contour_points: list[float]) -> str:
        """The classification of a contour based on the relationship between the first, mean, and last pitch of the melody.
        
        Parameters
        ----------
        contour_points : list[float]
            The three relevant contour points: [first_pitch, mean_pitch, last_pitch]
            
        Returns
        -------
        str
            The contour classification according to Huron's system
        """
        if len(contour_points) != 3:
            return "horizontal"
        
        p1, p_mean, pn = contour_points
        
        # Classify based on Huron's 8 categories
        if p1 < p_mean > pn:
            return "convex"
        elif p1 < p_mean == pn:
            return "ascending-horizontal"
        elif p1 < p_mean < pn:
            return "ascending"
        elif p1 == p_mean == pn:
            return "horizontal"
        elif p1 == p_mean > pn:
            return "horizontal-descending"
        elif p1 == p_mean < pn:
            return "horizontal-ascending"
        elif p1 > p_mean == pn:
            return "descending-horizontal"
        elif p1 > p_mean > pn:
            return "descending"
        elif p1 > p_mean < pn:
            return "concave"
        else:
            return "unclassified"


def get_huron_contour(melody: Melody) -> str:
    """Calculate Huron contour classification for an arbitrary melody.
    Used here for doctesting and offered for user convenience.
    
    Parameters
    ----------
    melody : Melody
        The melody object to analyze
        
    Returns
    -------
    str
        The Huron contour classification
        
    Examples
    --------
    Single note melody:
    >>> single_note_data = {"MIDI Sequence": "Note(start=0.0, end=1.0, pitch=60, velocity=100)"}
    >>> single_note = Melody(single_note_data)
    >>> get_huron_contour(single_note)
    'horizontal'
    
    The lick - convex contour (62, 64, 65, 67, 64, 60, 62):
    >>> lick_data = {"MIDI Sequence": "Note(start=0.0, end=1.0, pitch=62, velocity=100)Note(start=1.0, end=2.0, pitch=64, velocity=100)Note(start=2.0, end=3.0, pitch=65, velocity=100)Note(start=3.0, end=4.0, pitch=67, velocity=100)Note(start=4.0, end=6.0, pitch=64, velocity=100)Note(start=6.0, end=7.0, pitch=60, velocity=100)Note(start=7.0, end=8.0, pitch=62, velocity=100)"}
    >>> lick_melody = Melody(lick_data)
    >>> get_huron_contour(lick_melody)
    'convex'
    
    Ascending melody (60, 62, 64, 67):
    >>> asc_data = {"MIDI Sequence": "Note(start=0.0, end=1.0, pitch=60, velocity=100)Note(start=1.0, end=2.0, pitch=62, velocity=100)Note(start=2.0, end=3.0, pitch=64, velocity=100)Note(start=3.0, end=4.0, pitch=67, velocity=100)"}
    >>> asc_melody = Melody(asc_data)
    >>> get_huron_contour(asc_melody)
    'ascending'
    
    Descending melody (67, 64, 62, 60):
    >>> desc_data = {"MIDI Sequence": "Note(start=0.0, end=1.0, pitch=67, velocity=100)Note(start=1.0, end=2.0, pitch=64, velocity=100)Note(start=2.0, end=3.0, pitch=62, velocity=100)Note(start=3.0, end=4.0, pitch=60, velocity=100)"}
    >>> desc_melody = Melody(desc_data)
    >>> get_huron_contour(desc_melody)
    'descending'
    """
    hc = HuronContour(melody)
    return hc.huron_contour
