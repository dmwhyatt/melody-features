"""Shared MIDI fixtures for tests."""


def create_test_midi_file(pitches, starts, ends, tempo=120, filepath=None):
    """Create a MIDI file (mido.MidiFile or path) for testing."""
    import mido

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))

    ticks_per_second = 480 * (tempo / 60)
    current_time = 0
    for pitch, start, end in zip(pitches, starts, ends):
        start_ticks = int(start * ticks_per_second)
        delta_time = start_ticks - current_time
        track.append(
            mido.Message("note_on", channel=0, note=pitch, velocity=64, time=delta_time)
        )
        duration_ticks = int((end - start) * ticks_per_second)
        track.append(
            mido.Message(
                "note_off", channel=0, note=pitch, velocity=64, time=duration_ticks
            )
        )
        current_time = start_ticks + duration_ticks

    if filepath:
        mid.save(filepath)
        return filepath
    return mid
