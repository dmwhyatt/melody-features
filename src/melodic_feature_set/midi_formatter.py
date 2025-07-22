from collections import namedtuple
import pretty_midi

def format_midi_to_string(midi_file: str) -> str:
    """Reads a MIDI file and formats notes as a string of named tuples.
    
    Parameters
    ----------
    midi_file : str
        Path to the MIDI file to read
        
    Returns
    -------
    str
        String representation of notes in format:
        "Note(start=0.000000, end=0.500000, pitch=69, velocity=90)"
    """
    # Create named tuple for consistent formatting
    Note = namedtuple('Note', ['start', 'end', 'pitch', 'velocity'])
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Get notes from first instrument
    notes = midi_data.instruments[0].notes
    
    # Convert to list of named tuples
    note_tuples = [
        Note(
            start=float(note.start),
            end=float(note.end), 
            pitch=int(note.pitch),
            velocity=int(note.velocity)
        ) for note in notes
    ]
    
    # Format as string
    note_strings = [str(note) for note in note_tuples]
    return ', '.join(note_strings)

def format_essen_database():
    """Converts all MIDI files in the Essen database to formatted string representation.
    
    Reads each MIDI file from the Essen database directory, formats the notes using 
    format_midi_to_string(), and saves the output to a text file with the same name.
    """
    import os
    # Get all .mid files in the Essen database directory
    midi_files = []
    for root, dirs, files in os.walk("/Users/davidwhyatt/Documents/GitHub/mono-midi-transposition-dataset/midi"):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    print(midi_files)
    # Process each MIDI file
    # Create output JSON file
    output_file = "mono_midi_sequences.json"
    
    with open(output_file, 'w') as f:
        f.write("[\n")
        
        # Process each MIDI file
        for i, midi_file in enumerate(midi_files):
            try:
                # Format the MIDI file
                formatted_notes = format_midi_to_string(midi_file)
                
                # Write as JSON entry
                if i > 0:
                    f.write(",\n")
                    
                f.write("    {\n")
                f.write(f'        "ID": {i+1},\n')
                f.write(f'        "Original Melody": "{os.path.splitext(os.path.basename(midi_file))[0]}",\n')
                f.write('        "In Key": "TRUE",\n')  # Default values - you may want to calculate these
                f.write('        "Contour Dif": 0,\n')
                f.write('        "Change Note": 0,\n')
                f.write('        "Displacement": 0,\n')
                f.write('        "Oddity": 1,\n')
                f.write('        "Discrimination": 1.0,\n')
                f.write('        "Difficulty": 0.0,\n')
                f.write(f'        "MIDI Sequence": "{formatted_notes}"\n')
                f.write('    }')
            
            except Exception as e:
                print(f"Error processing {midi_file}: {str(e)}")
                continue

        f.write("\n]")

format_essen_database()
