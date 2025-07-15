from music21 import converter, note, stream, tempo
import glob
import os


def extract_stream(file_path):
    """Extract and return a music21 stream from a MIDI file."""
    try:
        score = converter.parse(file_path)
        return score
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_streams_from_directory(directory_path="./data/bach-cello/*.mid"):
    """Extract streams from all MIDI files in a directory."""
    file_list = glob.glob(directory_path)
    streams = []
    
    for file_path in file_list:
        print(f"Processing: {os.path.basename(file_path)}")
        stream_obj = extract_stream(file_path)
        stream_obj.show('text')  # Display the stream in text format
        if stream_obj:
            streams.append({
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'stream': stream_obj
            })
    
    return streams
    


def print_midi_text_representation():
    """Print a text representation of each MIDI file showing notes and timing."""
    file_list = glob.glob("./data/bach-cello/*.mid")
    
    for file_path in file_list:
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        try:
            score = converter.parse(file_path)
            flat_score = score.flat
            
            # Get all notes and sort by offset
            notes = flat_score.getElementsByClass(note.Note)
            notes_list = [(n.offset, n.nameWithOctave, n.quarterLength) for n in notes]
            notes_list.sort(key=lambda x: x[0])
            
            print(f"Total notes: {len(notes_list)}")
            print(f"Duration: {flat_score.duration.quarterLength} quarter notes")
            print("\nNote sequence (Time | Pitch | Duration):")
            print("-" * 40)
            
            for offset, pitch, duration in notes_list[:20]:  # Show first 20 notes
                print(f"{offset:6.2f} | {pitch:>4} | {duration:6.2f}")
            
            if len(notes_list) > 20:
                print(f"... and {len(notes_list) - 20} more notes")
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")


if __name__ == "__main__":
    extract_streams_from_directory()