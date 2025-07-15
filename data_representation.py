from music21 import converter, note
import glob

PARSED_DATA_PATH   = "parsed_data/"

file_list = glob.glob("./data/bach-cello/*.mid")

for i, file in enumerate(file_list):
    if i == 1:
        break

    print(f"Parsing {file}")
    # Load stream from MIDI file
    stream = converter.parse(file)

    # Iterate through all Note objects in the (flattened) Stream
    for n in stream.flat.getElementsByClass(note.Note):
        # 1. Onset (in quarter-note units from start of stream)
        onset_q = n.offset  
        
        # 2. Duration (in quarter-note units)
        dur_q   = n.duration.quarterLength
        
        # 3. Offset of note-off event
        offset_q = onset_q + dur_q
        
        print(f"{n.pitch} on at {onset_q:.2f} q-beats, off at {offset_q:.2f} q-beats")

        
