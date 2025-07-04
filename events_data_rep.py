# Import necessary music21 classes
from music21 import converter, note, stream, midi
from music21.tempo import MetronomeMark
import glob
import pickle as pkl
import os

PARSED_DATA_PATH   = "events_parsed_data/"
OUTPUT_MIDI_PATH = "evt_output/"

def parse_monophonic_music(parsed_data_path, seq_len):
    # Step 1: Parse the input monophonic music file into a music21 stream.
    # (Replace 'path/to/file' with the actual path to your MusicXML or MIDI file)
    file_list = glob.glob("./data/bach-cello/*.mid")
    events = []  # list to hold tuples of (time, event_string)

    for file in file_list:
        score = converter.parse(file)  # Load the first file in the list

        # Step 2: Flatten the stream to ensure all notes are in a single layer, sorted by time.
        flat_score = score.flat

        # Extract all Note objects from the flattened stream (ignore rests and other elements).
        notes = flat_score.getElementsByClass(note.Note)

        # Step 3: Collect note start (NOTE_ON) and note end (NOTE_OFF) events with their times.
        for n in notes:
            pitch_name = n.nameWithOctave    # e.g., "C4", "F#5" - note name with octave
            start_time = n.offset            # note start time (offset from beginning in quarterLength units)
            duration = n.quarterLength       # note duration in quarterLength units
            end_time = start_time + duration # note end time

            # Record the NOTE_ON and NOTE_OFF events with their times
            events.append((start_time, f"NOTE_ON({pitch_name})"))
            events.append((end_time, f"NOTE_OFF({pitch_name})"))

        # Step 4: Sort the events by time to get a chronological sequence.
        events.sort(key=lambda x: x[0])

        # Step 5: Iterate through sorted events and insert TIME_SHIFT events for gaps between events.
        timeline = []   # final list of timeline events as strings
        timeline.append("START")  # start of the timeline
        last_time = 0.0 # tracks the time of the last event processed
        for time, event in events:
            # Calculate the time difference from the previous event
            delta = time - last_time
            if delta > 0:
                # If there is a gap, insert a TIME_SHIFT representing the rest or sustain duration
                timeline.append(f"TIME_SHIFT({delta})")
            # Append the current event (NOTE_ON or NOTE_OFF)
            timeline.append(event)
            # Update last_time to the current event's time
            last_time = time

    notes_list = []
    print(f"Building sequences of length {seq_len}")
    for i in range(len(timeline) - seq_len):
        notes_list.append(" ".join(timeline[i : (i + seq_len)]))

    with open(os.path.join(parsed_data_path, "notes"), "wb") as f:
        pkl.dump(notes_list, f)

    return notes_list


def load_parsed_events(parsed_data_path=None):
    # Load the parsed events from the saved file.
    with open(os.path.join(parsed_data_path, "notes"), "rb") as f:
        notes = pkl.load(f)
    return notes

def reconstruct_midi_from_events(events: list, tempo_bpm: float = 120.0, output_path: str = None):
    """
    Reconstructs a music21 Stream (and optionally writes a MIDI file) from an event timeline.

    Args:
        events: list of event strings as returned by parse_monophonic_music().
        tempo_bpm: tempo in beats per minute for playback timing.
        output_path: if provided, path where the MIDI file will be written.

    Returns:
        music21.stream.Stream containing the reconstructed notes.
    """
    s = stream.Stream()
    # s.append(MetronomeMark(number=tempo_bpm))

    current_time = 0.0
    # Keep track of active notes: pitch_name -> start_time
    active = {}

    for evt in events:
        if evt.startswith('TIME_SHIFT'):
            # Extract the delta and advance time
            delta = float(evt[evt.find('(')+1:evt.find(')')])
            current_time += delta
        elif evt.startswith('NOTE_ON'):
            pitch = evt[evt.find('(')+1:evt.find(')')]
            # Record the start time of this pitch
            active[pitch] = current_time
        elif evt.startswith('NOTE_OFF'):
            pitch = evt[evt.find('(')+1:evt.find(')')]
            if pitch in active:
                start = active.pop(pitch)
                dur = current_time - start
                # Create the note and set timing
                n = note.Note(pitch)
                n.offset = start
                n.quarterLength = dur
                s.append(n)

    # Sort notes by offset and write MIDI if requested
    s.sort()
    if output_path:
        s.write('midi', fp=os.path.join(output_path, f"output.mid"))
    return s