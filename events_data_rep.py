# Import necessary music21 classes
from music21 import converter, note, stream, midi
from music21.tempo import MetronomeMark
import glob
import pickle as pkl
import os

PARSED_DATA_PATH   = "events_parsed_data/"
PRINT = False
OUTPUT_MIDI_PATH = "evt_output/"

def parse_monophonic_music_test(parsed_data_path=None):
    # Step 1: Parse the input monophonic music file into a music21 stream.
    # (Replace 'path/to/file' with the actual path to your MusicXML or MIDI file)
    file_list = glob.glob("./data/bach-cello/*.mid")

    for file in file_list:
        score = converter.parse(file)  # Load the first file in the list

        # Step 2: Flatten the stream to ensure all notes are in a single layer, sorted by time.
        flat_score = score.flat

        # Extract all Note objects from the flattened stream (ignore rests and other elements).
        notes = flat_score.getElementsByClass(note.Note)

        # Step 3: Collect note start (NOTE_ON) and note end (NOTE_OFF) events with their times.
        events = []  # list to hold tuples of (time, event_string)
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

        # Step 6: Output the timeline events.
        # Print each event on a new line in the order they occur.
        if PRINT:
            for evt in timeline:
                print(evt)

        # write the timeline to a file with name based on the input file
        output_filename = file.split('/')[-1].replace('.mid', '_events.pkl')
        with open(parsed_data_path + output_filename, "wb") as f:
            pkl.dump(timeline, f)


def load_parsed_events_test():
    # Load the parsed events from the saved file.
    with open(PARSED_DATA_PATH + "cs1-1pre_events.pkl", "rb") as f:
        timeline = pkl.load(f)
    return timeline

def load_parsed_events(parsed_data_path=None):
    # Load the parsed events from the saved file.
    parsed_events_files = glob.glob(parsed_data_path + "*.pkl")
    timelines = []  # list to hold all timelines
    for file in parsed_events_files:
        with open(file, "rb") as f:
            timeline = pkl.load(f)
        # concatenate all timelines into a single list
        timelines.extend(timeline)
    return timelines

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
    s.append(MetronomeMark(number=tempo_bpm))

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

# def on_epoch_end(self, epoch, logs=None):
#     info = self.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.5)
#     midi = info[-1]["midi"].chordify()
#     # print(info[-1]["prompt"])
#     midi.show()
#     midi.write("midi", fp=os.path.join("output", f"output-{epoch:04d}.mid"))

def test_evt_data_rep():
    # Test the event data representation by parsing a monophonic music file.
    parse_monophonic_music_test(parsed_data_path=PARSED_DATA_PATH)

    # Load the parsed events and reconstruct the MIDI stream.
    events = load_parsed_events_test()
    reconstructed_stream = reconstruct_midi_from_events(events, tempo_bpm=120.0, output_path=OUTPUT_MIDI_PATH)

    # # Show the reconstructed MIDI stream.
    # reconstructed_stream.show('midi')

if __name__ == "__main__":
    # Run the test function to parse and reconstruct MIDI from events.
    test_evt_data_rep()
    print("Event data representation test completed successfully.")