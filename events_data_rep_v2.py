# Import necessary music21 classes
from music21 import converter, note, stream, tempo
import music21
import glob
import pickle as pkl
import os
from fractions import Fraction

PARSED_DATA_PATH = "events_parsed_data/"
OUTPUT_MIDI_PATH = "evt_output/"

# Define duration buckets for TIME_SHIFT, DURATION, and REST events
DURATION_BUCKETS = [0.25, 0.5, 1.0, 2.0, 4.0]

def bucket_duration(dur):
    """Quantize a duration to the nearest bucket."""
    return min(DURATION_BUCKETS, key=lambda b: abs(dur - b))


def parse_monophonic_music(data_path, seq_len):
    """
    Parse all monophonic MIDI files and extract event sequences.

    Each sequence consists of:
      - NOTE_ON(pitch)
      - DURATION(dur)
      - TIME_SHIFT(delta) for gaps and implied rests
      - NOTE_OFF(pitch)

    Returns a list of tokenized sequences of length seq_len.
    """
    file_list = glob.glob("./data/bach-cello/*.mid")
    if not file_list:
        raise FileNotFoundError(f"No MIDI files found in directory: {data_path}")

    all_sequences = []

    for file_path in file_list:
        score = converter.parse(file_path)
        flat_score = score.flat

        notes = flat_score.getElementsByClass(note.Note)
        print(f"Parsing {file_path}: {len(notes)} notes found.")

        # Collect raw events: (time, token)
        events = []
        for n in notes:
            pitch = n.nameWithOctave
            start = n.offset
            dur = bucket_duration(n.quarterLength)
            end = start + dur
            events.append((start, f"NOTE_ON({pitch})"))
            events.append((start, f"DURATION({dur})"))
            events.append((end,   f"NOTE_OFF({pitch})"))

        if not events:
            continue

        # Sort events chronologically
        events.sort(key=lambda x: x[0])

        # Build timeline with TIME_SHIFT only
        timeline = []
        last_time = 0.0
        for time, token in events:
            delta = time - last_time
            if delta > 0:
                shift = bucket_duration(delta)
                timeline.append(f"TIME_SHIFT({shift})")
            timeline.append(token)
            last_time = time

        if not timeline:
            continue

        # Create fixed-length sequences
        if len(timeline) >= seq_len:
            for i in range(len(timeline) - seq_len + 1):
                all_sequences.append(" ".join(timeline[i:i+seq_len]))
        else:
            all_sequences.append(" ".join(timeline))

    # Save parsed sequences
    os.makedirs(PARSED_DATA_PATH, exist_ok=True)
    with open(os.path.join(PARSED_DATA_PATH, "notes.pkl"), "wb") as f:
        pkl.dump(all_sequences, f)

    print(f"Total sequences extracted: {len(all_sequences)}")
    return all_sequences


def load_parsed_events(parsed_data_path=None):
    """Load the parsed event sequences from disk."""
    path = parsed_data_path or PARSED_DATA_PATH
    with open(os.path.join(path, "notes.pkl"), "rb") as f:
        return pkl.load(f)


def reconstruct_midi_from_events(events, tempo_bpm=120.0, output_path=None):
    """
    Reconstruct monophonic MIDI from event tokens.

    Args:
      events: flat list of event tokens.
      tempo_bpm: playback tempo.
      output_path: directory to save MIDI file.

    Returns:
      music21.stream.Stream of reconstructed notes.
    """
    s = stream.Stream()
    s.append(music21.instrument.Violoncello())
    s.append(tempo.MetronomeMark(number=tempo_bpm))
    current_time = 0.0
    active = {}

    for token in events:
        if token.startswith('TIME_SHIFT'):
            val = token[token.find('(')+1:token.find(')')]
            shift = float(Fraction(val))
            current_time += shift
        elif token.startswith('NOTE_ON'):
            pitch = token[token.find('(')+1:token.find(')')]
            active[pitch] = current_time
        elif token.startswith('DURATION'):
            # Duration metadata; actual timing via NOTE_OFF
            continue
        elif token.startswith('NOTE_OFF'):
            pitch = token[token.find('(')+1:token.find(')')]
            if pitch in active:
                start = active.pop(pitch)
                dur = current_time - start
                n = note.Note(pitch)
                n.offset = start
                n.quarterLength = dur
                s.append(n)

    s.sort()
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        s.write('midi', fp=os.path.join(output_path, "reconstructed.mid"))
    return s
