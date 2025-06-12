import os
import pickle as pkl
import music21
import keras
import tensorflow as tf

from fractions import Fraction

from music21 import converter, note
import music21


def parse_midi_files(file_list, parser, seq_len, parsed_data_path=None):
    notes_list = []
    duration_list = []
    notes = []
    durations = []

    for i, file in enumerate(file_list):
        print(i + 1, "Parsing %s" % file)
        score = parser.parse(file).chordify()

        notes.append("START")
        durations.append("0.0")

        for element in score.flat:
            note_name = None
            duration_name = None

            if isinstance(element, music21.key.Key):
                note_name = str(element.tonic.name) + ":" + str(element.mode)
                duration_name = "0.0"

            elif isinstance(element, music21.meter.TimeSignature):
                note_name = str(element.ratioString) + "TS"
                duration_name = "0.0"

            elif isinstance(element, music21.chord.Chord):
                note_name = element.pitches[-1].nameWithOctave
                duration_name = str(element.duration.quarterLength)

            elif isinstance(element, music21.note.Rest):
                note_name = str(element.name)
                duration_name = str(element.duration.quarterLength)

            elif isinstance(element, music21.note.Note):
                note_name = str(element.nameWithOctave)
                duration_name = str(element.duration.quarterLength)

            if note_name and duration_name:
                notes.append(note_name)
                durations.append(duration_name)
        print(f"{len(notes)} notes parsed")

    notes_list = []
    duration_list = []

    print(f"Building sequences of length {seq_len}")
    for i in range(len(notes) - seq_len):
        notes_list.append(" ".join(notes[i : (i + seq_len)]))
        duration_list.append(" ".join(durations[i : (i + seq_len)]))

    if parsed_data_path:
        with open(os.path.join(parsed_data_path, "notes"), "wb") as f:
            pkl.dump(notes_list, f)
        with open(os.path.join(parsed_data_path, "durations"), "wb") as f:
            pkl.dump(duration_list, f)

    return notes_list, duration_list


def parse_midi_files_toEvents(file_list, parser, seq_len, parsed_data_path=None):
    notes_list = []
    timelines = []

    for i, file in enumerate(file_list):
        print(i + 1, "Parsing %s" % file)
        timeline = extract_event_timeline(file)

        # start token
        timeline.append("START")
        timelines.append(timeline)

    #     for element in score.flat:
    #         # print(f"Parsing element {element} of type {type(element)}")
    #         if isinstance(element, music21.key.Key):
    #             # keep your old key tokens if you like
    #             tonic, mode = element.tonic.name, element.mode
    #             events.append(f"{tonic}:{mode}")
    #         elif isinstance(element, music21.meter.TimeSignature):
    #             events.append(f"{element.ratioString}TS")

    #         # elif isinstance(element, music21.note.Rest):
    #         #     dur = element.duration.quarterLength
    #         #     events.append(f"TIME_SHIFT<{dur}>")

    #         elif isinstance(element, music21.note.Note):
    #             midi = element.pitch.midi
    #             midi  = element.nameWithOctave
    #             dur = str(element.duration.quarterLength)
    #             events.append(f"NOTE_ON<{midi}>")
    #             events.append(f"TIME_SHIFT<{dur}>")
    #             durations.append(dur)
    #             events.append(f"NOTE_OFF<{midi}>")

    #         elif isinstance(element, music21.chord.Chord):
    #             # note_name = element.pitches[-1].nameWithOctave
    #             # duration_name = str(element.duration.quarterLength)
    #             # events.append(f"NOTE_ON<{note_name}>")
    #             # dur = element.duration.quarterLength
    #             # events.append(f"TIME_SHIFT<{dur}>")
    #             # events.append(f"NOTE_OFF<{note_name}>")

    #             # fire on for each pitch
    #             for p in element.pitches:
    #                 events.append(f"CHORD_ON<{p.nameWithOctave}>")
    #             events.append(f"TIME_SHIFT<{element.duration.quarterLength}>")
    #             # fire off for each pitch
    #             for p in element.pitches:
    #                 events.append(f"CHORD_OFF<{p.nameWithOctave}>")

    #     print(f"{len(events)} events parsed so far")
    #     # print(f"Events: {events[:10][0:20]} ...")
    # timelines = []
        print(f"Building sequences of length {seq_len}")
        for timeline in timelines:
            # print(f"Timeline: {timeline[:10]}")
            if len(timeline) < seq_len:
                continue
            # print(f"Timeline length: {len(timeline)}")
            # print(f"Timeline: {timeline}")
            notes_list.append(" ".join(timeline))
            for i in range(len(timeline) - seq_len):
                notes_list.append(" ".join(timeline[i : i + seq_len]))

    if parsed_data_path:
        with open(os.path.join(parsed_data_path, "notes"), "wb") as f:
            pkl.dump(notes_list, f)
        # with open(os.path.join(parsed_data_path, "durations"), "wb") as f:
        #     pkl.dump(durations_list, f)

    return notes_list

def extract_event_timeline(file_path: str):
    """
    Parse a monophonic MIDI or MusicXML file and extract a timeline of events.

    Returns a list of strings in the form:
      - NOTE_ON(C4)
      - TIME_SHIFT(1.0)
      - NOTE_OFF(C4)
    where durations are in quarterLength units.
    """
    # Load and flatten the score
    score = converter.parse(file_path)
    flat_score = score.flat

    # Collect NOTE_ON and NOTE_OFF events
    events = []
    for n in flat_score.getElementsByClass(note.Note):
        pitch_name = n.nameWithOctave
        start_time = n.offset
        duration = n.quarterLength
        end_time = start_time + duration

        events.append((start_time, f"NOTE_ON({pitch_name})"))
        events.append((end_time, f"NOTE_OFF({pitch_name})"))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    # Build timeline with TIME_SHIFT
    timeline = []
    last_time = 0.0
    for time, evt in events:
        delta = time - last_time
        if delta > 0:
            timeline.append(f"TIME_SHIFT({delta})")
        timeline.append(evt)
        last_time = time

    return timeline


# def parse_midi_files_toEvents(file_list, parser, seq_len, parsed_data_path=None):
#     # Import necessary music21 classes


#     # Step 1: Parse the input monophonic music file into a music21 stream.
#     # (Replace 'path/to/file' with the actual path to your MusicXML or MIDI file)
#     file_path = "path/to/your/monophonic_music_file.musicxml"  # e.g., "melody.mid" or "score.musicxml"
#     score = converter.parse(file_path)

#     # Step 2: Flatten the stream to ensure all notes are in a single layer, sorted by time.
#     flat_score = score.flat

#     # Extract all Note objects from the flattened stream (ignore rests and other elements).
#     notes = flat_score.getElementsByClass(note.Note)

#     # Step 3: Collect note start (NOTE_ON) and note end (NOTE_OFF) events with their times.
#     events = []  # list to hold tuples of (time, event_string)
#     for n in notes:
#         pitch_name = n.nameWithOctave    # e.g., "C4", "F#5" - note name with octave
#         start_time = n.offset            # note start time (offset from beginning in quarterLength units)
#         duration = n.quarterLength       # note duration in quarterLength units
#         end_time = start_time + duration # note end time

#         # Record the NOTE_ON and NOTE_OFF events with their times
#         events.append((start_time, f"NOTE_ON({pitch_name})"))
#         events.append((end_time, f"NOTE_OFF({pitch_name})"))

#     # Step 4: Sort the events by time to get a chronological sequence.
#     events.sort(key=lambda x: x[0])

#     # Step 5: Iterate through sorted events and insert TIME_SHIFT events for gaps between events.
#     timeline = []   # final list of timeline events as strings
#     last_time = 0.0 # tracks the time of the last event processed
#     for time, event in events:
#         # Calculate the time difference from the previous event
#         delta = time - last_time
#         if delta > 0:
#             # If there is a gap, insert a TIME_SHIFT representing the rest or sustain duration
#             timeline.append(f"TIME_SHIFT({delta})")
#         # Append the current event (NOTE_ON or NOTE_OFF)
#         timeline.append(event)
#         # Update last_time to the current event's time
#         last_time = time

#     # Step 6: Output the timeline events.
#     # Print each event on a new line in the order they occur.
#     for evt in timeline:
#         print(evt)


def load_parsed_files(parsed_data_path):
    with open(os.path.join(parsed_data_path, "notes"), "rb") as f:
        notes = pkl.load(f)
    # Load durations if they exist, otherwise return empty list
    durations = []
    if os.path.exists(os.path.join(parsed_data_path, "durations")):
        with open(os.path.join(parsed_data_path, "durations"), "rb") as f:
            durations = pkl.load(f)
    return notes, durations

def get_midi_note(sample_note, sample_duration):
    new_note = None

    if "TS" in sample_note:
        new_note = music21.meter.TimeSignature(sample_note.split("TS")[0])

    elif "major" in sample_note or "minor" in sample_note:
        tonic, mode = sample_note.split(":")
        new_note = music21.key.Key(tonic, mode)

    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(
            float(Fraction(sample_duration))
        )
        new_note.storedInstrument = music21.instrument.Violoncello()

    elif "." in sample_note:
        notes_in_chord = sample_note.split(".")
        chord_notes = []
        for current_note in notes_in_chord:
            n = music21.note.Note(current_note)
            n.duration = music21.duration.Duration(
                float(Fraction(sample_duration))
            )
            n.storedInstrument = music21.instrument.Violoncello()
            chord_notes.append(n)
        new_note = music21.chord.Chord(chord_notes)

    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(
            float(Fraction(sample_duration))
        )
        new_note.storedInstrument = music21.instrument.Violoncello()

    elif sample_note != "START":
        new_note = music21.note.Note(sample_note)
        new_note.duration = music21.duration.Duration(
            float(Fraction(sample_duration))
        )
        new_note.storedInstrument = music21.instrument.Violoncello()

    return new_note


def reconstruct_midi_from_events(sample_note, tempo_bpm: float = 120.0, output_path: str = None):
    """
    Reconstructs a music21 Stream (and optionally writes a MIDI file) from an event timeline.

    Args:
        events: list of event strings as returned by extract_event_timeline().
        tempo_bpm: tempo in beats per minute for playback timing.
        output_path: if provided, path where the MIDI file will be written.

    Returns:
        music21.stream.Stream containing the reconstructed notes.
    """
    # s = stream.Stream()
    # s.append(midi.MetronomeMark(number=tempo_bpm))

    current_time = 0.0
    # Keep track of active notes: pitch_name -> start_time
    active = {}

    n = None
    if sample_note.startswith('TIME_SHIFT'):
        # Extract the delta and advance time
        delta = float(sample_note[sample_note.find('(')+1:sample_note.find(')')])
        current_time += delta
    elif sample_note.startswith('NOTE_ON'):
        pitch = sample_note[sample_note.find('(')+1:sample_note.find(')')]
        # Record the start time of this pitch
        active[pitch] = current_time
    elif sample_note.startswith('NOTE_OFF'):
        pitch = sample_note[sample_note.find('(')+1:sample_note.find(')')]
        if pitch in active:
            start = active.pop(pitch)
            dur = current_time - start
            # Create the note and set timing
            n = note.Note(pitch)
            n.offset = start
            n.quarterLength = dur
   
    return n

# Example usage:
# timeline = extract_event_timeline('path/to/mono.mid')
# reconstructed_stream = reconstruct_midi_from_events(timeline, tempo_bpm=100, output_path='recon.mid')

def get_midi_note_events(sample_note, sample_duration):
    new_note = None
    
    # TIME_SHIFT<dur>: insert a Rest of that length
    if sample_note.startswith("TIME_SHIFT"):
        # extract between “<” and “>”
        try:
            dur = float(sample_note.split("<")[1].split(">")[0])
        except:
            dur = float(sample_duration or 0.0)
        r = music21.note.Rest()
        r.duration = music21.duration.Duration(dur)
        r.storedInstrument = music21.instrument.Violoncello()
        return r

    # NOTE_OFF<pitch>: we rely on the TIME_SHIFT for timing, so skip
    if sample_note.startswith("NOTE_OFF"):
        return None

    # NOTE_ON<pitch>: start a new Note
    if sample_note.startswith("NOTE_ON"):
        midi = sample_note.split("<", 1)[1].split(">", 1)[0]
    
        n = music21.note.Note(midi)
        # n.pitch.midi = midi
        d = float(sample_duration) if sample_duration else 1.0
        n.duration = music21.duration.Duration(d)
        n.storedInstrument = music21.instrument.Violoncello()
        return n
    
    # if sample_note.startswith("CHORD_ON"):
    #     # extract between “<” and “>”
    #     pitches = sample_note.split("<")[1].split(">")[0].split(".")
    #     chord_notes = []
    #     for pitch in pitches:
    #         n = music21.note.Note(pitch)
    #         # n.duration = 
    #         n.storedInstrument = music21.instrument.Violoncello()
    #         chord_notes.append(n)
    #     return music21.chord.Chord(chord_notes)
    
    if sample_note.startswith("CHORD_OFF"):
        # we rely on the TIME_SHIFT for timing, so skip
        return None

    # fall through to your existing key/TS/rest/chord logic
    if "TS" in sample_note:
        new_note = music21.meter.TimeSignature(sample_note.split("TS")[0])

    elif "major" in sample_note or "minor" in sample_note:
        tonic, mode = sample_note.split(":")
        new_note = music21.key.Key(tonic, mode)

    # elif sample_note == "rest":
    #     new_note = music21.note.Rest()
    #     new_note.duration = music21.duration.Duration(
    #         float(Fraction(sample_duration))
    #     )
    #     new_note.storedInstrument = music21.instrument.Violoncello()

    # elif "." in sample_note:
    #     notes_in_chord = sample_note.split(".")
    #     chord_notes = []
    #     for current_note in notes_in_chord:
    #         n = music21.note.Note(current_note)
    #         n.duration = music21.duration.Duration(
    #             float(Fraction(sample_duration))
    #         )
    #         n.storedInstrument = music21.instrument.Violoncello()
    #         chord_notes.append(n)
    #     new_note = music21.chord.Chord(chord_notes)

    elif sample_note != "START":
        new_note = music21.note.Note(sample_note)
        new_note.duration = music21.duration.Duration(
            float(Fraction(sample_duration))
        )
        new_note.storedInstrument = music21.instrument.Violoncello()

    return new_note


class SinePositionEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding layer.
    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized
    in [Attention is All You Need](https://arxiv.org/abs/1706.03762).
    Takes as input an embedded token tensor. The input must have shape
    [batch_size, sequence_length, feature_size]. This layer will return a
    positional encoding the same size as the embedded token tensor, which
    can be added directly to the embedded token tensor.
    Args:
        max_wavelength: The maximum angular wavelength of the sine/cosine
            curves, as described in Attention is All You Need. Defaults to
            10000.
    Examples:
    ```python
    # create a simple embedding layer with sinusoidal positional encoding
    seq_len = 100
    vocab_size = 1000
    embedding_dim = 32
    inputs = keras.Input((seq_len,), dtype=tf.float32)
    embedding = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs)
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
    outputs = embedding + positional_encoding
    ```
    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs):
        # TODO(jbischof): replace `hidden_size` with`hidden_dim` for consistency
        # with other layers.
        input_shape = tf.shape(inputs)
        # length of sequence is the second last dimension of the inputs
        seq_length = input_shape[-2]
        hidden_size = input_shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        return tf.broadcast_to(positional_encodings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
