import os
import pickle as pkl
import music21
import keras
import tensorflow as tf

from fractions import Fraction


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
    events = []
    durations = []
    durations_list = []

    for i, file in enumerate(file_list):
        print(i + 1, "Parsing %s" % file)
        score = parser.parse(file).chordify()

        # start token
        events.append("START")

        for element in score.flat:
            # print(f"Parsing element {element} of type {type(element)}")
            if isinstance(element, music21.key.Key):
                # keep your old key tokens if you like
                tonic, mode = element.tonic.name, element.mode
                events.append(f"{tonic}:{mode}")
            elif isinstance(element, music21.meter.TimeSignature):
                events.append(f"{element.ratioString}TS")

            # elif isinstance(element, music21.note.Rest):
            #     dur = element.duration.quarterLength
            #     events.append(f"TIME_SHIFT<{dur}>")

            elif isinstance(element, music21.note.Note):
                midi = element.pitch.midi
                midi  = element.nameWithOctave
                dur = str(element.duration.quarterLength)
                events.append(f"NOTE_ON<{midi}>")
                events.append(f"TIME_SHIFT<{dur}>")
                durations.append(dur)
                events.append(f"NOTE_OFF<{midi}>")

            elif isinstance(element, music21.chord.Chord):
                # note_name = element.pitches[-1].nameWithOctave
                # duration_name = str(element.duration.quarterLength)
                # events.append(f"NOTE_ON<{note_name}>")
                # dur = element.duration.quarterLength
                # events.append(f"TIME_SHIFT<{dur}>")
                # events.append(f"NOTE_OFF<{note_name}>")

                # fire on for each pitch
                for p in element.pitches:
                    events.append(f"CHORD_ON<{p.nameWithOctave}>")
                events.append(f"TIME_SHIFT<{element.duration.quarterLength}>")
                # fire off for each pitch
                for p in element.pitches:
                    events.append(f"CHORD_OFF<{p.nameWithOctave}>")

        print(f"{len(events)} events parsed so far")
        # print(f"Events: {events[:10][0:20]} ...")

    print(f"Building sequences of length {seq_len}")
    for i in range(len(events) - seq_len):
        notes_list.append(" ".join(events[i : i + seq_len]))
    for i in range(len(durations) - seq_len):
        durations_list.append(" ".join(durations[i : i + seq_len]))

    if parsed_data_path:
        with open(os.path.join(parsed_data_path, "notes"), "wb") as f:
            pkl.dump(notes_list, f)
        with open(os.path.join(parsed_data_path, "durations"), "wb") as f:
            pkl.dump(durations_list, f)

    return notes_list, durations_list

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
    
    if sample_note.startswith("CHORD_ON"):
        # extract between “<” and “>”
        pitches = sample_note.split("<")[1].split(">")[0].split(".")
        chord_notes = []
        for pitch in pitches:
            n = music21.note.Note(pitch)
            # n.duration = 
            n.storedInstrument = music21.instrument.Violoncello()
            chord_notes.append(n)
        return music21.chord.Chord(chord_notes)
    
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
