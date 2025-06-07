# 🎶 Music Generation with Transformers
# In this script, we train a Transformer model to generate music in the style of the Bach cello suites.

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks

import music21

import sys, os
# ensure local transformer_utils.py is found before any installed package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from transformer_utils import (
    parse_midi_files_toEvents as parse_midi_files,
    load_parsed_files,
    get_midi_note,
    SinePositionEncoding,
)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable dynamic memory allocation
            tf.config.experimental.set_memory_growth(gpu, True)
        # Alternatively, limit memory usage explicitly (e.g., 4GB):
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        # )
    except RuntimeError as e:
        print(e)

# 0. Parameters
PARSE_MIDI_FILES   = False
PARSED_DATA_PATH   = "parsed_data/"
DATASET_REPETITIONS = 1

SEQ_LEN = 600
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 6
DROPOUT_RATE = 0.3
FEED_FORWARD_DIM = 512
LOAD_MODEL = False
USE_DURATIONS = False
# optimization
EPOCHS = 500
BATCH_SIZE = 128

GENERATE_LEN = 600

# 1. Prepare the Data
file_list = glob.glob("./data/bach-cello/*.mid")
print(f"Found {len(file_list)} midi files")

parser = music21.converter

example_score = (
    music21.converter.parse(file_list[1])
    .splitAtQuarterLength(12)[0]
    .chordify()
)
# example_score.show()
# example_score.show("text")

if PARSE_MIDI_FILES:
    notes, durations = parse_midi_files(
        file_list, parser, SEQ_LEN + 1, PARSED_DATA_PATH
    )
else:
    notes, durations = load_parsed_files(PARSED_DATA_PATH)

# if we’re only doing events, drop durations
if not USE_DURATIONS:
    durations = ["0.0"] * len(notes)

example_notes     = notes[658]
example_durations = durations[658]
print("\nNotes string\n", example_notes, "...")
print("\nDuration string\n", example_durations, "...")

# 2. Tokenize the data
def create_dataset(elements):
    ds = (
        tf.data.Dataset.from_tensor_slices(elements)
        .batch(BATCH_SIZE, drop_remainder=True)
        .shuffle(1000)
    )
    vec = layers.TextVectorization(standardize=None, output_mode="int")
    vec.adapt(ds)
    return ds, vec, vec.get_vocabulary()

notes_ds, notes_vec, notes_vocab = create_dataset(notes)
if USE_DURATIONS:
    durs_ds, durs_vec, durs_vocab = create_dataset(durations)
    seq_ds = tf.data.Dataset.zip((notes_ds, durs_ds))
else:
    seq_ds = notes_ds

# Display example tokens
example_tokenised_notes     = notes_vec(example_notes)
example_tokenised_durations = durations if not USE_DURATIONS else durs_vec(example_durations)
print(f"{'note token':10} {'duration token':10}")
for n, d in zip(example_tokenised_notes.numpy()[:11], example_tokenised_durations[:11] if not USE_DURATIONS else example_tokenised_durations.numpy()[:11]):
    print(f"{n:10}{d:10}")

notes_vocab_size     = len(notes_vocab)
durations_vocab_size = len(durations_vocab) if USE_DURATIONS else 0
print(f"\nNOTES_VOCAB: length = {notes_vocab_size}")
for i, note in enumerate(notes_vocab[:10]):
    print(f"{i}: {note}")
if USE_DURATIONS:
    print(f"\nDURATIONS_VOCAB: length = {durations_vocab_size}")
    for i, dur in enumerate(durations_vocab[:10]):
        print(f"{i}: {dur}")

# 3. Create the Training Set
if USE_DURATIONS:
    def prepare_inputs(n, d):
        n = tf.expand_dims(n, -1); d = tf.expand_dims(d, -1)
        tn = notes_vec(n); td = durs_vec(d)
        return (tn[:, :-1], td[:, :-1]), (tn[:, 1:], td[:, 1:])
else:
    def prepare_inputs(n):
        tn = notes_vec(tf.expand_dims(n, -1))
        return tn[:, :-1], tn[:, 1:]

ds = seq_ds.map(prepare_inputs).cache().prefetch(tf.data.AUTOTUNE).repeat(DATASET_REPETITIONS)
example_input_output = ds.take(1).get_single_element()
print(example_input_output)

# 5. Causal attention mask
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1,1], dtype=tf.int32)],
        0
    )
    return tf.tile(mask, mult)

# 6. Transformer Block layer
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, name, dropout_rate=DROPOUT_RATE):
        super().__init__(name=name)
        self.attn      = layers.MultiHeadAttention(num_heads, key_dim, output_shape=embed_dim)
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.ln_1      = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1     = layers.Dense(ff_dim, activation="relu")
        self.ffn_2     = layers.Dense(embed_dim)
        self.dropout_2 = layers.Dropout(dropout_rate)
        self.ln_2      = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attn_output, _ = self.attn(
            inputs, inputs,
            attention_mask=causal_mask,
            return_attention_scores=True
        )
        x = self.ln_1(inputs + self.dropout_1(attn_output))
        ffn = self.dropout_2(self.ffn_2(self.ffn_1(x)))
        return self.ln_2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(self.attn.get_config())
        return cfg

# 7. Token and Position Embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer="he_uniform")
        self.pos_emb   = SinePositionEncoding()

    def call(self, x):
        t = self.token_emb(x)
        p = self.pos_emb(t)
        return t + p

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocab_size": self.token_emb.input_dim, "embed_dim": self.token_emb.output_dim})
        return cfg

# 8. Build the Transformer model
if USE_DURATIONS:
    # — your existing two‐input model (note_inputs, dur_inputs → two outputs) —
    note_inputs = layers.Input((None,), dtype=tf.int32)
    dur_inputs  = layers.Input((None,), dtype=tf.int32)
    note_emb    = TokenAndPositionEmbedding(len(notes_vocab), EMBEDDING_DIM//2)(note_inputs)
    dur_emb     = TokenAndPositionEmbedding(len(durs_vocab), EMBEDDING_DIM//2)(dur_inputs)
    x = layers.Concatenate()([note_emb, dur_emb])
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn1")(x)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn2")(x)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn3")(x)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn4")(x)
    note_out = layers.Dense(len(notes_vocab), activation="softmax")(x)
    dur_out  = layers.Dense(len(durs_vocab),  activation="softmax")(x)
    model = models.Model([note_inputs, dur_inputs], [note_out, dur_out])
    model.compile(
        optimizer="adam",
        loss=[losses.SparseCategoricalCrossentropy(), losses.SparseCategoricalCrossentropy()]
    )
else:
    # single‐input, single‐output model
    evt_inputs = layers.Input((None,), dtype=tf.int32)
    evt_emb    = TokenAndPositionEmbedding(len(notes_vocab), EMBEDDING_DIM)(evt_inputs)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn1")(evt_emb)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn2")(x)
    x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn3")(x)
    # x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attn4")(x)
    evt_out = layers.Dense(len(notes_vocab), activation="softmax")(x)
    model   = models.Model(evt_inputs, evt_out)
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy()
    )

model.summary()

if LOAD_MODEL:
    model.load_weights("./checkpoint/checkpoint.weights.h5")

# 9. Train the Transformer
class MusicGenerator(callbacks.Callback):
    def __init__(self, index_to_note, index_to_duration, top_k=10):
        super().__init__()
        self.index_to_note     = index_to_note
        self.note_to_index     = {n: i for i, n in enumerate(index_to_note)}
        self.index_to_duration = index_to_duration
        self.duration_to_index = {d: i for i, d in enumerate(index_to_duration)}

    def sample_from(self, probs, temperature):
        p = probs ** (1/temperature)
        p = p / np.sum(p)
        return np.random.choice(len(p), p=p), p

    def get_note(self, notes, durs, temperature):
        idx, _ = 1, None
        while idx == 1:
            idx, _ = self.sample_from(notes[0][-1], temperature)
        note = self.index_to_note[idx]

        idx_d, _ = 1, None
        while idx_d == 1:
            idx_d, _ = self.sample_from(durs[0][-1], temperature)
        dur = self.index_to_duration[idx_d]

        return get_midi_note(note, dur), idx, note, idx_d, dur

    def generate(self, start_notes, start_durations, max_tokens, temperature):
        midi_stream = music21.stream.Stream()
        midi_stream.append(music21.instrument.Violoncello())
        tokens_n = [self.note_to_index.get(x, 1) for x in start_notes]
        tokens_d = [self.duration_to_index.get(x, 1) for x in start_durations]

        # Append initial seed tokens to midi stream
        for n, d in zip(start_notes, start_durations):
            note_obj = get_midi_note(n, d)
            if note_obj is not None:
                midi_stream.append(note_obj)

        info = []
        i = 0  # counter for gradual temperature increase
        while len(tokens_n) < max_tokens:
            # Gradually increase temperature
            temp = temperature + (i / max_tokens) * (1.0 - temperature)
            x1 = np.array([tokens_n])
            x2 = np.array([tokens_d])
            notes_pred, durs_pred = self.model.predict([x1, x2], verbose=0)
            note_obj = self.get_note(notes_pred, durs_pred, temp)
            new_note, idx_n, n_str, idx_d, d_str = note_obj
            if new_note is not None:
                midi_stream.append(new_note)
            tokens_n.append(idx_n)
            tokens_d.append(idx_d)
            start_notes.append(n_str)
            start_durations.append(d_str)

            info.append({
                "prompt": [start_notes.copy(), start_durations.copy()],
                "midi": midi_stream,
                "chosen_note": (n_str, d_str),
                "note_probs": notes_pred[0][-1],
                "duration_probs": durs_pred[0][-1],
                "atts": None,  # skip attention tracking here
            })
            if n_str == "START":  # or other termination condition
                break
            i += 1
        return info

    def generate_in_chunks(self, total_tokens, temperature):
        chunk_length = total_tokens // 3
        # Chunk 1: seed from fixed tokens
        chunk1_info = self.generate(["START"], ["0.0"], max_tokens=chunk_length, temperature=temperature)
        prompt1_notes, prompt1_durs = chunk1_info[-1]["prompt"]

        # Seed Chunk 2 with second half of chunk1 tokens
        half1 = len(prompt1_notes) // 2
        seed_notes2 = prompt1_notes[half1:].copy()
        seed_durations2 = prompt1_durs[half1:].copy()
        chunk2_info = self.generate(seed_notes2, seed_durations2, max_tokens=chunk_length, temperature=temperature)
        prompt2_notes, prompt2_durs = chunk2_info[-1]["prompt"]

        # Seed Chunk 3 with second half of chunk2 tokens
        half2 = len(prompt2_notes) // 2
        seed_notes3 = prompt2_notes[half2:].copy()
        seed_durations3 = prompt2_durs[half2:].copy()
        chunk3_info = self.generate(seed_notes3, seed_durations3, max_tokens=chunk_length, temperature=temperature)
        prompt3_notes, prompt3_durs = chunk3_info[-1]["prompt"]

        # Concatenate by taking all tokens from chunk1 then appending the new tokens from chunks 2 and 3
        # (avoiding duplicate seed tokens)
        full_notes = prompt1_notes + prompt2_notes[half2:] + prompt3_notes[len(prompt3_notes)//2:]
        full_durations = prompt1_durs + prompt2_durs[half2:] + prompt3_durs[len(prompt3_durs)//2:]

        # Build full midi stream from concatenated tokens
        full_midi = music21.stream.Stream()
        full_midi.append(music21.instrument.Violoncello())
        for n, d in zip(full_notes, full_durations):
            note_obj = get_midi_note(n, d)
            if note_obj is not None:
                full_midi.append(note_obj)

        return {"prompt": [full_notes, full_durations], "midi": full_midi}

    def on_epoch_end(self, epoch, logs=None):
        # Generate full sequence in three chunks, then concatenate
        full_info = self.generate_in_chunks(total_tokens=GENERATE_LEN, temperature=0.5)
        midi = full_info["midi"].chordify()
        # Optionally print the concatenated prompt
        # print("Combined prompt:", full_info["prompt"])
        midi.show()
        midi.write("midi", fp=os.path.join("output", f"output-{epoch:04d}.mid"))

# Callbacks & training
model_checkpoint = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.weights.h5",  # updated extension to .weights.h5
    save_weights_only=True, save_freq="epoch", verbose=0
)
tensorboard_cb = callbacks.TensorBoard(log_dir="./logs")
music_generator = MusicGenerator(notes_vocab, durations_vocab if USE_DURATIONS else [])

# Split the zipped dataset into train and validation
total_batches = len(notes) // BATCH_SIZE
val_batches = max(1, int(0.1 * total_batches))
train_ds = ds.skip(val_batches)
val_ds = ds.take(val_batches)

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[model_checkpoint, tensorboard_cb, music_generator]
)

# Save the final model
model.save("./models/model")

# 3. Generate music using the Transformer
info = music_generator.generate(["START"], ["0.0"], max_tokens=50, temperature=0.5)
midi_stream = info[-1]["midi"].chordify()
midi_stream.show()

# Write music to MIDI file
timestr = time.strftime("%Y%m%d-%H%M%S")
midi_stream.write("midi", fp=os.path.join("output", f"output-{timestr}.mid"))

# Note probabilities plot (optional)
# Attention plot (optional)