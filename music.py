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
    parse_midi_files,
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
PARSE_MIDI_FILES   = True
PARSED_DATA_PATH   = "parsed_data/"
DATASET_REPETITIONS = 1

SEQ_LEN = 200
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 6
DROPOUT_RATE = 0.3
FEED_FORWARD_DIM = 512
LOAD_MODEL = False

# optimization
EPOCHS = 500
BATCH_SIZE = 128

GENERATE_LEN = 200

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
    notes, durations = load_parsed_files("./parsed_data")

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
    vectorize_layer = layers.TextVectorization(
        standardize=None, output_mode="int"
    )
    # run adapt on CPU only:
    # with tf.device('/CPU:0'):
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()
    return ds, vectorize_layer, vocab

notes_seq_ds, notes_vectorize_layer, notes_vocab = create_dataset(notes)
durations_seq_ds, durations_vectorize_layer, durations_vocab = create_dataset(durations)
seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

# Display example tokens
example_tokenised_notes     = notes_vectorize_layer(example_notes)
example_tokenised_durations = durations_vectorize_layer(example_durations)
print(f"{'note token':10} {'duration token':10}")
for n, d in zip(example_tokenised_notes.numpy()[:11], example_tokenised_durations.numpy()[:11]):
    print(f"{n:10}{d:10}")

notes_vocab_size     = len(notes_vocab)
durations_vocab_size = len(durations_vocab)
print(f"\nNOTES_VOCAB: length = {notes_vocab_size}")
for i, note in enumerate(notes_vocab[:10]):
    print(f"{i}: {note}")
print(f"\nDURATIONS_VOCAB: length = {durations_vocab_size}")
for i, dur in enumerate(durations_vocab[:10]):
    print(f"{i}: {dur}")

# 3. Create the Training Set
def prepare_inputs(notes, durations):
    notes = tf.expand_dims(notes, -1)
    durations = tf.expand_dims(durations, -1)
    tokenized_notes     = notes_vectorize_layer(notes)
    tokenized_durations = durations_vectorize_layer(durations)
    x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
    y = (tokenized_notes[:, 1:],   tokenized_durations[:, 1:])
    return x, y

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

# 8. Build the Transformer model with two Transformer blocks
note_inputs      = layers.Input(shape=(None,), dtype=tf.int32)
dur_inputs       = layers.Input(shape=(None,), dtype=tf.int32)
note_emb         = TokenAndPositionEmbedding(notes_vocab_size, EMBEDDING_DIM//2)(note_inputs)
dur_emb          = TokenAndPositionEmbedding(durations_vocab_size, EMBEDDING_DIM//2)(dur_inputs)
embeddings       = layers.Concatenate()([note_emb, dur_emb])
x                = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attention_block1")(embeddings)
x                = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attention_block2")(x)
note_outputs     = layers.Dense(notes_vocab_size, activation="softmax",   name="note_outputs")(x)
duration_outputs = layers.Dense(durations_vocab_size, activation="softmax", name="duration_outputs")(x)

model = models.Model(inputs=[note_inputs, dur_inputs], outputs=[note_outputs, duration_outputs])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=[losses.SparseCategoricalCrossentropy(), losses.SparseCategoricalCrossentropy()]
)
model.summary()

if LOAD_MODEL:
    model.load_weights("./checkpoint/checkpoint.weights.h5")

# 9. Train the Transformer
class MusicGenerator(callbacks.Callback):
    def __init__(self, index_to_note, index_to_duration, top_k=10):
        super().__init__()
        self.index_to_note     = index_to_note
        self.note_to_index     = {n:i for i,n in enumerate(index_to_note)}
        self.index_to_duration = index_to_duration
        self.duration_to_index = {d:i for i,d in enumerate(index_to_duration)}

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

    def on_epoch_end(self, epoch, logs=None):
        info = self.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.5)
        midi = info[-1]["midi"].chordify()
        # print(info[-1]["prompt"])
        midi.show()
        midi.write("midi", fp=os.path.join("output", f"output-{epoch:04d}.mid"))

    # def on_epoch_end(self, epoch, logs=None):
    #     for sample_num in range(3):
    #         info = self.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.5)
    #         midi = info[-1]["midi"].chordify()
    #         print(f"Epoch {epoch}, Sample {sample_num}: {info[-1]['prompt']}")
    #         midi.show()
    #         filename = os.path.join("output", f"output-{epoch}-{sample_num}.mid")
    #         midi.write("midi", fp=filename)

    # import threading
    # def on_epoch_end(self, epoch, logs=None):
    #     def generate_and_save(epoch):
    #         for sample_num in range(3):
    #             info = self.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.5)
    #             midi = info[-1]["midi"].chordify()
    #             print(f"Epoch {epoch}, Sample {sample_num}: {info[-1]['prompt']}")
    #             midi.show()
    #             filename = os.path.join("output", f"output-{epoch}-{sample_num}.mid")
    #             midi.write("midi", fp=filename)
    #     threading.Thread(target=generate_and_save, args=(epoch,), daemon=True).start()

    def generate(self, start_notes, start_durations, max_tokens, temperature):
        midi_stream = music21.stream.Stream()
        midi_stream.append(music21.clef.BassClef())
        tokens_n = [self.note_to_index.get(x, 1) for x in start_notes]
        tokens_d = [self.duration_to_index.get(x, 1) for x in start_durations]

        for n, d in zip(start_notes, start_durations):
            note_obj = get_midi_note(n, d)
            if note_obj is not None:
                midi_stream.append(note_obj)

        info = []
        while len(tokens_n) < max_tokens:
            # Gradually increase temperature
            temp = temperature + (i / max_tokens) * (1.0 - temperature)
            x1 = np.array([tokens_n]); x2 = np.array([tokens_d])
            notes_pred, durs_pred = self.model.predict([x1, x2], verbose=0)
            note_obj = self.get_note(notes_pred, durs_pred, temp)
            new_note, idx_n, n_str, idx_d, d_str = note_obj
            if new_note is not None:
                midi_stream.append(new_note)
            tokens_n.append(idx_n); tokens_d.append(idx_d)
            start_notes.append(n_str); start_durations.append(d_str)

            info.append({
                "prompt": [start_notes.copy(), start_durations.copy()],
                "midi": midi_stream,
                "chosen_note": (n_str, d_str),
                "note_probs": notes_pred[0][-1],
                "duration_probs": durs_pred[0][-1],
                "atts": None,  # skip attention tracking here
            })
            if n_str == "START":
                break
        return info

# Callbacks & training
model_checkpoint = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.weights.h5",  # updated extension to .weights.h5
    save_weights_only=True, save_freq="epoch", verbose=0
)
tensorboard_cb = callbacks.TensorBoard(log_dir="./logs")
music_generator = MusicGenerator(notes_vocab, durations_vocab)

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