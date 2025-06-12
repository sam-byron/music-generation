import os
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import music21
import sys

# pus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# (Optional) mixed precision to lower memory footprint
# mixed_precision.set_global_policy('mixed_float16')

# ensure local transformer_utils.py is found first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from transformer_utils import (
    parse_midi_files_toEvents as parse_midi_files,
    # parse_midi_files,
    load_parsed_files,
    get_midi_note,
    SinePositionEncoding,
    get_midi_note_events,
    reconstruct_midi_from_events,
)

# 0. Parameters
PARSE_MIDI_FILES    = True
PARSED_DATA_PATH    = "parsed_data/"
DATASET_REPETITIONS = 1

SEQ_LEN         = 600
EMBEDDING_DIM   = 256
KEY_DIM         = 256
N_HEADS         = 5
DROPOUT_RATE    = 0.3
FEED_FORWARD_DIM= 512

LOAD_MODEL      = False  # <-- switch on/off loading a pre‐trained model
USE_DURATIONS   = False   # <-- switch on/off two‐stream durations
EPOCHS          = 500
BATCH_SIZE      = 32

GENERATE_LEN    = 600

# 0.5. GPU setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Prepare the Data
file_list = glob.glob("./data/bach-cello/*.mid")
print(f"Found {len(file_list)} MIDI files")

parser = music21.converter

if PARSE_MIDI_FILES:
    notes = parse_midi_files(
        file_list, parser, SEQ_LEN + 1, PARSED_DATA_PATH
    )
else:
    notes, durations = load_parsed_files(PARSED_DATA_PATH)

# if only events, drop durations stream
if not USE_DURATIONS:
    durations = ["0.0"] * len(notes)

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
    # define empty duration vocab so downstream code can refer to it safely
    durs_vec = None
    durs_vocab = []
    seq_ds = notes_ds

# 3. Create the Training Set
if USE_DURATIONS:
    def prepare_inputs(n, d):
        n = tf.expand_dims(n, -1)
        d = tf.expand_dims(d, -1)
        tn = notes_vec(n)
        td = durs_vec(d)
        return (tn[:, :-1], td[:, :-1]), (tn[:, 1:], td[:, 1:])
else:
    def prepare_inputs(n):
        tn = notes_vec(tf.expand_dims(n, -1))
        return tn[:, :-1], tn[:, 1:]

ds = (
    seq_ds
    .map(prepare_inputs)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
    .repeat(DATASET_REPETITIONS)
)

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

# 6. Transformer Block
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
        batch_size = tf.shape(inputs)[0]
        seq_len    = tf.shape(inputs)[1]
        mask       = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attn_output, _ = self.attn(inputs, inputs,
                                   attention_mask=mask,
                                   return_attention_scores=True)
        x   = self.ln_1(inputs + self.dropout_1(attn_output))
        ffn = self.dropout_2(self.ffn_2(self.ffn_1(x)))
        return self.ln_2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(self.attn.get_config())
        return cfg

# 7. Token + Position Embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim,
                                          embeddings_initializer="he_uniform")
        self.pos_emb   = SinePositionEncoding()

    def call(self, x):
        t = self.token_emb(x)
        p = self.pos_emb(t)
        return t + p

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.token_emb.input_dim,
            "embed_dim":  self.token_emb.output_dim
        })
        return cfg

# 8. Build the Transformer model
if USE_DURATIONS:
    note_inputs = layers.Input((None,), dtype=tf.int32)
    dur_inputs  = layers.Input((None,), dtype=tf.int32)
    note_emb    = TokenAndPositionEmbedding(len(notes_vocab),
                                            EMBEDDING_DIM//2)(note_inputs)
    dur_emb     = TokenAndPositionEmbedding(len(durs_vocab),
                                            EMBEDDING_DIM//2)(dur_inputs)
    x = layers.Concatenate()([note_emb, dur_emb])
    for i in range(1):
        x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM,
                             FEED_FORWARD_DIM, name=f"attn{i+1}")(x)
    note_out = layers.Dense(len(notes_vocab), activation="softmax")(x)
    dur_out  = layers.Dense(len(durs_vocab),  activation="softmax")(x)
    model    = models.Model([note_inputs, dur_inputs], [note_out, dur_out])
    model.compile(
        optimizer="adam",
        loss=[losses.SparseCategoricalCrossentropy(),
              losses.SparseCategoricalCrossentropy(),
              run_eagerly-True]
    )
else:
    evt_inputs = layers.Input((None,), dtype=tf.int32)
    evt_emb    = TokenAndPositionEmbedding(len(notes_vocab),
                                           EMBEDDING_DIM)(evt_inputs)
    x = evt_emb
    for i in range(1):
        x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM,
                             FEED_FORWARD_DIM, name=f"attn{i+1}")(x)
    evt_out = layers.Dense(len(notes_vocab), activation="softmax")(x)
    model   = models.Model(evt_inputs, evt_out)
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy()
    )

model.summary()

if LOAD_MODEL:
    model.load_weights("./checkpoint/checkpoint.weights.h5")

# 9. MusicGenerator callback
class MusicGenerator(callbacks.Callback):
    def __init__(self, index_to_note, index_to_duration, top_k=10):
        super().__init__()
        self.index_to_note     = index_to_note
        self.note_to_index     = {n: i for i, n in enumerate(index_to_note)}
        self.index_to_duration = index_to_duration
        self.duration_to_index = {d: i for i, d in enumerate(index_to_duration)}
        self.use_durations     = len(index_to_duration) > 0

    def sample_from(self, probs, temperature):
        p = probs ** (1/temperature)
        p = p / np.sum(p)
        return np.random.choice(len(p), p=p), p

    def get_note(self, notes_pred, durs_pred, temperature):
        # sample note
        idx_n, _ = 1, None
        while idx_n == 1:
            idx_n, _ = self.sample_from(notes_pred[0, -1], temperature)
        n_str = self.index_to_note[idx_n]

        # sample duration if used
        if self.use_durations:
            idx_d, _ = 1, None
            while idx_d == 1:
                idx_d, _ = self.sample_from(durs_pred[0, -1], temperature)
            d_str = self.index_to_duration[idx_d]
        else:
            idx_d, d_str = None, None
        if self.use_durations:
            return get_midi_note(n_str, d_str), idx_n, n_str, idx_d, d_str
        else:
            return reconstruct_midi_from_events(n_str, None), idx_n, n_str, None, None

    def generate(self, start_notes, start_durations, max_tokens, temperature):
        stream = music21.stream.Stream()
        stream.append(music21.instrument.Violoncello())

        tokens_n = [self.note_to_index.get(x, 1) for x in start_notes]
        tokens_d = [self.duration_to_index.get(x, 1) for x in start_durations]

        for n, d in zip(start_notes, start_durations):
            if USE_DURATIONS:
                note_obj = get_midi_note(n, d)
            else:
                note_obj = reconstruct_midi_from_events(n, None)
            if note_obj is not None:
                stream.append(note_obj)

        info, i = [], 0
        while len(tokens_n) < max_tokens:
            temp = temperature + (i/max_tokens)*(1.0-temperature)
            x1 = np.array([tokens_n])
            if self.use_durations:
                x2 = np.array([tokens_d])
                notes_pred, durs_pred = model.predict([x1, x2], verbose=0)
            else:
                notes_pred = model.predict(x1, verbose=0)
                durs_pred  = None

            note_obj, idx_n, n_str, idx_d, d_str = self.get_note(
                notes_pred, durs_pred, temp
            )
            if note_obj is not None:
                stream.append(note_obj)

            tokens_n.append(idx_n)
            start_notes.append(n_str)
            if self.use_durations:
                tokens_d.append(idx_d)
                start_durations.append(d_str)

            info.append({
                "prompt": [start_notes.copy(), start_durations.copy()],
                "midi":   stream,
                "chosen_note": (n_str, d_str),
                "note_probs": notes_pred[0, -1],
                "duration_probs": durs_pred[0, -1] if durs_pred is not None else None
            })
            i += 1

        return info

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 20 == 0:
        full = self.generate(["START"], ["0.0"],
                            max_tokens=GENERATE_LEN,
                            temperature=0.5)
        midi = full[-1]["midi"].chordify()
        midi.show()
        out_fp = os.path.join("output", f"epoch-{epoch:04d}.mid")
        midi.write("midi", fp=out_fp)

# 10. Train
model_checkpoint = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.weights.h5",
    save_weights_only=True, save_freq="epoch"
)
tensorboard_cb = callbacks.TensorBoard(log_dir="./logs")
music_generator = MusicGenerator(notes_vocab, durs_vocab)

# split train/val
total_batches = len(notes) // BATCH_SIZE
val_batches   = max(1, int(0.1 * total_batches))
train_ds      = ds.skip(val_batches)
val_ds        = ds.take(val_batches)

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[model_checkpoint, tensorboard_cb, music_generator]
)

# save final model
model.save("./models/model")

# 11. One‐off generation
final_info = music_generator.generate(
    ["START"], ["0.0"], max_tokens=50, temperature=0.5
)
out_stream = final_info[-1]["midi"].chordify()
out_stream.show()
ts = time.strftime("%Y%m%d-%H%M%S")
out_stream.write("midi", fp=os.path.join("output", f"final-{ts}.mid"))