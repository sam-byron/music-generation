import os
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import music21
import sys
from tensorflow.keras.layers import TextVectorization

# ensure local transformer_utils.py is found first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from transformer_utils import (
    SinePositionEncoding,
)

from events_data_rep_v2 import (
    parse_monophonic_music as parse_midi_files,
    load_parsed_events as load_parsed_files,
    reconstruct_midi_from_events,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# (Optional) mixed precision to lower memory footprint
# mixed_precision.set_global_policy('mixed_float16')

# 0. Parameters
PARSE_MIDI_FILES    = True
PARSED_DATA_PATH   = "events_parsed_data/"
OUTPUT_MIDI_PATH = "evt_output/"
DATASET_REPETITIONS = 1
SHOW_MIDI = False

SEQ_LEN         = 400
EMBEDDING_DIM   = 256
KEY_DIM         = 256
N_HEADS         = 7
DROPOUT_RATE    = 0.3
FEED_FORWARD_DIM= 512

LOAD_MODEL      = False  # <-- switch on/off loading a pre‐trained model
EPOCHS          = 500
BATCH_SIZE      = 128

GENERATE_LEN    = 600


class StringVectorization(TextVectorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab_tensor = None

    def adapt(self, data, **kwargs):
        super().adapt(data, **kwargs)
        vocab = self.get_vocabulary()
        self._vocab_tensor = tf.constant(vocab, dtype=tf.string)

    def call(self, inputs):
        # convert inputs to string
        if isinstance(inputs, tf.Tensor):
            if inputs.dtype != tf.string:
                inputs = tf.strings.as_string(inputs)
        return super().call(inputs)

# 2. Tokenize the data
def create_dataset(elements, BATCH_SIZE=16):
    ds = (
        tf.data.Dataset.from_tensor_slices(elements)
        .batch(BATCH_SIZE, drop_remainder=True)
        .shuffle(1000)
    )
    vec = StringVectorization(standardize=None)
    vec.adapt(ds)
    return ds, vec, vec.get_vocabulary()

# def prepare_inputs(n):
#         tn = notes_vec(tf.expand_dims(n, -1))
#         return tn[:, :-1], tn[:, 1:]

def main():
    
    # 1. Prepare the Data
    file_list = glob.glob("./data/bach-cello/*.mid")
    print(f"Found {len(file_list)} MIDI files")

    # parser = music21.converter

    if PARSE_MIDI_FILES:
        notes = parse_midi_files(
            PARSED_DATA_PATH, SEQ_LEN
        )
    else:
        notes = load_parsed_files(PARSED_DATA_PATH)
    
    notes_ds, notes_vec, notes_vocab = create_dataset(notes)
    print(f"Notes vocabulary: {notes_vocab}")
    seq_ds = notes_ds

    # 3. Create the Training Set
    def prepare_inputs(n):
        tn = notes_vec(tf.expand_dims(n, -1))
        return tn[:, :-1], tn[:, 1:]

    ds = (
        seq_ds
        .map(prepare_inputs)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
        # .repeat(DATASET_REPETITIONS)
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
    evt_inputs = layers.Input((None,), dtype=tf.int32)
    evt_emb    = TokenAndPositionEmbedding(len(notes_vocab),
                                        EMBEDDING_DIM)(evt_inputs)
    x = evt_emb
    for i in range(3):
        x = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM,
                            FEED_FORWARD_DIM, name=f"attn{i+1}")(x)
    x = layers.Dense(len(notes_vocab), activation="softmax")(x)
    evt_out = layers.Dense(len(notes_vocab), activation="softmax")(x)
    model   = models.Model(evt_inputs, evt_out)
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=losses.SparseCategoricalCrossentropy()
    )

    model.summary()

    if LOAD_MODEL:
        model.load_weights("./checkpoint/checkpoint.weights.h5")

    # 9. MusicGenerator callback
    class MusicGenerator(callbacks.Callback):
        def __init__(self, index_to_note, top_k=10):
            super().__init__()
            self.index_to_note     = index_to_note
            self.note_to_index     = {n: i for i, n in enumerate(index_to_note)}
            self.top_k = top_k

        def sample_from(self, probs, temperature):
            p = probs ** (1/temperature)
            p = p / np.sum(p)
            return np.random.choice(len(p), p=p), p

        def get_note(self, notes_pred, temperature):
            # sample note
            idx_n, _ = 1, None
            while idx_n == 1:
                idx_n, _ = self.sample_from(notes_pred[0, -1], temperature)
            note = self.index_to_note[idx_n]

            return idx_n, note

        def generate(self, start_notes, max_tokens, temperature):
            # stream = music21.stream.Stream()
            # stream.append(music21.instrument.Violoncello())
            events = []

            tokens_n = [self.note_to_index.get(x, 1) for x in start_notes]

            for n in start_notes:
                # events.append(f"NOTE_ON({n})")
                events.append(n)

            i = 0
            
            # Pre-allocate the start
            while len(tokens_n) < max_tokens:
                temp = temperature + (i/max_tokens)*(1.0-temperature)
                x1 = np.array([tokens_n])
            
                # notes_pred = model.predict(x1, verbose=0)
                # this is presumably faster than predict
                notes_pred = model(x1, training=False).numpy()

                idx_n, note = self.get_note(
                    notes_pred, temp
                )
                events.append(note)
                tokens_n.append(idx_n)
                start_notes.append(note)

                # info.append({
                #     "prompt": [start_notes.copy()],
                #     "midi":   stream,
                #     "chosen_note": (note),
                #     "note_probs": notes_pred[0, -1],
                # })
                i += 1
            # Convert events to a music21 Stream
            print(events)
            stream = reconstruct_midi_from_events(
                events, tempo_bpm=120.0, output_path=OUTPUT_MIDI_PATH
            )

            return stream

        def on_epoch_end(self, epoch, logs=None):
            # if epoch % 20 == 0:
            stream = self.generate(["START"],
                                max_tokens=GENERATE_LEN,
                                temperature=0.5)
            midi = stream.chordify()
            if SHOW_MIDI:
                midi.show('text')
            out_fp = os.path.join(OUTPUT_MIDI_PATH, f"epoch-{epoch:04d}.mid")
            midi.write("midi", fp=out_fp)

    # 10. Train
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath="./checkpoint/checkpoint.weights.h5",
        save_weights_only=True, save_freq="epoch"
    )
    tensorboard_cb = callbacks.TensorBoard(log_dir="./logs")
    music_generator = MusicGenerator(notes_vocab)

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
        ["START"], max_tokens=50, temperature=0.5
    )
    out_stream = final_info[-1]["midi"].chordify()
    out_stream.show()
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_stream.write("midi", fp=os.path.join("output", f"final-{ts}.mid"))

if __name__ == "__main__":
    main()