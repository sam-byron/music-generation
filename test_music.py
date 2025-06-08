import os
import unittest
import numpy as np
import tensorflow as tf
import music21

# python

from music import (
    model,
    TokenAndPositionEmbedding,
    TransformerBlock,
    MusicGenerator
)

class TestModelCompilation(unittest.TestCase):
    def test_model_compiled(self):
        # Test that model has valid input, output, and an optimizer.
        self.assertIsNotNone(model.input, "Model input is None")
        self.assertIsNotNone(model.output, "Model output is None")
        self.assertIsNotNone(model.optimizer, "Model optimizer is not set")

class TestGetConfigMethods(unittest.TestCase):
    def test_transformer_block_get_config(self):
        # Instantiate a TransformerBlock and call get_config
        block = TransformerBlock(num_heads=2, key_dim=64, embed_dim=128, ff_dim=256, name="test_block")
        config = block.get_config()
        self.assertIsInstance(config, dict)
        # The config should include updated keys from the MultiHeadAttention layer
        # Exact keys depend on internal implementation. Check that config is not empty.
        self.assertTrue(len(config) > 0, "TransformerBlock.get_config returned empty config")

    def test_token_and_position_embedding_get_config(self):
        emb_layer = TokenAndPositionEmbedding(vocab_size=100, embed_dim=64)
        config = emb_layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("vocab_size", config)
        self.assertIn("embed_dim", config)

class DummyMusicGenerator(MusicGenerator):
    # Override sample_from to avoid while loop and return a fixed index.
    def sample_from(self, probs, temperature):
        return 2, probs  # Always return index 2

    # In testing, assign the global model to self.model to allow prediction calls.
    def set_dummy_model(self):
        self.model = model

class TestMusicGenerator(unittest.TestCase):
    def setUp(self):
        # Create a dummy vocabulary for notes and durations.
        self.notes_vocab = ["START", "dummy", "C4"]
        self.durations_vocab = ["0.0", "1.0", "0.5"]
        self.music_gen = MusicGenerator(self.notes_vocab, self.durations_vocab)
        # For testing callback methods, assign model manually.
        self.music_gen.model = model

    def test_sample_from(self):
        probs = np.array([0.1, 0.2, 0.7])
        idx, _ = self.music_gen.sample_from(probs, temperature=0.5)
        self.assertTrue(0 <= idx < len(probs), "sample_from returned invalid index")

    def test_get_note(self):
        # Use DummyMusicGenerator to override sample_from for deterministic behavior.
        dummy_gen = DummyMusicGenerator(self.notes_vocab, self.durations_vocab)
        dummy_gen.set_dummy_model()
        # Create dummy predictions with shape (1, 1, vocab_size) where last token prob forces index 2.
        notes_pred = np.array([[[0.0, 0.0, 1.0]]])
        durs_pred  = np.array([[[0.0, 0.0, 1.0]]])
        note_obj, idx, note_str, idx_d, dur_str = dummy_gen.get_note(notes_pred, durs_pred, temperature=0.5)
        self.assertEqual(idx, 2)
        self.assertEqual(note_str, "C4")
        self.assertEqual(idx_d, 2)
        self.assertEqual(dur_str, "0.5")
        self.assertTrue(isinstance(note_obj, music21.note.Note), "get_note did not return a music21.note.Note instance")

if __name__ == '__main__':
    unittest.main()