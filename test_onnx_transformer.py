import unittest
import numpy as np
import math

# Import the functions and classes from your main project file.
# Adjust the import below to match your project structure.
from onnx_transformer_with_attention import (
    scaled_dot_product_attention,
    ONNXTransformer,
)

# Create a fake transformer subclass to override the model inference,
# so that tests run without a real ONNX model.
class FakeONNXTransformer(ONNXTransformer):
    def __init__(self, num_layers=2, hidden_dim=8, page_size=4):
        # Do not load a real ONNX model; just set the parameters.
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.page_size = page_size
        self.kv_cache = {layer: {"keys": [], "values": []} for layer in range(num_layers)}

    def forward(self, input_ids):
        """Simulate forward pass with predictable dummy outputs."""
        batch_size = input_ids.shape[0]
        vocab_size = 1000
        # Create dummy logits where each row is identical.
        logits = np.full((batch_size, vocab_size), 0.5, dtype=np.float32)
        # For each transformer layer, return a constant tensor as new key and value.
        new_keys = {
            layer: np.full((batch_size, 1, self.hidden_dim), fill_value=layer + 1, dtype=np.float32)
            for layer in range(self.num_layers)
        }
        new_values = {
            layer: np.full((batch_size, 1, self.hidden_dim), fill_value=layer + 10, dtype=np.float32)
            for layer in range(self.num_layers)
        }
        return logits, new_keys, new_values

class TestAttentionFunctions(unittest.TestCase):
    def test_scaled_dot_product_attention_shapes(self):
        """Verify that attention output and weights have correct shapes."""
        # Create simple query, key and value tensors.
        batch_size, num_queries, d_k, num_keys, d_v = 1, 3, 4, 5, 6
        query = np.random.randn(batch_size, num_queries, d_k).astype(np.float32)
        key = np.random.randn(batch_size, num_keys, d_k).astype(np.float32)
        value = np.random.randn(batch_size, num_keys, d_v).astype(np.float32)

        output, weights = scaled_dot_product_attention(query, key, value)
        self.assertEqual(output.shape, (batch_size, num_queries, d_v))
        self.assertEqual(weights.shape, (batch_size, num_queries, num_keys))
        # Check that attention weights sum to 1 along the num_keys dimension.
        np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones((batch_size, num_queries)), atol=1e-5)

    def test_cache_update_and_flatten(self):
        """Test that updating the cache produces the expected number of pages."""
        transformer = FakeONNXTransformer(num_layers=2, hidden_dim=8, page_size=4)
        
        # Simulate two forward passes.
        dummy_input = np.array([[1, 2, 3]])
        _, new_keys, new_values = transformer.forward(dummy_input)
        
        # First update: pages should be created for each layer.
        transformer.update_cache(new_keys, new_values)
        for layer in range(transformer.num_layers):
            self.assertEqual(len(transformer.kv_cache[layer]["keys"]), 1)
        
        # Second update: if the page is not yet full, it should be concatenated.
        transformer.update_cache(new_keys, new_values)
        for layer in range(transformer.num_layers):
            # The first page should now have shape (batch, 2, hidden_dim)
            keys_page = transformer.kv_cache[layer]["keys"][0]
            self.assertEqual(keys_page.shape[1], 2)
        
        # Force creation of a second page by updating repeatedly.
        for _ in range(3):
            transformer.update_cache(new_keys, new_values)
        # Check that for at least one layer, a second page has been created.
        for layer in range(transformer.num_layers):
            self.assertTrue(len(transformer.kv_cache[layer]["keys"]) >= 2)
        
        # Test get_flat_cache returns concatenated tensors.
        keys, values = transformer.get_flat_cache(0)
        expected_length = sum(page.shape[1] for page in transformer.kv_cache[0]["keys"])
        self.assertEqual(keys.shape[1], expected_length)
        self.assertEqual(values.shape[1], expected_length)

    def test_autoregressive_generation_loop(self):
        """Simulate two autoregressive passes and verify that new tokens are generated."""
        transformer = FakeONNXTransformer(num_layers=2, hidden_dim=8, page_size=4)
        # Start with a prompt of a couple tokens.
        generated_tokens = [10, 20]
        # Use a dummy input_ids.
        input_ids = np.array(generated_tokens).reshape(1, -1)
        
        # Perform a forward pass to simulate generating one token.
        logits, new_keys, new_values = transformer.forward(input_ids)
        # Using our dummy forward, logits will be constant (0.5 for all tokens)
        # For testing, we can simply choose token 0 (since argmax of constant array is 0).
        next_token = int(np.argmax(logits[0]))
        generated_tokens.append(next_token)
        transformer.update_cache(new_keys, new_values)
        
        # Check that the generated_tokens list length increased.
        self.assertEqual(len(generated_tokens), 3)
        # Verify that the cache for each layer has been updated.
        for layer in range(transformer.num_layers):
            keys, values = transformer.get_flat_cache(layer)
            self.assertIsNotNone(keys)
            self.assertGreaterEqual(keys.shape[1], 1)

if __name__ == '__main__':
    unittest.main()
