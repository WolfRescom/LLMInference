#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import argparse
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Real scaled dot-product attention.
    Args:
        query: shape (batch_size, num_queries, d_k)
        key: shape (batch_size, num_keys, d_k)
        value: shape (batch_size, num_keys, d_v)
        mask: (optional) broadcastable mask for attention scores.
    Returns:
        attention_output: shape (batch_size, num_queries, d_v)
        attention_weights: shape (batch_size, num_queries, num_keys)
    """
    d_k = query.shape[-1]
    # Compute raw attention scores.
    scores = np.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    # Apply softmax.
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    # Compute weighted sum of values.
    attention_output = np.matmul(attention_weights, value)
    return attention_output, attention_weights

class ONNXTransformer:
    def __init__(self, model_path, num_layers=4, hidden_dim=32, page_size=8):
        """
        Initialize the ONNX transformer session and prepare KV cache.
        Args:
            model_path: Path to the ONNX model file.
            num_layers: Number of transformer layers in the model.
            hidden_dim: Dimensionality of the model’s key/value vectors.
            page_size: Maximum tokens per page in the KV cache.
        """
        self.session = ort.InferenceSession(model_path)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.page_size = page_size
        # Initialize an empty KV cache for each layer.
        self.kv_cache = {
            layer: {"keys": [], "values": []}
            for layer in range(num_layers)
        }
        print("Model inputs:", [inp.name for inp in self.session.get_inputs()])

    def forward(self, input_ids):
        """
        Run one forward pass of the transformer.
        Args:
            input_ids: numpy array of shape (batch_size, sequence_length).
        Returns:
            logits: numpy array (batch_size, vocab_size).
            new_keys: dict mapping layer index -> new key tensor (batch_size, 1, hidden_dim).
            new_values: dict mapping layer index -> new value tensor (batch_size, 1, hidden_dim).
        """
        # Build the feed dictionary.
        # If past key values are available, they need to be added to the input.
        feed = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones(input_ids.shape, dtype=np.int64)
    }
        # Optionally add past keys/values if your model requires them.
        # For a real model, these inputs might be named like "past_key_0", "past_value_0", etc.
        for layer in range(self.num_layers):
            flat_keys, flat_values = self.get_flat_cache(layer)
            # If there is no cached value yet, use zeros (or omit the input if allowed).
            if flat_keys is None:
                feed[f"past_key_{layer}"] = np.zeros((input_ids.shape[0], 0, self.hidden_dim), dtype=np.float32)
                feed[f"past_value_{layer}"] = np.zeros((input_ids.shape[0], 0, self.hidden_dim), dtype=np.float32)
            else:
                feed[f"past_key_{layer}"] = flat_keys
                feed[f"past_value_{layer}"] = flat_values

        # Run the model; assume outputs[0] is logits and for each layer, outputs[1+2*layer] and outputs[2+2*layer]
        # are the new key and value respectively.
        outputs = self.session.run(None, feed)
        logits = outputs[0]
        new_keys = {}
        new_values = {}
        for layer in range(self.num_layers):
            new_keys[layer] = outputs[1 + 2 * layer]
            new_values[layer] = outputs[2 + 2 * layer]
        return logits, new_keys, new_values

    def update_cache(self, new_keys, new_values):
        """
        Update the KV cache for each transformer layer using a paged mechanism.
        A new page is created if the last page is full.
        """
        for layer in range(self.num_layers):
            # Update keys.
            keys_pages = self.kv_cache[layer]["keys"]
            if len(keys_pages) == 0 or keys_pages[-1].shape[1] >= self.page_size:
                self.kv_cache[layer]["keys"].append(new_keys[layer])
            else:
                self.kv_cache[layer]["keys"][-1] = np.concatenate(
                    [keys_pages[-1], new_keys[layer]], axis=1)
            # Update values similarly.
            values_pages = self.kv_cache[layer]["values"]
            if len(values_pages) == 0 or values_pages[-1].shape[1] >= self.page_size:
                self.kv_cache[layer]["values"].append(new_values[layer])
            else:
                self.kv_cache[layer]["values"][-1] = np.concatenate(
                    [values_pages[-1], new_values[layer]], axis=1)

    def get_flat_cache(self, layer):
        """
        Flatten the KV cache of the given layer across pages.
        Returns:
            concatenated keys and values (or None if empty).
        """
        if len(self.kv_cache[layer]["keys"]) == 0:
            return None, None
        keys = np.concatenate(self.kv_cache[layer]["keys"], axis=1)
        values = np.concatenate(self.kv_cache[layer]["values"], axis=1)
        return keys, values

def select_next_token(logits):
    """
    Simplified greedy selection: choose the token with the maximum logit.
    """
    return int(np.argmax(logits[0]))

def main():
    parser = argparse.ArgumentParser(
        description="ONNX Transformer Inference with KV Caching and Real Attention")
    parser.add_argument("--model", required=True, help="Path to the ONNX transformer model")
    parser.add_argument("--prompt", required=True, help="Comma-separated initial token IDs")
    parser.add_argument("--num_tokens", type=int, default=20, help="Number of tokens to generate")
    args = parser.parse_args()

    prompt_tokens = [int(x) for x in args.prompt.split(",")]
    # Start with the full prompt as the initial input.
    input_ids = np.array(prompt_tokens).reshape(1, -1)

    # Initialize the transformer (adjust num_layers, hidden_dim, and page_size as needed).
    transformer = ONNXTransformer(args.model, num_layers=4, hidden_dim=32, page_size=8)
    generated_tokens = list(prompt_tokens)
    print("Initial prompt:", generated_tokens)
    print("Beginning autoregressive generation...\n")

    # Autoregressive generation loop.
    for _ in range(args.num_tokens):
        logits, new_keys, new_values = transformer.forward(input_ids)
        next_token = select_next_token(logits)
        print("Generated token:", next_token)
        generated_tokens.append(next_token)
        # Update the KV cache with the new keys and values.
        transformer.update_cache(new_keys, new_values)
        # For the next forward pass, provide only the new token.
        input_ids = np.array([[next_token]])

    print("\nFinal generated sequence:", generated_tokens)

    # For demonstration, perform a real attention computation on layer 0’s cache.
    keys, values = transformer.get_flat_cache(0)
    if keys is not None:
        # Create a dummy query vector (in a real scenario, query comes from the transformer).
        query = np.random.randn(1, 1, transformer.hidden_dim).astype(np.float32)
        attn_output, attn_weights = scaled_dot_product_attention(query, keys, values)
        print("\nReal Attention Output (Layer 0):", attn_output)
        print("Attention Weights (Layer 0):", attn_weights)
    else:
        print("No cache available for attention computation.")

if __name__ == "__main__":
    main()
