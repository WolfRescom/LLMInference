import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model with caching enabled.
model = GPT2LMHeadModel.from_pretrained("gpt2", use_cache=True)
model.eval()

# Initialize the tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create dummy input data.
input_text = "Hello, world"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]       # Shape: [batch_size, sequence_length]
attention_mask = inputs["attention_mask"]

# Create dummy past key/value tensors.
# GPT-2 has a number of layers given by model.config.n_layer.
num_layers = model.config.n_layer
batch_size = input_ids.shape[0]
num_heads = model.config.n_head
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads

# Instead of using an empty past (0 tokens), we use a dummy past with length 1.
dummy_past = []
for _ in range(num_layers):
    # Each layer's past is a tuple: (past_key, past_value)
    dummy_key = torch.zeros((batch_size, num_heads, 1, head_dim), dtype=torch.float32)
    dummy_value = torch.zeros((batch_size, num_heads, 1, head_dim), dtype=torch.float32)
    dummy_past.append((dummy_key, dummy_value))
# Pack the past as a tuple of tuples.
dummy_past = tuple(dummy_past)

# Prepare input names.
# Here we “flatten” the past tuple into one input (using a placeholder name like "past")
# In more advanced setups, you might export each past for each layer separately.
input_names = ["input_ids", "attention_mask", "past"]
output_names = ["logits", "present"]

# Define dynamic axes.
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence"},
    "attention_mask": {0: "batch_size", 1: "sequence"},
    "logits": {0: "batch_size", 1: "sequence"},
    "present": {0: "batch_size"}  # You may further specify dynamic axes for the cached outputs.
}

# Export the model.
torch.onnx.export(
    model,
    (input_ids, attention_mask, dummy_past),
    "onnx/gpt2_with_cache.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=11,  # Use an appropriate opset version.
)

print("Model exported to onnx/gpt2_with_cache.onnx")
