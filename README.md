# GPT-from-Scratch 🧠

A small GPT-style language model built completely from scratch in PyTorch, following the Transformer architecture.

This project covers the **end-to-end process of training and running a GPT-like model**:

- **Dataset preparation**:
  - Tokenization with GPT-2 BPE ([tiktoken](https://github.com/openai/tiktoken)).
  - Saved to `.bin` files using memory-mapped arrays for efficient streaming.
- **Model architecture**:
  - Token embeddings + positional embeddings.
  - Stacked Transformer blocks:
    - LayerNorm
    - Multi-Head Causal Self-Attention (with causal masking)
    - Feed-forward MLP (4× expansion → GELU → projection)
    - Residual connections
  - Final linear projection with tied weights.
- **Training**:
  - Optimizer: AdamW (with weight decay, gradient clipping).
  - Scheduler: Linear warmup + cosine decay.
  - Mixed precision training (AMP) with gradient accumulation.
  - Tracks training/validation loss and saves best model checkpoint.
- **Generation**:
  - Autoregressive decoding.
  - Sampling controls: `temperature` and `top-k`.

---

##  Quick Example

```python
import torch
from tiktoken import get_encoding

model.eval()
sentence = "Once upon a time there was a little girl."
context = torch.tensor(
    get_encoding("gpt2").encode_ordinary(sentence),
    dtype=torch.long,
    device="cuda" if torch.cuda.is_available() else "cpu"
).unsqueeze(0)

y = model.generate(context, max_new_tokens=200, temperature=0.9, top_k=50)
print(get_encoding("gpt2").decode(y.squeeze().tolist()))

**Sample output**:
Once upon a time there was a little girl. She was three years old and loved going to the beach. ...

