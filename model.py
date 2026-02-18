"""
VoyagerGPT Model Architecture
A bigram GPT built from scratch for Star Trek text generation
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
BLOCK_SIZE = 256  # maximum context length for predictions
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

# Vocabulary
CHARS = ['\n', ' ', '!', '#', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '®', '�']

VOCAB_SIZE = len(CHARS)

# Create mappings
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}


def encode(s: str) -> list:
    """Encoder: take a string, output a list of integers"""
    return [STOI[c] for c in s]


def decode(l: list) -> str:
    """Decoder: take a list of integers, output a string"""
    return ''.join([ITOS[i] for i in l])


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits"""
    if temperature != 1.0:
        logits = logits / temperature
    return logits


class Head(nn.Module):
    """One head of self-attention with KV-cache support"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # Kept for compatibility with existing checkpoints; not used in forward.
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_kv=None):
        """
        Args:
            x:        (B, T, C) – new input tokens only (T=1 during incremental decoding)
            past_kv:  optional tuple (past_k, past_v) each (B, T_past, hs)
        Returns:
            out:        (B, T, hs)
            present_kv: tuple (k, v) each (B, T_past+T, hs) – full updated cache
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Prepend cached keys/values from previous steps
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)  # (B, T_past+T, hs)
            v = torch.cat([past_v, v], dim=1)  # (B, T_past+T, hs)

        present_kv = (k, v)  # will be returned as the new cache

        T_total = k.shape[1]          # T_past + T
        T_past  = T_total - T

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T_total)

        # Dynamic causal mask: query at absolute position (T_past+i) may only
        # attend to keys at absolute positions 0 … T_past+i.
        q_pos = torch.arange(T_past, T_past + T, device=x.device).unsqueeze(1)  # (T, 1)
        k_pos = torch.arange(T_total,             device=x.device).unsqueeze(0)  # (1, T_total)
        wei = wei.masked_fill((k_pos > q_pos).unsqueeze(0), float('-inf'))       # (B, T, T_total)

        wei = F.softmax(wei, dim=-1)  # (B, T, T_total)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, hs)
        return out, present_kv


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel with KV-cache support"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_kvs=None):
        """
        Args:
            x:         (B, T, C)
            past_kvs:  list of (past_k, past_v) per head, or None
        Returns:
            out:          (B, T, C)
            present_kvs:  list of (k, v) per head
        """
        if past_kvs is None:
            past_kvs = [None] * len(self.heads)

        head_outs, present_kvs = [], []
        for head, past_kv in zip(self.heads, past_kvs):
            h_out, present_kv = head(x, past_kv)
            head_outs.append(h_out)
            present_kvs.append(present_kv)

        out = torch.cat(head_outs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, present_kvs


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation, with KV-cache support"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kvs=None):
        """
        Args:
            x:         (B, T, C)
            past_kvs:  list of (past_k, past_v) per head, or None
        Returns:
            x:            (B, T, C)
            present_kvs:  list of (k, v) per head
        """
        sa_out, present_kvs = self.sa(self.ln1(x), past_kvs)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present_kvs


class GPTLanguageModel(nn.Module):
    """GPT Language Model"""

    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=N_EMBD, block_size=BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # ModuleList (vs Sequential) so we can pass per-layer KV caches explicitly.
        # State-dict keys are identical ("blocks.0.*", "blocks.1.*", …) so existing
        # checkpoints load without modification.
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None):
        """
        Args:
            idx:       (B, T) token indices.  T=1 during incremental decoding.
            targets:   (B, T) target indices for training loss, or None.
            past_kvs:  list[list[(k, v)]] – per-layer, per-head KV caches, or None.
        Returns:
            logits:       (B, T, vocab_size)  — or (B*T, vocab_size) when targets given
            loss:         scalar or None
            present_kvs:  updated list[list[(k, v)]] for all layers
        """
        device = idx.device
        B, T = idx.shape

        # Compute position offset from the cache so embeddings stay consistent
        # across prefill and incremental steps.
        T_past = 0
        if past_kvs is not None and past_kvs[0] is not None:
            # past_kvs[layer][head] = (k, v);  k.shape = (B, T_past, hs)
            T_past = past_kvs[0][0][0].shape[1]

        tok_emb = self.token_embedding_table(idx)                                     # (B, T, C)
        pos     = torch.arange(T_past, T_past + T, device=device)
        pos_emb = self.position_embedding_table(pos)                                  # (T, C)
        x = tok_emb + pos_emb                                                         # (B, T, C)

        if past_kvs is None:
            past_kvs = [None] * self.n_layer

        present_kvs = []
        for block, block_past_kvs in zip(self.blocks, past_kvs):
            x, block_present_kvs = block(x, block_past_kvs)
            present_kvs.append(block_present_kvs)

        x      = self.ln_f(x)                   # (B, T, C)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss, present_kvs

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressive token generation with KV-caching.

        Strategy
        --------
        • First call  – "prefill": run the full current context through the model
          and store the resulting KV cache.
        • Subsequent calls – "incremental": feed only the single new token; the
          cached K/V tensors supply the history, making each step O(T) instead of
          O(T²) in attention computation.
        • Cache overflow – when the cached sequence reaches block_size the next
          incremental position would be out-of-range for the position embedding
          table.  In that case we fall back to a full recompute on the last
          block_size tokens (identical to the original behaviour) and discard
          the resulting cache so the pattern can repeat.
        """
        past_kvs = None

        for _ in range(max_new_tokens):
            # Current length of the KV cache (0 if no cache yet)
            T_past = past_kvs[0][0][0].shape[1] if past_kvs is not None else 0

            if T_past >= self.block_size:
                # ── Cache full: full recompute, then discard cache ──────────
                idx_cond          = idx[:, -self.block_size:]
                logits, _, _      = self(idx_cond)
                past_kvs          = None              # reset; next step re-prefills

            elif past_kvs is None:
                # ── Prefill: first step, process the entire context ─────────
                idx_cond          = idx[:, -self.block_size:]
                logits, _, past_kvs = self(idx_cond)

            else:
                # ── Incremental: only the single newest token ───────────────
                idx_cond          = idx[:, -1:]
                logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs)

            # Sample the next token from the last position's logits
            logits   = logits[:, -1, :]                        # (B, C)
            logits   = apply_temperature(logits, temperature)
            probs    = F.softmax(logits, dim=-1)               # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx      = torch.cat((idx, idx_next), dim=1)       # (B, T+1)

        return idx

