"""A small decoder-only GPT trained from scratch on educational text."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Milestone 1: clean and prepare an educational text corpus
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prepare_corpus(notes_path: str | Path) -> str:
    path = Path(notes_path)
    if not path.exists():
        raise FileNotFoundError(f"Notes file not found: {path}")
    text = clean_text(path.read_text(encoding="utf-8"))
    if len(text.encode("utf-8")) < 300:
        raise ValueError("The notes file is too short. Add at least a few paragraphs of educational text.")
    return text


# ---------------------------------------------------------------------------
# Milestone 2: byte-level BPE tokenizer and token embeddings
# ---------------------------------------------------------------------------
class BytePairTokenizer:
    """A compact byte-level BPE tokenizer. IDs 0..255 represent raw bytes."""

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def fit(self, text: str, vocab_size: int = 320, min_frequency: int = 2) -> "BytePairTokenizer":
        tokens = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        target = max(256, int(vocab_size))
        while 256 + len(merges) < target and len(tokens) > 1:
            counts = Counter(zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            new_id = 256 + len(merges)
            merged: list[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            if len(merged) == len(tokens):
                break
            merges.append((int(pair[0]), int(pair[1])))
            tokens = merged
        self.merges = merges
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}
        return self

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            new_id = 256 + rank
            merged: list[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        expanded: list[int] = []

        def expand(token_id: int) -> None:
            if token_id < 256:
                expanded.append(token_id)
            else:
                merge_index = token_id - 256
                if not 0 <= merge_index < len(self.merges):
                    raise ValueError(f"Unknown token id: {token_id}")
                left, right = self.merges[merge_index]
                expand(left)
                expand(right)

        for token_id in token_ids:
            expand(int(token_id))
        return bytes(expanded).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte-level-bpe", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, data: dict) -> "BytePairTokenizer":
        if data.get("type") != "byte-level-bpe":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(map(int, pair)) for pair in data["merges"]])


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_heads < 1 or self.n_embd % self.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
        if self.block_size < 2 or self.n_layers < 1 or self.vocab_size < 256:
            raise ValueError("Invalid GPT configuration")


# ---------------------------------------------------------------------------
# Milestone 3: causal self-attention and transformer blocks
# ---------------------------------------------------------------------------
class CausalSelfAttentionHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        scores = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.causal_mask[:time_steps, :time_steps] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ value


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttentionHead(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(torch.cat([head(x) for head in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadCausalAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.n_embd)
        self.feedforward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time_steps = token_ids.shape
        if time_steps > self.config.block_size:
            raise ValueError("Input is longer than the model context window")
        positions = torch.arange(time_steps, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        logits = self.language_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(batch * time_steps, -1), targets.reshape(batch * time_steps))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 20) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


# ---------------------------------------------------------------------------
# Milestone 4: training and reusable checkpoints
# ---------------------------------------------------------------------------
def _batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if len(data) <= block_size + 1:
        raise ValueError("Not enough tokens to create a training batch; add more notes.")
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def _estimate_loss(model: GPTLanguageModel, train_data: torch.Tensor, val_data: torch.Tensor,
                   batch_size: int, block_size: int, device: str, batches: int = 5) -> tuple[float, float]:
    model.eval()
    values = []
    with torch.no_grad():
        for source in (train_data, val_data):
            losses = []
            for _ in range(batches):
                x, y = _batch(source, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(float(loss.item()))
            values.append(sum(losses) / len(losses))
    model.train()
    return values[0], values[1]


def save_checkpoint(path: str | Path, model: GPTLanguageModel, tokenizer: BytePairTokenizer,
                    optimizer: torch.optim.Optimizer | None = None, step: int = 0,
                    history: list[dict] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "tokenizer": tokenizer.to_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "history": history or [],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run the train command first.")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        payload = torch.load(path, map_location=device)
    config = GPTConfig(**payload["config"])
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer, payload


def train_model(notes_path: str | Path, checkpoint_path: str | Path = "checkpoints/latest.pt",
                steps: int = 500, vocab_size: int = 320, block_size: int = 96,
                batch_size: int = 16, n_embd: int = 64, n_heads: int = 4,
                n_layers: int = 2, learning_rate: float = 3e-4,
                device: str = "cpu", resume: bool = False, log_every: int = 100):
    # Small CPU thread pools avoid large thread-launch overhead for this compact model.
    torch.set_num_threads(min(torch.get_num_threads(), 1))
    torch.manual_seed(1337)
    text = prepare_corpus(notes_path)
    encoded_text = None
    start_step = 0
    history: list[dict] = []
    optimizer = None
    checkpoint_path = Path(checkpoint_path)

    if resume and checkpoint_path.exists():
        model, tokenizer, payload = load_checkpoint(checkpoint_path, device)
        if payload.get("optimizer_state") is None:
            raise ValueError("This checkpoint has no optimizer state and cannot be resumed.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("step", 0))
        history = payload.get("history", [])
        print(f"Resuming from checkpoint step {start_step} (BPE vocab {tokenizer.vocab_size}).")
    else:
        split_at = int(0.9 * len(text))
        tokenizer = BytePairTokenizer().fit(text[:split_at], vocab_size=vocab_size)
        config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size,
                           n_embd=n_embd, n_heads=n_heads, n_layers=n_layers)
        model = GPTLanguageModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print(f"Prepared {len(text):,} characters; BPE vocabulary: {tokenizer.vocab_size} tokens.")

    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = max(block_size + 2, int(0.9 * len(all_tokens)))
    split_at = min(split_at, len(all_tokens) - block_size - 2)
    train_data, val_data = all_tokens[:split_at], all_tokens[split_at:]
    if len(val_data) <= block_size + 1:
        val_data = train_data[-(block_size + 2):]
    if len(train_data) <= block_size + 1 or len(val_data) <= block_size + 1:
        raise ValueError("The corpus is too small for this context size. Add more text or reduce --block-size.")

    model.train()
    total_steps = start_step + steps
    for step in range(start_step + 1, total_steps + 1):
        x, y = _batch(train_data, batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % max(1, log_every) == 0 or step == total_steps:
            train_loss, val_loss = _estimate_loss(model, train_data, val_data,
                                                  batch_size, model.config.block_size, device)
            record = {"step": step, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)}
            history.append(record)
            print(f"step {step:>5} | train loss {train_loss:.4f} | validation loss {val_loss:.4f}")
            save_checkpoint(checkpoint_path, model, tokenizer, optimizer, step, history)

    print(f"\nSaved trained model and BPE tokenizer to: {checkpoint_path}")
    return model, tokenizer, history


# ---------------------------------------------------------------------------
# Milestone 5: generation and lightweight notes retrieval for grounded prompts
# ---------------------------------------------------------------------------
def _content_words(text: str) -> set[str]:
    return set(_content_counts(text))


def _content_counts(text: str) -> Counter:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "why", "how",
                  "does", "do", "of", "to", "in", "on", "for", "and", "or", "it", "this",
                  "that", "with", "from", "by", "be", "can", "i", "we", "you", "your"}
    return Counter(word for word in re.findall(r"[a-z0-9]+", text.lower())
                   if len(word) > 1 and word not in stop_words)


def retrieve_context(notes_text: str, query: str, max_chars: int = 650) -> str:
    """Return the most relevant notes paragraph using simple keyword overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text(notes_text)) if p.strip()]
    query_words = _content_words(query)
    if not paragraphs:
        return ""

    def score(paragraph: str) -> tuple[int, int]:
        counts = _content_counts(paragraph)
        overlap = sum(counts[word] for word in query_words)
        return (overlap, -abs(len(paragraph) - max_chars // 2))

    ranked = sorted(enumerate(paragraphs), key=lambda item: (score(item[1]), -item[0]), reverse=True)
    chosen = ranked[0][1][:max_chars]
    if not query_words or score(ranked[0][1])[0] == 0:
        chosen = paragraphs[0][:max_chars]
    return chosen


def _extractive_answer(notes_text: str, question: str) -> str:
    """Fallback answer that quotes the most relevant sentence(s) from the notes."""
    context = retrieve_context(notes_text, question, max_chars=900)
    # The sample corpus includes explicit revision pairs; use their answer directly.
    if "Answer:" in context:
        answer = context.split("Answer:", 1)[1].strip()
        if answer:
            return answer
    terms = _content_words(question)
    sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (len(terms & _content_words(item[1])), -item[0]),
        reverse=True,
    )
    selected = [sentence.strip() for _, sentence in ranked[:2] if sentence.strip()]
    selected.sort(key=lambda sentence: context.find(sentence))
    return " ".join(selected) if selected else context.strip()


def generate_text(model: GPTLanguageModel, tokenizer: BytePairTokenizer, prompt: str,
                  max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 20,
                  device: str = "cpu") -> str:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = tokenizer.encode(" ")
    context = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k)
    return tokenizer.decode(output[0].tolist())


def answer_question(model: GPTLanguageModel, tokenizer: BytePairTokenizer, notes_path: str | Path,
                    question: str, max_new_tokens: int = 90, temperature: float = 0.7,
                    top_k: int = 20, device: str = "cpu") -> str:
    notes = prepare_corpus(notes_path)
    context = retrieve_context(notes, question, max_chars=500)
    # Keep the question and answer marker in the context window, even for long notes.
    suffix_ids = tokenizer.encode(f"\nQuestion: {question.strip()}\nAnswer:")
    available = max(0, model.config.block_size - len(suffix_ids))
    context_ids = tokenizer.encode("Study notes:\n" + context)[-available:] if available else []
    prompt_ids = (context_ids + suffix_ids)[-model.config.block_size:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_k=top_k)
    generated = tokenizer.decode(output_ids[0, len(prompt_ids):].tolist()).strip()

    # A compact from-scratch model may still be learning fluent answers. Use its
    # completion when it is made mostly of words found in the notes; otherwise
    # return a short, clearly labelled answer extracted from the relevant notes.
    generated_words = re.findall(r"[a-zA-Z]{2,}", generated.lower())
    known_words = _content_words(notes)
    grounded_ratio = (sum(word in known_words for word in generated_words) / len(generated_words)
                      if generated_words else 0.0)
    if len(generated_words) >= 5 and grounded_ratio >= 0.65 and "�" not in generated:
        return generated
    return "From your notes: " + _extractive_answer(notes, question)


def complete_prompt(model: GPTLanguageModel, tokenizer: BytePairTokenizer, notes_path: str | Path,
                    prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                    top_k: int = 20, device: str = "cpu") -> str:
    """Generate a continuation, falling back to a matching notes passage if needed."""
    generated = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device)
    continuation = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
    notes = prepare_corpus(notes_path)
    words = re.findall(r"[a-zA-Z]{2,}", continuation.lower())
    known = _content_words(notes)
    ratio = sum(word in known for word in words) / len(words) if words else 0.0
    if len(words) >= 5 and ratio >= 0.65 and "�" not in continuation:
        return generated

    note_text = _extractive_answer(notes, prompt)
    if note_text and note_text.lower().startswith(prompt.strip().lower()):
        first_sentence = re.split(r"(?<=[.!?])\s+", note_text, maxsplit=1)[0]
        return first_sentence
    if _content_words(prompt) & _content_words(note_text):
        return "From your notes: " + note_text
    return generated
