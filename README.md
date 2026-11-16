# Course Notes AI Tutor

A command-line revision tutor built around a **small decoder-only GPT trained from scratch in PyTorch**. It uses a byte-level Byte-Pair Encoding (BPE) tokenizer, learned token and position embeddings, hand-written causal self-attention, transformer blocks, next-token cross-entropy training, text generation, and reusable checkpoints.

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/evnr297kn2/badge.svg)](https://msrit.ask.thili.ai/b/evnr297kn2)

> This is an educational mini-model, not a pretrained ChatGPT-style service. Its answers are generated from patterns in the supplied notes and can be incomplete or incorrect. Check important facts against your course material.

## Features

- Train on the included sample notes or your own UTF-8 `.txt` file.
- Byte-level BPE tokenizer (raw bytes plus learned frequent-pair merges).
- Token embeddings and learned positional embeddings.
- Scaled dot-product multi-head self-attention with a causal mask.
- Transformer blocks with residual connections, layer normalization, and feed-forward networks.
- Training loop with forward pass, cross-entropy loss, backpropagation, AdamW, and validation loss.
- Text completion and question-style revision prompts, with a small keyword-based notes passage retriever.
- Save and reload the model, tokenizer, configuration, training step, and optimizer state in a checkpoint.
- Continue training from a saved checkpoint with `--resume`.
- Interactive command-line chat.

## 1. Setup

Python 3.10+ is recommended. PyTorch runs on CPU; a CUDA GPU is used automatically when available.

```bash
# Extract the project ZIP, then enter the project folder
cd course-notes-ai-tutor

python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

No API key, external model, or internet connection is needed after installing PyTorch.

## 2. Train the GPT

The included `data/course_notes.txt` is a starter educational corpus.

```bash
python -m app.cli train
```

Default training settings are deliberately small enough for a laptop:

- 500 training steps
- BPE vocabulary target: 320 tokens
- context length: 96 tokens
- embedding size: 64
- 4 attention heads and 2 transformer blocks
- batch size: 16

Training prints training and validation cross-entropy loss. The checkpoint is saved to `checkpoints/latest.pt`.

For a quick smoke run:

```bash
python -m app.cli train --steps 30 --checkpoint checkpoints/quick.pt
```

For your own notes:

```bash
python -m app.cli train --notes path/to/my_course_notes.txt --checkpoint checkpoints/my_course.pt
python -m app.cli ask "Explain supervised learning" --notes path/to/my_course_notes.txt --checkpoint checkpoints/my_course.pt
```

The notes file should be plain UTF-8 text with several paragraphs. Keep the text you use appropriate to your learning project and make sure you have permission to use it.

## 3. Ask questions or complete text

A sample checkpoint trained for 900 steps on the included notes is provided at `checkpoints/latest.pt`, so you can try the tutor immediately. To retrain it or use your own notes, run the training command above first.

```bash
# Ask a revision question; the app selects a relevant passage from the notes
python -m app.cli ask "What is overfitting?"

# Continue a prompt (uses GPT generation, with a notes-based fallback if the
# small model's completion is not readable)
python -m app.cli complete "A primary key is"

# Start an interactive session
python -m app.cli chat
```

You can select another notes file when asking questions:

```bash
python -m app.cli ask "Explain TCP and UDP" --notes path/to/networking_notes.txt
```

Generation controls:

```bash
python -m app.cli ask "What is a stack?" --temperature 0.7 --top-k 15 --max-new-tokens 80
```

Lower temperature makes sampling less random. Top-k restricts each next-token choice to the k highest-scoring tokens.

## 4. Checkpoints: save, load, and resume

- Every training log point saves `checkpoints/latest.pt` (or the path supplied with `--checkpoint`).
- The included `checkpoints/latest.pt` is a sample model trained for 900 steps; it can be loaded immediately.
- Each checkpoint contains model weights, model configuration, BPE merges, training step/history, and optimizer state.
- `ask`, `complete`, and `chat` load the saved checkpoint automatically.
- Continue a run without starting from random weights:

```bash
python -m app.cli train --resume --steps 500
```

Use the same notes corpus when resuming. To train a fresh model, omit `--resume` or choose a new checkpoint path.

## 5. Run tests

```bash
pytest -q
```

## Project structure

```text
course-notes-ai-tutor/
├── app/
│   ├── core.py          # BPE, embeddings, attention, GPT, training, generation, checkpoints
│   └── cli.py           # train / ask / complete / chat commands
├── data/
│   └── course_notes.txt # sample educational corpus
├── checkpoints/         # saved model checkpoints (created during training)
├── tests/               # automated tests
├── requirements.txt
└── README.md
```

## Milestones completed

- [x] Prepare and clean an educational text corpus.
- [x] Implement byte-level BPE and token IDs; use learned token and positional embeddings.
- [x] Build scaled dot-product causal self-attention and transformer blocks.
- [x] Train GPT with next-token cross-entropy, backpropagation, and AdamW; report validation loss.
- [x] Generate autoregressive text with temperature and top-k sampling.
- [x] Build a command-line tutor using `input()` and `print()`.
- [x] Save/load trained checkpoints and resume training from optimizer/model state.

## Model limitations

This model is intentionally small and is trained only on the notes you provide. It learns next-token patterns; it is not an instruction-tuned assistant and does not guarantee factual answers. The question command adds a relevant notes paragraph to the prompt, but generated explanations should still be verified. More training data, a longer run, and a larger model may improve fluency but also require more compute.
