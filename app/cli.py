"""Command-line interface for the Course Notes AI Tutor."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from app.core import answer_question, complete_prompt, load_checkpoint, train_model

DEFAULT_NOTES = "data/course_notes.txt"
DEFAULT_CHECKPOINT = "checkpoints/latest.pt"


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="course-notes-ai-tutor",
        description="Train a small GPT from scratch on educational notes and use it for revision.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that the project is ready")

    train = sub.add_parser("train", help="train the GPT model from educational notes")
    train.add_argument("--notes", default=DEFAULT_NOTES, help="UTF-8 .txt notes file")
    train.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train.add_argument("--steps", type=int, default=500, help="additional training steps")
    train.add_argument("--vocab-size", type=int, default=320)
    train.add_argument("--block-size", type=int, default=96)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--n-embd", type=int, default=64)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--n-layers", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--resume", action="store_true", help="continue from the existing checkpoint")
    train.add_argument("--log-every", type=int, default=100)

    ask = sub.add_parser("ask", help="ask a revision question using your notes")
    ask.add_argument("question", nargs="?", help="question to answer")
    ask.add_argument("--notes", default=DEFAULT_NOTES)
    ask.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ask.add_argument("--max-new-tokens", type=int, default=90)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--top-k", type=int, default=20)
    ask.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    complete = sub.add_parser("complete", help="continue a text prompt")
    complete.add_argument("prompt", help="starting text")
    complete.add_argument("--notes", default=DEFAULT_NOTES, help="notes used for a grounded fallback")
    complete.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    complete.add_argument("--max-new-tokens", type=int, default=100)
    complete.add_argument("--temperature", type=float, default=0.8)
    complete.add_argument("--top-k", type=int, default=20)
    complete.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    chat = sub.add_parser("chat", help="start an interactive revision session")
    chat.add_argument("--notes", default=DEFAULT_NOTES)
    chat.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    chat.add_argument("--max-new-tokens", type=int, default=90)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-k", type=int, default=20)
    chat.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Course Notes AI Tutor is ready. Run 'python -m app.cli train' to train your GPT.")
        return

    device = _device(args.device)
    if args.command == "train":
        if args.steps < 1:
            raise SystemExit("--steps must be at least 1")
        train_model(args.notes, args.checkpoint, args.steps, args.vocab_size,
                    args.block_size, args.batch_size, args.n_embd, args.n_heads,
                    args.n_layers, args.learning_rate, device, args.resume, args.log_every)
        return

    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    if args.command == "ask":
        question = args.question or input("What would you like to revise? ").strip()
        if not question:
            raise SystemExit("Please enter a question.")
        print(answer_question(model, tokenizer, args.notes, question, args.max_new_tokens,
                              args.temperature, args.top_k, device))
    elif args.command == "complete":
        print(complete_prompt(model, tokenizer, args.notes, args.prompt, args.max_new_tokens,
                              args.temperature, args.top_k, device))
    elif args.command == "chat":
        print("Course Notes AI Tutor — type /quit to exit.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGood luck with your revision!")
                break
            if question.lower() in {"/quit", "/exit", "quit", "exit"}:
                print("Good luck with your revision!")
                break
            if question:
                print("\nTutor:")
                print(answer_question(model, tokenizer, args.notes, question,
                                      args.max_new_tokens, args.temperature, args.top_k, device))


if __name__ == "__main__":
    main()
