import torch

from app.core import (
    BytePairTokenizer,
    GPTConfig,
    GPTLanguageModel,
    clean_text,
    generate_text,
    load_checkpoint,
    prepare_corpus,
    save_checkpoint,
)


def test_clean_text_normalizes_whitespace():
    assert clean_text("  A\tB  \r\n\r\n\n C  ") == "A B\n\nC\n"


def test_sample_corpus_is_available():
    text = prepare_corpus("data/course_notes.txt")
    assert "MACHINE LEARNING" in text
    assert len(text) > 300


def test_bpe_round_trip_and_merges():
    text = "attention attention transformer transformer"
    tokenizer = BytePairTokenizer().fit(text, vocab_size=280)
    assert tokenizer.vocab_size > 256
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_shapes_and_causal_mask():
    torch.manual_seed(7)
    config = GPTConfig(vocab_size=270, block_size=8, n_embd=32, n_heads=4, n_layers=2, dropout=0.0)
    model = GPTLanguageModel(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    changed_future = tokens.clone()
    changed_future[:, 5:] = torch.randint(0, config.vocab_size, changed_future[:, 5:].shape)
    first_logits, _ = model(tokens)
    changed_logits, _ = model(changed_future)
    assert torch.allclose(first_logits[:, :5], changed_logits[:, :5], atol=1e-6)


def test_checkpoint_saves_and_loads_model_and_tokenizer(tmp_path):
    tokenizer = BytePairTokenizer().fit("neural networks learn patterns", vocab_size=265)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=32, n_heads=4, n_layers=1)
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tokenizer, optimizer, step=12, history=[{"step": 12}])
    loaded, loaded_tokenizer, payload = load_checkpoint(path)
    assert loaded.config.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("neural")) == "neural"
    assert payload["step"] == 12
    assert payload["optimizer_state"] is not None


def test_generation_returns_prompt_and_completion():
    tokenizer = BytePairTokenizer().fit("science data model learning science data model", vocab_size=270)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=16, n_embd=32, n_heads=4, n_layers=1)
    model = GPTLanguageModel(config)
    result = generate_text(model, tokenizer, "Science:", max_new_tokens=3)
    assert result.startswith("Science:")
    assert len(result) >= len("Science:")


def test_question_answer_uses_notes_and_returns_only_completion(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("A stack follows Last In, First Out. Its operations are push and pop.\n" * 8)
    tokenizer = BytePairTokenizer().fit(notes.read_text(), vocab_size=270)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=48, n_embd=32, n_heads=4, n_layers=1)
    model = GPTLanguageModel(config)
    from app.core import answer_question
    answer = answer_question(model, tokenizer, notes, "What is a stack?", max_new_tokens=2)
    assert "Study notes:" not in answer
    assert "Question:" not in answer
    assert "stack" in answer.lower() or "last in, first out" in answer.lower()


def test_completion_falls_back_to_relevant_notes_when_model_is_untrained(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("A stack follows Last In, First Out. Its operations are push and pop.\n" * 8)
    tokenizer = BytePairTokenizer().fit(notes.read_text(), vocab_size=270)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=48, n_embd=32, n_heads=4, n_layers=1)
    model = GPTLanguageModel(config)
    from app.core import complete_prompt
    completion = complete_prompt(model, tokenizer, notes, "A stack follows", max_new_tokens=2)
    assert completion.lower().startswith("a stack follows")
    assert "last in, first out" in completion.lower()


def test_training_can_resume_from_saved_checkpoint(tmp_path):
    from app.core import train_model
    notes = tmp_path / "notes.txt"
    notes.write_text(("A model learns patterns from educational examples. Validation checks performance.\n\n") * 20)
    checkpoint = tmp_path / "resume.pt"
    train_model(notes, checkpoint, steps=1, vocab_size=270, block_size=16,
                batch_size=2, n_embd=16, n_heads=2, n_layers=1,
                device="cpu", log_every=1)
    train_model(notes, checkpoint, steps=1, resume=True, device="cpu", log_every=1)
    _, _, payload = load_checkpoint(checkpoint)
    assert payload["step"] == 2
