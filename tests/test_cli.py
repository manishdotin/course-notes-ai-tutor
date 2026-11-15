from app.cli import main


def test_hello_command(capsys):
    main(["hello"])
    assert "ready" in capsys.readouterr().out.lower()


def test_help_is_available():
    from app.cli import build_parser
    parser = build_parser()
    assert "train" in parser.format_help()
