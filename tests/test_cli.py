from app.cli import main


def test_hello_command(capsys):
    main(["hello"])
    output = capsys.readouterr().out

    assert "ready" in output.lower()


def test_suggest_command(capsys):
    main([
        "suggest",
        "--prompt", "your order",
        "--length", "10"
    ])

    output = capsys.readouterr().out

    assert "Suggested continuation:" in output


def test_suggest_with_trigram(capsys):
    main([
        "suggest",
        "--prompt", "your order",
        "--n", "3",
        "--length", "10"
    ])

    output = capsys.readouterr().out

    assert "Suggested continuation:" in output


def test_invalid_length():
    import pytest

    with pytest.raises(SystemExit):
        main(["suggest", "--prompt", "your order", "--length", "0"])