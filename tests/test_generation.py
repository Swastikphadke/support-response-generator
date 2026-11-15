import pytest

from app.core import NGramModel, generate_continuation


@pytest.fixture
def trained_model():
    corpus = [
        ["your", "order", "has", "been", "confirmed", "."],
        ["your", "order", "has", "been", "shipped", "."],
        ["your", "payment", "has", "been", "confirmed", "."]
    ]

    model = NGramModel(n=2, k=1.0)
    model.fit(corpus)
    return model


def test_generates_continuation(trained_model):
    result = generate_continuation(
        trained_model,
        "your order",
        max_tokens=10
    )

    assert result
    assert isinstance(result, str)
    assert "has" in result


def test_respects_token_limit(trained_model):
    result = generate_continuation(
        trained_model,
        "your order",
        max_tokens=2
    )

    assert len(result.split()) <= 2


def test_stops_at_sentence_ending(trained_model):
    result = generate_continuation(
        trained_model,
        "your order",
        max_tokens=10
    )

    assert result.endswith(".")


def test_empty_prompt_returns_empty_string(trained_model):
    assert generate_continuation(trained_model, "") == ""


def test_invalid_token_limit(trained_model):
    with pytest.raises(ValueError):
        generate_continuation(trained_model, "your order", max_tokens=0)


def test_unseen_context_returns_empty_string(trained_model):
    result = generate_continuation(
        trained_model,
        "quantum spaceship",
        max_tokens=10
    )

    assert result == ""