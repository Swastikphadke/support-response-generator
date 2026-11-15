
from app.core import (
    anonymize_text,
    tokenize,
    prepare_corpus,
    load_corpus,
)


def test_milestone_1(tmp_path):
    sample = (
        "Your order ORD-839201 is confirmed. "
        "Contact us at alice@example.com or +1 234 567 8901. "
        "Visit https://example.com"
    )

    anonymized = anonymize_text(sample)

    assert "ORD-839201" not in anonymized
    assert "alice@example.com" not in anonymized
    assert "234 567 8901" not in anonymized
    assert "https://example.com" not in anonymized

    assert "ordertoken" in anonymized
    assert "emailtoken" in anonymized
    assert "phonetoken" in anonymized
    assert "urltoken" in anonymized

    tokens = tokenize("Hello, THANK you!")
    assert tokens == ["hello", ",", "thank", "you", "!"]

    conversations = prepare_corpus(
        "Your order is confirmed.\n\nThank you for contacting us."
    )

    assert len(conversations) == 2

    corpus_file = tmp_path / "support.txt"
    corpus_file.write_text(
        "Your order is confirmed.\nThank you for contacting us.",
        encoding="utf-8"
    )

    assert load_corpus(corpus_file) == conversations
