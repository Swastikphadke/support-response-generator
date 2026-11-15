import pytest

from app.core import NGramModel


@pytest.fixture
def sample_corpus():
    return [
        ["a", "b", "a"],
        ["a", "b", "c"]
    ]


def test_bigram_counts(sample_corpus):
    model = NGramModel(n=2, k=1.0)
    model.fit(sample_corpus)

    assert model.ngram_counts[("a", "b")] == 2
    assert model.ngram_counts[("b", "a")] == 1
    assert model.ngram_counts[("b", "c")] == 1


def test_laplace_probability(sample_corpus):
    model = NGramModel(n=2, k=1.0)
    model.fit(sample_corpus)

    # P(b | a) = (2 + 1) / (2 + 3) = 0.6
    assert model.probability(["a"], "b") == pytest.approx(0.6)

    # P(a | a) = (0 + 1) / (2 + 3) = 0.2
    assert model.probability(["a"], "a") == pytest.approx(0.2)


def test_probabilities_sum_to_one(sample_corpus):
    model = NGramModel(n=2, k=1.0)
    model.fit(sample_corpus)

    total = sum(
        model.probability(["a"], word)
        for word in model.vocabulary
    )

    assert total == pytest.approx(1.0)


def test_trigram_model(sample_corpus):
    model = NGramModel(n=3, k=1.0)
    model.fit(sample_corpus)

    assert model.ngram_counts[("a", "b", "a")] == 1
    assert model.ngram_counts[("a", "b", "c")] == 1

    assert model.probability(
        ["a", "b"], "a"
    ) == pytest.approx(0.4)


def test_unigram_model(sample_corpus):
    model = NGramModel(n=1, k=1.0)
    model.fit(sample_corpus)

    assert model.ngram_counts[("a",)] == 3
    assert model.probability([], "a") == pytest.approx(4 / 9)


def test_unseen_context_uses_smoothing(sample_corpus):
    model = NGramModel(n=2, k=1.0)
    model.fit(sample_corpus)

    # An unseen context gives each vocabulary word equal probability.
    assert model.probability(
        ["unknown"], "a"
    ) == pytest.approx(1 / 3)


def test_next_word_suggestions(sample_corpus):
    model = NGramModel(n=2, k=1.0)
    model.fit(sample_corpus)

    suggestions = model.suggest_next(["a"], limit=2)

    assert len(suggestions) == 2
    assert suggestions[0][0] == "b"
    assert suggestions[0][1] == pytest.approx(0.6)


def test_invalid_model_settings():
    with pytest.raises(ValueError):
        NGramModel(n=0)

    with pytest.raises(ValueError):
        NGramModel(n=2, k=0)