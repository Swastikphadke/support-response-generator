"""Core functionality for the Support Response Generator."""

import re
from collections import Counter
from pathlib import Path
from typing import Sequence


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)

ORDER_PATTERN = re.compile(
    r"\b(?:ORD|ORDER)[-# ]?[A-Z0-9]{5,}\b",
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"
)


def anonymize_text(text: str) -> str:
    """Replace common personal and identifying information."""
    text = ORDER_PATTERN.sub("ordertoken", text)
    text = EMAIL_PATTERN.sub("emailtoken", text)
    text = URL_PATTERN.sub("urltoken", text)
    text = PHONE_PATTERN.sub("phonetoken", text)
    return text


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word and punctuation tokens."""
    return re.findall(
        r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]",
        text.lower()
    )


def prepare_corpus(text: str) -> list[list[str]]:
    """Anonymise, clean and tokenize each support response."""
    conversations = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        cleaned = anonymize_text(line)
        tokens = tokenize(cleaned)

        if tokens:
            conversations.append(tokens)

    return conversations


def load_corpus(path: str | Path) -> list[list[str]]:
    """Load and prepare a support-response corpus from a text file."""
    text = Path(path).read_text(encoding="utf-8")
    return prepare_corpus(text)


class NGramModel:
    """An n-gram language model with add-k smoothing."""

    def __init__(self, n: int = 2, k: float = 1.0):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.n = n
        self.k = k
        self.vocabulary: set[str] = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()

    def fit(self, corpus: list[list[str]]) -> None:
        """Learn n-gram and context counts from tokenized conversations."""
        self.vocabulary = {
            token for conversation in corpus for token in conversation
        }
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        for conversation in corpus:
            for i in range(len(conversation) - self.n + 1):
                ngram = tuple(conversation[i:i + self.n])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def probability(
        self,
        context: Sequence[str],
        word: str
    ) -> float:
        """Calculate P(word | context) using add-k smoothing."""
        if not self.vocabulary or word not in self.vocabulary:
            return 0.0

        context = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        ngram = context + (word,)

        count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]
        vocabulary_size = len(self.vocabulary)

        return (
            (count + self.k) /
            (context_count + self.k * vocabulary_size)
        )

    def suggest_next(
        self,
        context: Sequence[str],
        limit: int = 5
    ) -> list[tuple[str, float]]:
        """Return the most probable next words and their probabilities."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        suggestions = [
            (word, self.probability(context, word))
            for word in self.vocabulary
        ]

        return sorted(
            suggestions,
            key=lambda item: (-item[1], item[0])
        )[:limit]

def generate_continuation(
    model: NGramModel,
    prompt: str,
    max_tokens: int = 20,
    min_words: int = 3
) -> str:
    """Generate a response continuation from a partial prompt."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    if min_words < 0:
        raise ValueError("min_words cannot be negative")

    tokens = tokenize(prompt)
    if not tokens:
        return ""

    generated = []
    word_count = 0
    sentence_endings = {".", "!", "?"}

    for _ in range(max_tokens):
        context = (
            tuple(tokens[-(model.n - 1):])
            if model.n > 1 else ()
        )

        # Stop if the model has never seen this context.
        if model.context_counts[context] == 0:
            break

        suggestions = model.suggest_next(context, limit=len(model.vocabulary))

        if word_count < min_words:
            suggestions = [
                item for item in suggestions
                if item[0] not in sentence_endings
                and re.fullmatch(r"[a-z]+(?:'[a-z]+)?|[0-9]+", item[0])
            ]

        if not suggestions:
            break

        next_token = suggestions[0][0]
        generated.append(next_token)
        tokens.append(next_token)

        if next_token in sentence_endings:
            break

        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?|[0-9]+", next_token):
            word_count += 1

    continuation = " ".join(generated)
    continuation = re.sub(r"\s+([,.!?;:])", r"\1", continuation)

    return continuation