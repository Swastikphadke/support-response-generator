
"""Core functionality for the Support Response Generator."""

import re
from pathlib import Path


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
