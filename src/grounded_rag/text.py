"""Tokenization shared by retrieval, the relevance gate and the grounding gate.

Deliberately simple and deterministic: lowercase, Unicode-normalized word and number
tokens (decimals kept, so "99.9" survives), English stopwords removed, and plurals
folded ("refunds" -> "refund"). Every component uses the same function, so a word
counts the same way when it is retrieved, checked for coverage and checked for support.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# A plain word list reads better than one word per line.
STOPWORDS = frozenset(
    """a an and any are as at be been but by can could did do does for from had has have
    how i if in into is it its me my no not of on or our so than that the their them
    then there these they this those to too us was we were what when where which while
    who whom why will with would you your""".split()  # noqa: SIM905
)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def _fold_plural(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss") and not word[0].isdigit():
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning: no stopwords, plurals folded."""
    return [_fold_plural(token) for token in tokens(text) if token not in STOPWORDS]


def coverage(question: str, text: str) -> float:
    """Share of the question's content words that appear in ``text``."""
    wanted = set(content_tokens(question))
    if not wanted:
        return 0.0
    return len(wanted & set(content_tokens(text))) / len(wanted)
