"""Targeted sub-query: one short search query per unsupported claim."""
import re
from factcheck_rag import config

_SYSTEM = (
    "You write search queries for fact-checking. Given a question and a "
    "claim, write ONE short search query (at most 12 words) that would find "
    "a Wikipedia passage confirming or refuting the claim. Keep the key "
    "entities, names, dates and numbers of the claim. Output only the query, "
    "on one line, without quotes."
)


def parse_query(raw: str, fallback: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(search\s+)?query\s*:\s*", "", line, flags=re.I)
        line = line.strip().strip('"\'`').strip()
        if line:
            return line
    return fallback


class SubQueryWriter:
    def __init__(self, generator):
        self._gen = generator

    def write(self, question: str, claim: str) -> str:
        user = f"Question: {question}\nClaim: {claim}\nQuery:"
        raw = self._gen.chat(_SYSTEM, user, max_new_tokens=config.SUBQUERY_MAX_TOKENS)
        return parse_query(raw, fallback=claim)
