"""Rewrite one sentence so that its unsupported claims are fixed or removed.

Only the sentence that contains the unsupported claim(s) is rewritten; every
other sentence of the answer is kept verbatim. The rewriter sees the new
evidence (with the passage numbers used in the answer), so it can cite it.
"""
import re
from factcheck_rag import config

_SYSTEM = (
    "You correct ONE sentence of an answer using evidence passages.\n"
    "Rules:\n"
    "- Fix or delete ONLY the facts listed as unsupported. Keep every other "
    "fact of the sentence unchanged.\n"
    "- Every fact you keep or add must be stated in the evidence. Never use "
    "your own knowledge.\n"
    "- If the evidence gives the correct fact, replace the unsupported fact "
    "with it. If the evidence says nothing about it, delete that fact.\n"
    "- End the sentence with the numbers of the passages that support it, "
    "like [2][6]. Use only the passage numbers shown in the evidence.\n"
    "- If nothing is left after the correction, output exactly: NONE\n"
    "- Output only the corrected sentence."
)


def _passage_line(n: int, p: dict) -> str:
    return f"[{n}] (Title: {p.get('title', '')}) {p['text']}"


_NONE_RE = re.compile(r"(^|\s)NONE\.?(?=\s|$)")


def parse_rewrite(raw: str):
    """Return the corrected sentence, or None if the sentence must be deleted.

    Qwen sometimes echoes the sentence and appends 'NONE' (seen on Colab):
    any standalone NONE token is dropped, and the sentence is deleted only
    if nothing else is left."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    text = " ".join(lines)
    text = re.sub(r"^(corrected\s+sentence|sentence|answer)\s*:\s*", "", text, flags=re.I)
    text = _NONE_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip().strip('"').strip()
    return text or None


class SentenceRewriter:
    def __init__(self, generator):
        self._gen = generator

    def rewrite(self, question: str, sentence: str, unsupported: list,
                evidence: dict):
        """evidence: {passage_number (1-based, global): passage}."""
        block = "\n\n".join(_passage_line(n, p) for n, p in sorted(evidence.items()))
        bad = "\n".join(f"- {c}" for c in unsupported)
        user = (
            f"Question: {question}\n\n"
            f"Evidence:\n{block}\n\n"
            f"Sentence: {sentence}\n\n"
            f"Unsupported facts:\n{bad}\n\n"
            f"Corrected sentence:"
        )
        raw = self._gen.chat(_SYSTEM, user, max_new_tokens=config.REWRITE_MAX_TOKENS)
        return parse_rewrite(raw), raw
