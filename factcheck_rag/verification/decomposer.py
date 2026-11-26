"""Claim decomposer: draft answer -> list of atomic claims.

The answer is first split into sentences (keeping each sentence's citation
tags), then the LLM rewrites every sentence as one or more atomic,
self-contained claims. Each claim inherits the citations of its sentence,
so the verifier knows which passages to check it against.

Claim dict:
    {
      "id":           "c0",
      "text":         "Neil Armstrong was the first person to walk on the Moon.",
      "sentence_idx": 0,
      "sentence":     "The first person to walk on the Moon was Neil Armstrong [1].",
      "citations":    [1],        # 1-based passage numbers, as in the prompt
    }\n"""
import re
from factcheck_rag import config

_CITE_RE      = re.compile(r"\[(\d+)\]")
_SENT_SPLIT   = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_LEADING_CITE = re.compile(r"^((?:\s*(?:,|and)?\s*\[\d+\])+)[\s.,;]*")

# a period after these does not end a sentence ("St. Louis", "Dr. Who")
_ABBREVIATIONS = {
    "st", "mt", "ft", "mr", "mrs", "ms", "dr", "jr", "sr", "prof", "gen", "col",
    "lt", "sgt", "capt", "gov", "sen", "rep", "rev", "no", "vs", "etc", "inc",
    "ltd", "co", "corp", "dept", "est", "approx", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e",
}
# "Based on the information provided, X" -> "X": these lead-ins made the
# decomposer return NONE for sentences that do contain facts
_META_LEADIN = re.compile(
    r"^(?:based on|according to|from)\s+(?:the\s+)?(?:provided\s+|given\s+|available\s+)?"
    r"(?:information|search results|passages|results|texts|sources|documents|data)"
    r"(?:\s+provided|\s+given)?\s*,\s*", re.I)


def _ends_with_abbreviation(text: str) -> bool:
    if not text.endswith("."):
        return False
    token = text.split()[-1][:-1]                       # last word without its final "."
    if token.lower() in _ABBREVIATIONS:
        return True
    if re.fullmatch(r"[A-Z]", token):                   # initial: "J. K. Rowling"
        return True
    return bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token))   # "U.S", "U.K"


def drop_meta_leadin(sentence: str) -> str:
    out = _META_LEADIN.sub("", sentence, count=1)
    return out[:1].upper() + out[1:] if out != sentence else sentence

_SYSTEM = (
    "You split ONE sentence of an answer into atomic factual claims.\n"
    "Rules:\n"
    "- Each claim states exactly ONE fact.\n"
    "- Each claim must be understandable ALONE: never use pronouns (he, she, "
    "it, they, their, this, the song, the duo...). Replace them with the full "
    "name of the entity, found in the previous sentences or in the question.\n"
    "- Ignore phrases about sources, such as 'according to the search "
    "results' or 'the passages say', but DO extract the facts they introduce.\n"
    "- Do NOT add any information that is not in the sentence.\n"
    "- Only decompose the sentence marked 'Sentence:'; previous sentences "
    "are context only.\n"
    "- Output one claim per line, each line starting with '- '.\n"
    "- Approximate, hedged or concluding statements ('around 46', 'probably', "
    "'therefore X') are still claims: extract them, keeping the number or "
    "conclusion exactly as written.\n"
    "- If the sentence contains no factual claim (for example it only says "
    "that information is missing), output exactly: NONE\n\n"
    "Example\n"
    "Question: Who was the first person to walk on the moon?\n"
    "Previous sentences: Neil Armstrong was the first person to walk on the Moon.\n"
    "Sentence: According to the passages, he stepped onto the surface on "
    "July 20, 1969, during the Apollo 11 mission.\n"
    "Claims:\n"
    "- Neil Armstrong stepped onto the Moon's surface on July 20, 1969.\n"
    "- Neil Armstrong walked on the Moon during the Apollo 11 mission."
)


def extract_citations(text: str) -> list:
    """Return the sorted unique citation numbers found in text."""
    return sorted({int(m) for m in _CITE_RE.findall(text)})


def strip_citations(text: str) -> str:
    text = _CITE_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)   # "Armstrong ." -> "Armstrong."
    return re.sub(r"\s{2,}", " ", text).strip()


def split_sentences(answer: str) -> list:
    """Split an answer into sentences. Citation tags that ended up at the
    start of the next sentence (e.g. '... 1969. [2] Later ...') are moved
    back to the sentence they belong to."""
    sentences = []
    for chunk in answer.split("\n"):
        chunk = chunk.strip().lstrip("-*• ").strip()
        if not chunk:
            continue
        parts = []
        for part in _SENT_SPLIT.split(chunk):
            part = part.strip()
            if parts and _ends_with_abbreviation(parts[-1]):
                parts[-1] = parts[-1] + " " + part          # "St." + "Louis ..."
            else:
                parts.append(part)
        for sent in parts:
            m = _LEADING_CITE.match(sent)
            if m and sentences:
                sentences[-1] = sentences[-1] + " " + m.group(1).strip()
                sent = sent[m.end():].strip()
            if sent:
                sentences.append(sent)
    return sentences


def parse_claims(raw: str) -> list:
    """Parse the LLM output ('- claim' per line) into a list of strings."""
    if raw.strip().upper().startswith("NONE"):
        return []
    claims = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line.lower().startswith("claims:"):
            continue
        if len(line.split()) >= 3:          # drop fragments
            claims.append(strip_citations(line))
    return claims


class ClaimDecomposer:
    def __init__(self, generator):
        # reuse the already-loaded generator so the 7B model is loaded once
        self._gen = generator

    def decompose_sentence(self, question: str, sentence: str,
                           previous: list = ()) -> list:
        context = " ".join(strip_citations(p) for p in previous) or "(none)"
        user = (
            f"Question: {question}\n"
            f"Previous sentences: {context}\n"
            f"Sentence: {drop_meta_leadin(strip_citations(sentence))}\n"
            f"Claims:"
        )
        raw = self._gen.chat(_SYSTEM, user, max_new_tokens=config.DECOMP_MAX_TOKENS)
        return parse_claims(raw)

    def decompose(self, question: str, answer: str) -> list:
        claims = []
        sentences = split_sentences(answer)
        for s_idx, sentence in enumerate(sentences):
            cites = extract_citations(sentence)
            for text in self.decompose_sentence(question, sentence, sentences[:s_idx]):
                claims.append({
                    "id":           f"c{len(claims)}",
                    "text":         text,
                    "sentence_idx": s_idx,
                    "sentence":     sentence,
                    "citations":    cites,
                })
        return claims
