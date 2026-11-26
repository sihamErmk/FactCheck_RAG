"""Phase 3: repair the unsupported claims of a checked answer.

One round, for each claim the verifier rejected:

    sub-query (LLM) -> retrieve -> rewrite the claim (LLM) -> verify again

Supported claims are never touched. A claim is dropped when the rewrite finds
nothing (NONE) or when it is still unsupported after config.MAX_ROUNDS rounds.
New passages are appended, so a citation [n] keeps meaning passages[n-1].
"""
import re
import time
from factcheck_rag import config
from factcheck_rag.verification.decomposer import (
    extract_citations, split_sentences, strip_citations)

_SUBQUERY_SYSTEM = (
    "You write search queries for a Wikipedia search engine.\n"
    "Given a question and one claim from an answer to it, write ONE short "
    "query (at most 12 words) that would find a passage confirming or "
    "correcting the claim. Name the entities in full. Do not repeat a query "
    "listed under 'Already tried'.\n"
    "Output only the query."
)

_REWRITE_SYSTEM = (
    "You fix one claim of an answer using new evidence passages. Use only "
    "the passages, not your own knowledge.\n"
    "- If a passage states the claim, output the claim unchanged.\n"
    "- If the passages state a different fact about the same thing (another "
    "date, number or name), rewrite the claim to match them, changing as "
    "little as possible.\n"
    "- If the passages say nothing about the claim, output exactly: NONE\n"
    "End the claim with the number(s) of the passage(s) that state it, like "
    "[6] or [6][8].\n"
    "Output one sentence, or NONE, and nothing else."
)

# answer used when every claim was dropped
NO_ANSWER = "The search results do not contain enough information to answer the question."


def parse_subquery(raw: str, fallback: str) -> str:
    """First non-empty line, without a 'Query:' label or quotes."""
    for line in raw.splitlines():
        line = re.sub(r"^query:\s*", "", line.strip(), flags=re.I).strip().strip("\"'").strip()
        if line:
            return line
    return fallback


def parse_rewrite(raw: str, shown: list):
    """Return (claim text, citations), or None when the model found nothing
    (NONE, empty output or a fragment). Citations outside `shown` (the
    passage numbers in the prompt) are ignored; if none is left, the claim
    cites every passage shown."""
    line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
    line = re.sub(r"^(?:[-*•]|fixed claim:|claim:)\s*", "", line, flags=re.I).strip()
    if line.rstrip(".").upper() == "NONE":   # "None of ..." is a real claim
        return None
    text = strip_citations(line)
    if len(text.split()) < 3:                     # same fragment rule as parse_claims
        return None
    cites = [c for c in extract_citations(line) if c in shown]
    return text, cites or list(shown)


def with_citations(text: str, cites: list) -> str:
    """'Armstrong walked first.' + [6] -> 'Armstrong walked first [6].' (ALCE style)."""
    if not cites:
        return text
    text = text.rstrip()
    end = text[-1] if text[-1:] in ("!", "?") else "."
    tags = "".join(f"[{c}]" for c in cites)
    return f"{text.rstrip('.!?').rstrip()} {tags}{end}"


def add_passages(passages: list, new: list) -> list:
    """Append the passages of `new` not already in `passages` (in place) and
    return the 1-based number of every passage of `new`."""
    pos = {(p.get("title", ""), p["text"]): i + 1 for i, p in enumerate(passages)}
    numbers = []
    for p in new:
        key = (p.get("title", ""), p["text"])
        if key not in pos:
            passages.append(p)
            pos[key] = len(passages)
        if pos[key] not in numbers:
            numbers.append(pos[key])
    return numbers


def rebuild_answer(answer: str, claims: list) -> str:
    """Sentences whose claims were all supported in Phase 2 are kept word for
    word (sentences without claims too). Any other sentence is replaced by its
    surviving claims, each with its own citations."""
    out = []
    for i, sentence in enumerate(split_sentences(answer)):
        mine = [c for c in claims if c["sentence_idx"] == i]
        if all(c["status"] == "kept" for c in mine):
            out.append(sentence)
        else:
            out += [with_citations(c["text"], c["citations"])
                    for c in mine if c["status"] != "dropped"]
    return " ".join(out) or NO_ANSWER


class Repairer:
    def __init__(self, generator, retriever, checker, max_rounds: int = config.MAX_ROUNDS):
        self._gen = generator              # the loaded Qwen, shared with Phase 2
        self._retriever = retriever
        self._checker = checker            # ClaimChecker: same evidence rule and verifier as Phase 2
        self.max_rounds = max_rounds

    def _subquery(self, question: str, claim: dict) -> str:
        tried = "; ".join(h["query"] for h in claim["history"]) or "(none)"
        user = (f"Question: {question}\n"
                f"Claim: {claim['text']}\n"
                f"Already tried: {tried}\n"
                f"Query:")
        raw = self._gen.chat(_SUBQUERY_SYSTEM, user, max_new_tokens=config.SUBQUERY_MAX_TOKENS)
        return parse_subquery(raw, fallback=claim["text"])

    def _rewrite(self, question: str, claim: dict, passages: list, shown: list):
        block = "\n\n".join(
            f"[{n}] (Title: {passages[n - 1].get('title', '')}) {passages[n - 1]['text']}"
            for n in shown)
        user = (f"Question: {question}\n"
                f"Claim: {claim['text']}\n\n"
                f"Passages:\n{block}\n\n"
                f"Fixed claim:")
        raw = self._gen.chat(_REWRITE_SYSTEM, user, max_new_tokens=config.REWRITE_MAX_TOKENS)
        return parse_rewrite(raw, shown)

    def repair(self, question: str, answer: str, passages: list, claims: list) -> dict:
        """`claims` = the verified claims from Phase 2 (check["claims"]).
        Inputs are not modified."""
        t0 = time.perf_counter()
        passages = list(passages)
        claims = [dict(c, status="kept" if c["supported"] else "open", history=[])
                  for c in claims]

        rounds = 0
        for r in range(1, self.max_rounds + 1):
            todo = [c for c in claims if c["status"] == "open"]
            if not todo:
                break
            rounds = r
            for c in todo:
                query = self._subquery(question, c)
                shown = add_passages(passages, self._retriever.retrieve(query, top_k=config.TOP_K))
                step = {"round": r, "query": query, "passages": shown}
                c["history"].append(step)
                if not shown:                         # nothing retrieved: try again next round
                    continue
                fixed = self._rewrite(question, c, passages, shown)
                if fixed is None:
                    step["rewrite"] = None
                    c["status"] = "dropped"
                    continue
                c["text"], c["citations"] = fixed
                self._checker.verify_claims([c], passages)
                step.update(rewrite=c["text"], supported=c["supported"], score=c["score"])
                if c["supported"]:
                    c["status"] = "repaired"

        for c in claims:
            if c["status"] == "open":                 # still unsupported after the last round
                c["status"] = "dropped"

        count = lambda s: sum(c["status"] == s for c in claims)
        return {
            "answer":     rebuild_answer(answer, claims),
            "passages":   passages,
            "claims":     claims,
            "rounds":     rounds,
            "n_kept":     count("kept"),
            "n_repaired": count("repaired"),
            "n_dropped":  count("dropped"),
            "repair_s":   round(time.perf_counter() - t0, 3),
        }
