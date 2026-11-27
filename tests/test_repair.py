"""GPU-free unit tests: python -m pytest tests  (or python tests/test_repair.py)"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factcheck_rag.repair import repairer as rp
from factcheck_rag.repair.repairer import (
    NO_ANSWER, Repairer, add_passages, parse_rewrite, parse_subquery, with_citations)
from factcheck_rag.verification.factchecker import ClaimChecker


def test_parse_subquery():
    assert parse_subquery('Query: "Apollo 11 landing date"\n', "x") == "Apollo 11 landing date"
    assert parse_subquery("\n  first moon walk\nextra", "x") == "first moon walk"
    assert parse_subquery("", "fallback claim") == "fallback claim"


def test_parse_rewrite():
    assert parse_rewrite("NONE", [6]) is None
    assert parse_rewrite("", [6]) is None
    assert parse_rewrite("None of the Beatles wrote it [6].", [6]) == (
        "None of the Beatles wrote it.", [6])
    assert parse_rewrite("Fixed claim: Armstrong landed in 1969 [7].", [6, 7]) == (
        "Armstrong landed in 1969.", [7])
    # citation of a passage that was not shown is ignored -> cite all shown
    assert parse_rewrite("Armstrong landed in 1969 [2].", [6, 7]) == (
        "Armstrong landed in 1969.", [6, 7])


def test_with_citations():
    assert with_citations("Armstrong walked first.", [6]) == "Armstrong walked first [6]."
    assert with_citations("Did he walk?", [1, 2]) == "Did he walk [1][2]?"
    assert with_citations("No tags.", []) == "No tags."


def test_add_passages_dedupes_and_numbers():
    passages = [{"title": "A", "text": "a"}]
    nums = add_passages(passages, [{"title": "B", "text": "b"}, {"title": "A", "text": "a"}])
    assert nums == [2, 1] and len(passages) == 2


# ---------------------------------------------------------------- fakes
class _FakeGenerator:
    """Sub-query: echoes the claim. Rewrite: 'unknown' -> NONE, 'stubborn' ->
    same claim (stays unsupported), anything else -> a corrected claim citing
    the first passage shown."""
    def __init__(self):
        self.calls = []

    def chat(self, system, user, max_new_tokens=0):
        self.calls.append((system, user))
        claim = re.search(r"Claim: (.*)", user).group(1)
        if system == rp._SUBQUERY_SYSTEM:
            return f"Query: {claim}"
        if "unknown" in claim:
            return "NONE"
        if "stubborn" in claim:
            return claim
        first = re.search(r"^\[(\d+)\]", user, flags=re.M).group(1)
        return f"Corrected fact is here [{first}]."


class _FakeRetriever:
    """Always returns one new passage per query plus passage 1 again."""
    def retrieve(self, query, top_k=5):
        return [{"title": "New", "text": f"about {query}"}, {"title": "T0", "text": "x0"}]


class _FakeVerifier:
    name = "fake"
    def verify(self, claim, evidence):
        ok = claim.startswith("Corrected")
        return {"supported": ok, "score": float(ok), "verifier": "fake"}


def _claim(i, text, sentence_idx, citations, supported):
    return {"id": f"c{i}", "text": text, "sentence_idx": sentence_idx,
            "citations": citations, "supported": supported}


def _run(answer, claims, max_rounds=3):
    passages = [{"title": f"T{i}", "text": f"x{i}"} for i in range(5)]
    gen = _FakeGenerator()
    rep = Repairer(gen, _FakeRetriever(), ClaimChecker(gen, _FakeVerifier()), max_rounds)
    return rep.repair("q", answer, passages, claims), gen, passages


# ---------------------------------------------------------------- loop
def test_supported_untouched_unsupported_rewritten():
    answer = "True fact one [1]. Wrong fact two [2]."
    claims = [_claim(0, "True fact one.", 0, [1], True),
              _claim(1, "Wrong fact two.", 1, [2], False)]
    res, _, passages = _run(answer, claims)
    assert res["answer"] == "True fact one [1]. Corrected fact is here [6]."
    assert len(res["passages"]) == 6 and len(passages) == 5     # input list not modified
    assert (res["rounds"], res["n_kept"], res["n_repaired"], res["n_dropped"]) == (1, 1, 1, 0)
    assert claims[1]["text"] == "Wrong fact two."                 # input claims not modified


def test_mixed_sentence_is_rebuilt_from_its_claims():
    answer = "Armstrong walked first and wrong date here [1]."
    claims = [_claim(0, "Armstrong walked first.", 0, [1], True),
              _claim(1, "Wrong date is here.", 0, [1], False)]
    res, _, _ = _run(answer, claims)
    assert res["answer"] == "Armstrong walked first [1]. Corrected fact is here [6]."


def test_none_drops_claim_and_empty_answer_falls_back():
    res, _, _ = _run("An unknown fact [1].", [_claim(0, "An unknown fact.", 0, [1], False)])
    assert res["answer"] == NO_ANSWER
    assert (res["rounds"], res["n_dropped"]) == (1, 1)


def test_stops_after_max_rounds_and_passes_tried_queries():
    res, gen, _ = _run("A stubborn fact [1].", [_claim(0, "A stubborn fact.", 0, [1], False)],
                       max_rounds=3)
    c = res["claims"][0]
    assert res["rounds"] == 3 and len(c["history"]) == 3 and c["status"] == "dropped"
    assert res["answer"] == NO_ANSWER
    sub_prompts = [u for s, u in gen.calls if s == rp._SUBQUERY_SYSTEM]
    assert "Already tried: (none)" in sub_prompts[0]
    assert "Already tried: A stubborn fact." in sub_prompts[1]


def test_nothing_to_repair_keeps_answer_verbatim():
    answer = "True fact one [1]. The passages do not say more."
    res, gen, _ = _run(answer, [_claim(0, "True fact one.", 0, [1], True)])
    assert res["answer"] == answer and res["rounds"] == 0 and gen.calls == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
