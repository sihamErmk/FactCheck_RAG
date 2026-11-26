"""GPU-free tests of the Phase 3 repair loop, with fake LLM / retriever / verifier.

    python tests/test_repair_loop.py
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factcheck_rag.repair.loop import RepairLoop
from factcheck_rag.repair.rewriter import parse_rewrite
from factcheck_rag.repair.subquery import parse_query


class FakeLLM:
    """Answers the three prompts of the loop: decompose, sub-query, rewrite."""
    def __init__(self, rewrite_to):
        self.rewrite_to = rewrite_to          # function(sentence) -> raw LLM output
        self.calls = []

    def chat(self, system, user, max_new_tokens=0):
        if system.startswith("You split ONE sentence"):          # decomposer: split on " and "
            sentence = user.split("Sentence: ")[1].split("\n")[0]
            return "\n".join(f"- {part}" for part in sentence.split(" and "))
        if system.startswith("You write search queries"):
            self.calls.append("subquery")
            return "Query: total number of Virginia state parks"
        if system.startswith("You correct ONE sentence"):
            self.calls.append("rewrite")
            sentence = user.split("Sentence: ")[1].split("\n")[0]
            return self.rewrite_to(sentence)
        raise AssertionError("unexpected prompt")


class FakeRetriever:
    def retrieve(self, query, top_k=3):
        return [{"id": 99, "title": "List of Virginia state parks",
                 "text": "Virginia has 46 properties in its state park system."}]


class DigitVerifier:
    """Supported iff every number of the claim appears in the evidence."""
    name = "fake"
    def verify(self, claim, evidence):
        text = " ".join(p["text"] for p in evidence)
        ok = all(n in text for n in re.findall(r"\d+", claim)) and bool(evidence)
        return {"supported": ok, "score": float(ok), "verifier": self.name}


PASSAGES = [{"id": 1, "title": "Virginia", "text": "There are 44 official state parks."}]
DRAFT = "There are 44 state parks [1]. There are around 47 parks in total [1]."


def _loop(rewrite_to, max_rounds=3):
    llm = FakeLLM(rewrite_to)
    return RepairLoop(llm, FakeRetriever(), DigitVerifier(), max_rounds=max_rounds), llm


def test_unsupported_claim_is_repaired_and_others_untouched():
    loop, llm = _loop(lambda s: "There are 46 properties in total [2].")
    out = loop.repair("How many state parks in Virginia?", DRAFT, PASSAGES)
    assert out["claim_precision_initial"] == 0.5
    assert out["claim_precision"] == 1.0 and out["all_supported"]
    assert out["n_rounds"] == 1 and llm.calls == ["subquery", "rewrite"]
    # supported sentence kept verbatim, only the bad one rewritten
    assert out["answer"] == "There are 44 state parks [1]. There are 46 properties in total [2]."
    assert len(out["passages"]) == 2                          # new passage appended as [2]
    assert [s["claim_precision"] for s in out["snapshots"]] == [0.5, 1.0]


def test_sentence_deleted_when_rewriter_says_none():
    loop, _ = _loop(lambda s: "NONE")
    out = loop.repair("q", DRAFT, PASSAGES)
    assert out["answer"] == "There are 44 state parks [1]."
    assert out["claim_precision"] == 1.0 and out["n_rounds"] == 1


def test_stops_after_max_rounds():
    # changes the sentence each time but never fixes the number
    loop, llm = _loop(lambda s: s.replace("around", "about") if "around" in s
                      else s.replace("about", "around"))
    out = loop.repair("q", DRAFT, PASSAGES)
    assert out["n_rounds"] == 3 and not out["all_supported"]
    assert len(out["snapshots"]) == 4                         # round 0 + 3 rounds
    assert llm.calls.count("rewrite") == 3
    assert len(out["passages"]) == 2                          # same hit not added twice


def test_unchanged_sentence_is_not_retried():
    loop, llm = _loop(lambda s: s)                            # rewriter returns it as is
    out = loop.repair("q", DRAFT, PASSAGES)
    assert out["n_rounds"] == 1 and llm.calls.count("rewrite") == 1
    assert out["rounds"][0]["rewrites"][0]["status"] == "unchanged"
    assert out["answer"] == DRAFT


def test_rewrite_that_loses_a_supported_fact_is_rejected():
    # "44 state parks" is supported, "47 in total" is not; the rewrite drops the 44
    draft = "There are 44 state parks and 47 parks in total [1]."
    loop, _ = _loop(lambda s: "There are 45 parks [2].")
    out = loop.repair("q", draft, PASSAGES)
    w = out["rounds"][0]["rewrites"][0]
    assert w["status"] == "rejected" and out["answer"] == draft
    assert out["claim_precision"] == 0.5 and out["n_rounds"] == 1


def test_no_repair_when_all_supported():
    loop, llm = _loop(lambda s: s)
    out = loop.repair("q", "There are 44 state parks [1].", PASSAGES)
    assert out["n_rounds"] == 0 and llm.calls == [] and out["all_supported"]


def test_parsers():
    assert parse_query('Query: "Virginia state parks count"', "x") == "Virginia state parks count"
    assert parse_query("", "fallback") == "fallback"
    assert parse_rewrite("Corrected sentence: There are 46 [2].") == "There are 46 [2]."
    assert parse_rewrite("NONE") is None and parse_rewrite("") is None
    # seen on Colab: Qwen echoes the sentence and appends NONE
    assert parse_rewrite("Duo 2Cellos played [1][2][3]. NONE") == "Duo 2Cellos played [1][2][3]."
    assert parse_rewrite("Duo 2Cellos played [1].\nNONE") == "Duo 2Cellos played [1]."
    assert parse_rewrite("NONE.") is None


class RankedRetriever:
    """Returns already-known passages first, like the real index did on Colab."""
    def retrieve(self, query, top_k=3):
        ranked = [{"id": 1, "title": "Virginia", "text": "There are 44 official state parks."}]
        ranked += [{"id": 100 + i, "title": f"New {i}", "text": f"new passage {i}"} for i in range(10)]
        return ranked[:top_k]


def test_subquery_brings_new_passages_not_only_known_ones():
    loop = RepairLoop(FakeLLM(lambda s: s), RankedRetriever(), DigitVerifier(), max_rounds=1)
    out = loop.repair("q", DRAFT, PASSAGES)
    q = out["rounds"][0]["subqueries"][0]
    assert q["new_passages"] == 3                      # top_k NEW passages despite the known hit
    assert q["passages"] == [1, 2, 3, 4]               # known [1] ranked first is kept as evidence
    assert len(out["passages"]) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
