"""GPU-free unit tests: python -m pytest tests  (or python tests/test_text_parsing.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factcheck_rag.verification.decomposer import (
    ClaimDecomposer, drop_meta_leadin, extract_citations, parse_claims, split_sentences,
    strip_citations)
from factcheck_rag.verification.factchecker import ClaimChecker
from factcheck_rag.verification.verifier import parse_judgement


def test_split_keeps_trailing_citations_with_their_sentence():
    a = ("The provided passages do not mention it. Therefore, I cannot answer. "
         "[1], [2], [3]")
    s = split_sentences(a)
    assert len(s) == 2 and extract_citations(s[1]) == [1, 2, 3]


def test_split_and_strip():
    a = "Armstrong walked first [1]. He landed on July 20, 1969. [2][3] Done."
    s = split_sentences(a)
    assert s[0] == "Armstrong walked first [1]."
    assert extract_citations(s[1]) == [2, 3]
    assert strip_citations(s[0]) == "Armstrong walked first."


def test_split_ignores_abbreviations():
    # seen on Colab: "St." was cut, giving "Louis in 1995, becoming the St."
    a = ("The Rams moved to St. Louis in 1995, becoming the St. Louis Rams [1]. "
         "J. K. Rowling lived in the U.S. in Jan. 1990 [2].")
    assert split_sentences(a) == [
        "The Rams moved to St. Louis in 1995, becoming the St. Louis Rams [1].",
        "J. K. Rowling lived in the U.S. in Jan. 1990 [2]."]


def test_drop_meta_leadin():
    assert drop_meta_leadin("Based on the information provided, the INC ran every state.") ==         "The INC ran every state."
    assert drop_meta_leadin("According to the search results, Ronaldo leads.") == "Ronaldo leads."
    assert drop_meta_leadin("From 1990, he played.") == "From 1990, he played."


def test_parse_claims():
    assert parse_claims("NONE") == []
    assert parse_claims("- Neil Armstrong walked on the Moon.\n- ok") == [
        "Neil Armstrong walked on the Moon."]
    assert parse_claims("1. A b c.\n2) D e f.") == ["A b c.", "D e f."]


def test_parse_judgement():
    assert parse_judgement("Quote: 44 official state parks\nVerdict: SUPPORTED") == (
        True, "44 official state parks")
    assert parse_judgement("Quote: NONE\nVerdict: UNSUPPORTED")[0] is False
    assert parse_judgement("Quote: x\nVerdict: NOT SUPPORTED")[0] is False
    assert parse_judgement("")[0] is False


class _EchoGenerator:
    """Fake LLM: turns each sentence into one claim, NONE for 'no info'."""
    def chat(self, system, user, max_new_tokens=0):
        sentence = user.split("Sentence: ")[1].split("\n")[0]
        return "NONE" if "do not" in sentence else f"- {sentence}"


class _FakeVerifier:
    name = "fake"
    def verify(self, claim, evidence):          # supported iff evidence was cited
        return {"supported": len(evidence) < 5, "score": 1.0, "verifier": "fake"}


def test_claim_checker_end_to_end():
    passages = [{"title": f"T{i}", "text": "x"} for i in range(5)]
    answer = "Fact one is here [1]. Passages do not say more. Uncited fact is here."
    res = ClaimChecker(_EchoGenerator(), _FakeVerifier()).check("q", answer, passages)
    assert res["n_claims"] == 2
    assert [c["cited"] for c in res["claims"]] == [True, False]
    assert res["claim_precision"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
