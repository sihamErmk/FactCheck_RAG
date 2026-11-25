"""Phase 2 entry point: decompose a draft answer and verify every claim.

Evidence rule: a claim is checked against the passages its sentence cites.
A claim whose sentence cites nothing is checked against ALL retrieved
passages and marked `cited=False` (so it can be counted separately; the
repair loop in Phase 3 treats it like any other unsupported claim when it
fails).
"""
import time
from factcheck_rag.verification.decomposer import ClaimDecomposer


class ClaimChecker:
    def __init__(self, generator, verifier):
        self.decomposer = ClaimDecomposer(generator)
        self.verifier = verifier

    @staticmethod
    def evidence_for(claim: dict, passages: list) -> list:
        cited = [passages[c - 1] for c in claim["citations"] if 1 <= c <= len(passages)]
        return cited if cited else list(passages)

    def verify_claims(self, claims: list, passages: list) -> list:
        """Verify already-decomposed claims (Phase 3 re-verifies rewritten claims)."""
        for claim in claims:
            evidence = self.evidence_for(claim, passages)
            claim["cited"] = any(1 <= c <= len(passages) for c in claim["citations"])
            claim.update(self.verifier.verify(claim["text"], evidence))
        return claims

    def check(self, question: str, answer: str, passages: list) -> dict:
        t0 = time.perf_counter()
        claims = self.decomposer.decompose(question, answer)
        t1 = time.perf_counter()
        self.verify_claims(claims, passages)
        t2 = time.perf_counter()
        n = len(claims)
        n_sup = sum(c["supported"] for c in claims)
        return {
            "claims":          claims,
            "n_claims":        n,
            "n_supported":     n_sup,
            # no claims (e.g. "the passages do not say") -> nothing unsupported
            "claim_precision": round(n_sup / n, 4) if n else None,
            "decompose_s":     round(t1 - t0, 3),
            "verify_s":        round(t2 - t1, 3),
        }
