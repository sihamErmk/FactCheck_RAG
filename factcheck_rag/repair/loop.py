"""Phase 3: the FactCheck-RAG repair loop.

    draft answer
      -> decompose into claims + verify            (Phase 2, round 0)
      -> for each round t = 1..T, while some claim is unsupported:
           for each sentence holding unsupported claims:
             for each unsupported claim:
               write a targeted sub-query, retrieve new passages
             rewrite ONLY that sentence with the new evidence
             re-decompose + re-verify ONLY that sentence
      -> final answer (+ one snapshot per round)

Sentences without unsupported claims are never touched. New passages are
appended to the passage list, so citation numbers [n] stay valid: [1]..[k]
are the original passages, [k+1].. the ones found by sub-queries.

Because rounds are sequential, the snapshot after round t is exactly what a
run with T = t would return: one run with T = 3 also gives the T = 1 and
T = 2 results for the Phase 5 ablation.
"""
import time
from collections import OrderedDict

from factcheck_rag import config
from factcheck_rag.repair.rewriter import SentenceRewriter
from factcheck_rag.repair.subquery import SubQueryWriter
from factcheck_rag.verification.decomposer import extract_citations, split_sentences
from factcheck_rag.verification.factchecker import ClaimChecker


def _precision(claims: list):
    return round(sum(c["supported"] for c in claims) / len(claims), 4) if claims else None


def _key(p: dict):
    return p.get("id", (p.get("title"), p.get("text")))


class RepairLoop:
    def __init__(self, generator, retriever, verifier,
                 max_rounds: int = config.REPAIR_MAX_ROUNDS,
                 top_k: int = config.REPAIR_TOP_K):
        self.checker = ClaimChecker(generator, verifier)
        self.subqueries = SubQueryWriter(generator)
        self.rewriter = SentenceRewriter(generator)
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.top_k = top_k

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _add_passage(pool: list, passage: dict) -> int:
        """Append passage to the pool unless already there; return its 1-based number."""
        key = _key(passage)
        for i, p in enumerate(pool):
            if _key(p) == key:
                return i + 1
        pool.append(passage)
        return len(pool)

    def _retrieve_new(self, query: str, pool: list):
        """Retrieve until `top_k` passages NOT already in the pool are found.

        Already-known passages ranked above them are kept too (the sentence may
        simply cite the wrong one). Returns (passage numbers in rank order,
        how many are new)."""
        known = {_key(p) for p in pool}
        hits = self.retriever.retrieve(query, top_k=self.top_k + len(pool))
        nums, new = [], 0
        for h in hits:
            if new >= self.top_k:
                break
            if _key(h) not in known:
                new += 1
            nums.append(self._add_passage(pool, h))
        return nums, new

    def _verify_sentence(self, question, sentences, s_idx, passages) -> list:
        sentence = sentences[s_idx]
        previous = [s for s in sentences[:s_idx] if s]
        texts = self.checker.decomposer.decompose_sentence(question, sentence, previous)
        claims = [{"text": t, "sentence_idx": s_idx, "sentence": sentence,
                   "citations": extract_citations(sentence)} for t in texts]
        return self.checker.verify_claims(claims, passages)

    @staticmethod
    def _snapshot(round_no, sentences, claims, t0):
        return {
            "round":           round_no,
            "answer":          " ".join(s for s in sentences if s),
            "n_claims":        len(claims),
            "n_supported":     sum(c["supported"] for c in claims),
            "claim_precision": _precision(claims),
            "elapsed_s":       round(time.perf_counter() - t0, 3),
        }

    # ------------------------------------------------------------ main
    def repair(self, question: str, answer: str, passages: list) -> dict:
        t0 = time.perf_counter()
        pool = [dict(p) for p in passages]
        sentences = split_sentences(answer)

        initial = self.checker.check(question, answer, pool)
        claims = initial["claims"]
        snapshots = [self._snapshot(0, sentences, claims, t0)]
        rounds = []

        stuck = set()        # sentences the rewriter cannot improve: not retried
        for round_no in range(1, self.max_rounds + 1):
            unsupported = [c for c in claims
                           if not c["supported"] and c["sentence_idx"] not in stuck]
            if not unsupported:
                break
            by_sentence = OrderedDict()
            for c in unsupported:
                by_sentence.setdefault(c["sentence_idx"], []).append(c)

            log = {"round": round_no, "subqueries": [], "rewrites": []}
            for s_idx, bad in by_sentence.items():
                before = sentences[s_idx]
                # evidence = passages the sentence already cites + sub-query hits
                cited = [n for n in extract_citations(before) if 1 <= n <= len(pool)]
                found = []
                for c in bad:
                    query = self.subqueries.write(question, c["text"])
                    nums, new = self._retrieve_new(query, pool)
                    found += [n for n in nums if n not in found]
                    log["subqueries"].append({"claim": c["text"], "query": query,
                                              "passages": nums, "new_passages": new})
                keep = found[: config.REWRITE_MAX_EVIDENCE]
                keep += [n for n in cited if n not in keep][: config.REWRITE_MAX_EVIDENCE - len(keep)]
                evidence = {n: pool[n - 1] for n in keep}

                after, raw = self.rewriter.rewrite(question, before, [c["text"] for c in bad], evidence)
                entry = {"sentence_idx": s_idx, "before": before, "after": after,
                         "evidence": sorted(evidence)}
                log["rewrites"].append(entry)

                # 1) no change: retrying the same sentence would only repeat itself
                if after is not None and " ".join(after.split()) == " ".join(before.split()):
                    entry["status"] = "unchanged"
                    stuck.add(s_idx)
                    continue

                # 2) re-verify the new sentence; keep it only if it does not lose
                #    supported facts (Qwen sometimes changes a correct fact)
                old = [c for c in claims if c["sentence_idx"] == s_idx]
                sentences[s_idx] = after or ""
                new = self._verify_sentence(question, sentences, s_idx, pool) if after else []
                if sum(c["supported"] for c in new) < sum(c["supported"] for c in old):
                    entry["status"] = "rejected"
                    sentences[s_idx] = before
                    stuck.add(s_idx)
                    continue

                entry["status"] = "rewritten" if after else "deleted"
                claims = [c for c in claims if c["sentence_idx"] != s_idx] + new
                claims.sort(key=lambda c: c["sentence_idx"])

            rounds.append(log)
            snapshots.append(self._snapshot(round_no, sentences, claims, t0))

        for i, c in enumerate(claims):
            c["id"] = f"c{i}"
        return {
            "answer":                  snapshots[-1]["answer"],
            "draft_answer":            answer,
            "passages":                pool,
            "claims":                  claims,
            "claim_precision_initial": initial["claim_precision"],
            "claim_precision":         _precision(claims),
            "n_claims_initial":        len(initial["claims"]),
            "n_claims":                len(claims),
            "n_rounds":                len(rounds),
            "all_supported":           all(c["supported"] for c in claims),
            "rounds":                  rounds,
            "snapshots":               snapshots,
            "repair_s":                round(time.perf_counter() - t0, 3),
        }


class FactCheckRAG:
    """End-to-end system: retrieve -> generate draft -> verify -> repair."""

    def __init__(self, retriever, generator, verifier, **loop_kwargs):
        self.retriever = retriever
        self.generator = generator
        self.loop = RepairLoop(generator, retriever, verifier, **loop_kwargs)

    def run(self, question: str, top_k: int = config.TOP_K) -> dict:
        t0 = time.perf_counter()
        passages = self.retriever.retrieve(question, top_k=top_k)
        draft = self.generator.generate(question, passages)
        out = self.loop.repair(question, draft, passages)
        out.update({"question": question, "latency_s": round(time.perf_counter() - t0, 3)})
        return out
