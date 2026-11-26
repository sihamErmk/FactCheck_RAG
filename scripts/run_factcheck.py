"""Phase 3: run the FactCheck-RAG repair loop on the baseline drafts.

Reuses the answers of run_baseline.py as drafts, so FactCheck-RAG and
standard RAG start from exactly the same text and passages.

    python scripts/run_factcheck.py --verifier nli --limit 5
    python scripts/run_factcheck.py --verifier judge --max-rounds 3

Writes outputs/factcheck_<verifier>.jsonl (resumable). Each line keeps the
draft, the final answer, the claims, the sub-queries and rewrites of every
round, and one snapshot per round (answer + claim precision after round t).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from collections import Counter
from statistics import mean
from tqdm import tqdm
from factcheck_rag import config
from factcheck_rag.generation.generator import Generator
from factcheck_rag.retrieval.retriever import Retriever
from factcheck_rag.verification.verifier import build_verifier
from factcheck_rag.repair.loop import RepairLoop


def summarize(rows: list, max_rounds: int):
    def macro(values):
        values = [v for v in values if v is not None]
        return f"{mean(values):.3f}" if values else "n/a"

    print(f"\nQuestions               : {len(rows)}")
    print(f"Claim P before repair   : {macro(r['claim_precision_initial'] for r in rows)}")
    print(f"Claim P after repair    : {macro(r['claim_precision'] for r in rows)}")
    # snapshot of round t = result of a run with T = t (early stop -> last snapshot)
    for t in range(1, max_rounds + 1):
        at_t = [r["snapshots"][min(t, len(r["snapshots"]) - 1)]["claim_precision"] for r in rows]
        print(f"  Claim P with T = {t}     : {macro(at_t)}")
    with_claims = [r for r in rows if r.get("n_claims", 1) > 0]
    emptied = [r for r in rows if r.get("n_claims_initial", 0) > 0 and r.get("n_claims", 1) == 0]
    print(f"Answers fully supported : {sum(r['all_supported'] for r in with_claims)} / {len(with_claims)} "
          f"(answers with at least one claim)")
    print(f"Answers with no claim   : {len(rows) - len(with_claims)} "
          f"({len(emptied)} lost all their claims during repair)")
    status = Counter(w.get("status", "?") for r in rows for rd in r["rounds"] for w in rd["rewrites"])
    print(f"Rewrite attempts        : {sum(status.values())} -> rewritten {status['rewritten']}, "
          f"deleted {status['deleted']}, rejected (lost supported facts) {status['rejected']}, "
          f"unchanged {status['unchanged']}")
    print(f"New passages retrieved  : {sum(q.get('new_passages', 0) for r in rows for rd in r['rounds'] for q in rd['subqueries'])}")
    print(f"Mean repair rounds used : {mean(r['n_rounds'] for r in rows):.2f}")
    print(f"Mean repair time        : {mean(r['repair_s'] for r in rows):.1f} s / question")
    print(f"Mean total latency      : {mean(r['latency_s'] for r in rows):.1f} s / question "
          f"(baseline generation + verification + repair)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", choices=["nli", "judge"], default=config.VERIFIER)
    ap.add_argument("--input", default=os.path.join(config.RESULTS_DIR, "baseline.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=config.REPAIR_MAX_ROUNDS)
    args = ap.parse_args()
    out = args.out or os.path.join(config.RESULTS_DIR, f"factcheck_{args.verifier}.jsonl")

    with open(args.input, encoding="utf-8") as f:
        drafts = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        drafts = drafts[: args.limit]

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}

    gen = Generator()
    loop = RepairLoop(gen, Retriever(), build_verifier(args.verifier, generator=gen),
                      max_rounds=args.max_rounds)

    with open(out, "a", encoding="utf-8") as f:
        for row in tqdm(drafts, desc=f"factcheck[{args.verifier}]"):
            if row["sample_id"] in done:
                continue
            res = loop.repair(row["question"], row["answer"], row["passages"])
            res.update({
                "sample_id":   row["sample_id"],
                "question":    row["question"],
                "verifier":    args.verifier,
                "max_rounds":  args.max_rounds,
                "baseline_latency_s": row["latency_s"],
                "latency_s":   round(row["latency_s"] + res["repair_s"], 3),
                "qa_pairs":    row.get("qa_pairs"),
                "annotations": row.get("annotations"),
            })
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

    with open(out, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    summarize(rows, args.max_rounds)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
