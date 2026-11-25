"""Phase 2: decompose + verify the answers produced by run_baseline.py.

    python scripts/run_verify.py --verifier nli
    python scripts/run_verify.py --verifier judge --limit 20

Writes outputs/verified_<verifier>.jsonl (resumable) and prints the mean
claim precision, i.e. the "Claim P" of standard RAG in the results table.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from statistics import mean
from tqdm import tqdm
from factcheck_rag.generation.generator import Generator
from factcheck_rag.verification.verifier import build_verifier
from factcheck_rag.verification.factchecker import ClaimChecker
from factcheck_rag import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", choices=["nli", "judge"], default=config.VERIFIER)
    ap.add_argument("--input", default=os.path.join(config.RESULTS_DIR, "baseline.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(config.RESULTS_DIR, f"verified_{args.verifier}.jsonl")

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}

    gen = Generator()                                   # decomposer (+ judge)
    checker = ClaimChecker(gen, build_verifier(args.verifier, generator=gen))

    with open(out, "a", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"verify[{args.verifier}]"):
            if row["sample_id"] in done:
                continue
            row["check"] = checker.check(row["question"], row["answer"], row["passages"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    with open(out, encoding="utf-8") as f:
        checks = [json.loads(l)["check"] for l in f if l.strip()]
    precs = [c["claim_precision"] for c in checks if c["claim_precision"] is not None]
    all_claims = [cl for c in checks for cl in c["claims"]]
    print(f"\nQuestions          : {len(checks)}")
    print(f"Claims total       : {len(all_claims)}  "
          f"(mean {len(all_claims) / max(len(checks), 1):.1f} per answer)")
    print(f"Claim precision    : {mean(precs):.3f}  (macro, over {len(precs)} answers)")
    if all_claims:
        print(f"Claims supported   : {sum(c['supported'] for c in all_claims) / len(all_claims):.3f}  (micro)")
        print(f"Uncited claims     : {sum(not c['cited'] for c in all_claims)}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
