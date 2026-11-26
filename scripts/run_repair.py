"""Phase 3: run the repair loop on the answers checked by run_verify.py.

    python scripts/run_repair.py --verifier nli
    python scripts/run_repair.py --verifier judge --limit 20
    python scripts/run_repair.py --verifier nli --max-rounds 1    # rounds ablation

Reads outputs/verified_<verifier>.jsonl and writes
outputs/repaired_<verifier>_T<rounds>.jsonl (resumable). Each row gets a
"repair" dict: the repaired answer, the extended passage list (citations
index into it), every claim with its status (kept / repaired / dropped) and
per-round history, and the rounds used.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from statistics import mean
from tqdm import tqdm
from factcheck_rag.generation.generator import Generator
from factcheck_rag.retrieval.retriever import Retriever
from factcheck_rag.verification.verifier import build_verifier
from factcheck_rag.verification.factchecker import ClaimChecker
from factcheck_rag.repair.repairer import Repairer
from factcheck_rag import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", choices=["nli", "judge"], default=config.VERIFIER)
    ap.add_argument("--max-rounds", type=int, default=config.MAX_ROUNDS)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    inp = args.input or os.path.join(
        config.RESULTS_DIR, f"verified_{args.verifier}.jsonl")
    out = args.out or os.path.join(
        config.RESULTS_DIR, f"repaired_{args.verifier}_T{args.max_rounds}.jsonl")

    with open(inp, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}

    gen = Generator()                       # sub-queries, rewrites (+ judge)
    checker = ClaimChecker(gen, build_verifier(args.verifier, generator=gen))
    repairer = Repairer(gen, Retriever(), checker, max_rounds=args.max_rounds)

    with open(out, "a", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"repair[{args.verifier}, T={args.max_rounds}]"):
            if row["sample_id"] in done:
                continue
            row["repair"] = repairer.repair(
                row["question"], row["answer"], row["passages"], row["check"]["claims"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    with open(out, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    reps = [r["repair"] for r in rows]
    n_kept = sum(r["n_kept"] for r in reps)
    n_repaired = sum(r["n_repaired"] for r in reps)
    n_dropped = sum(r["n_dropped"] for r in reps)
    touched = [r for r in reps if r["rounds"]]
    # end-to-end seconds per question: baseline RAG + decompose + verify + repair
    totals = [r["latency_s"] + r["check"]["decompose_s"] + r["check"]["verify_s"]
              + r["repair"]["repair_s"] for r in rows]
    print(f"\nQuestions          : {len(reps)}  ({len(touched)} needed repair)")
    print(f"Claims             : {n_kept + n_repaired + n_dropped}  kept {n_kept}, "
          f"repaired {n_repaired}, dropped {n_dropped}")
    if touched:
        print(f"Rounds used        : {mean(r['rounds'] for r in touched):.2f}  "
              f"(mean over answers that needed repair)")
        print(f"Repair time        : {mean(r['repair_s'] for r in touched):.1f} s  "
              f"(same answers)")
    if totals:
        print(f"End-to-end latency : {mean(totals):.1f} s per question")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
