"""Run single-pass RAG on the ASQA subset and save one JSON line per question.

Resumable: questions already in the output file are skipped, so a Colab
disconnect does not lose work.

    python scripts/run_baseline.py                # 200 questions
    python scripts/run_baseline.py --limit 5      # quick test
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from datasets import load_dataset
from tqdm import tqdm
from factcheck_rag.pipeline import RAGPipeline
from factcheck_rag import config


def load_asqa(n: int):
    ds = load_dataset(config.ASQA_DATASET, split=config.ASQA_SPLIT)
    return ds.select(range(min(n, len(ds))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=config.ASQA_SUBSET)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "baseline.jsonl"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}

    data = load_asqa(args.limit)
    rag = RAGPipeline()
    with open(args.out, "a", encoding="utf-8") as f:
        for row in tqdm(data, desc="ASQA"):
            if row["sample_id"] in done:
                continue
            res = rag.run(row["ambiguous_question"])
            res.update({
                "sample_id":   row["sample_id"],
                # kept for ALCE evaluation in Phase 4
                "qa_pairs":    row["qa_pairs"],
                "annotations": row["annotations"],
            })
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
