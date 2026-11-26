# FactCheck-RAG

> Iterative atomic claim verification and targeted re-retrieval
> for hallucination mitigation in RAG systems.

**Université Claude Bernard Lyon 1 — MLTA 2026**  
Siham El Kouaze · Aya Ait Sidi Abdelkrim · Aghiles Merzouki  
Instructor: Chao Zhang

---

## What is FactCheck-RAG?

Standard RAG retrieves passages and generates an answer in one shot.
The model can still write fluent sentences that no retrieved passage
actually supports — and it has no way of noticing.

FactCheck-RAG fixes this by adding a **verification and repair loop**
after generation. Each factual claim in the answer is checked
individually, and only the unsupported ones are re-retrieved and rewritten.

---

## Full Pipeline (All Phases)

```
Question
    │
    ▼
┌─────────────┐
│  Retriever  │  FAISS + bge-large-en over Wikipedia
└──────┬──────┘
       │  top-k passages
       ▼
┌─────────────┐
│  Generator  │  Qwen2.5-7B-Instruct (4-bit)
└──────┬──────┘
       │  draft answer
       ▼
┌──────────────────┐
│ Claim Decomposer │  draft → atomic claims
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   Verifier   │  DeBERTa-v3 NLI or LLM judge
└──────┬───────┘
       │                         │
  all supported             unsupported claim
       │                         │
       ▼                         ▼
 Final Answer          Targeted sub-query
 with citations     → Retriever → Rewrite claim
                    → Verify again (max T=3 rounds)
```

---

## Project Structure

```
FactCheck_RAG/
├── factcheck_rag/              # the code (Python package)
│   ├── config.py               # all settings
│   ├── pipeline.py             # standard RAG: retriever + generator         (Phase 1)
│   ├── retrieval/              # indexer.py, retriever.py                    (Phase 1)
│   ├── generation/             # generator.py                                (Phase 1)
│   ├── verification/           # decomposer.py, verifier.py, factchecker.py  (Phase 2)
│   └── repair/                 # repairer.py: sub-query, re-retrieve, rewrite (Phase 3)
├── scripts/                    # what you run: build_index, run_baseline, run_verify, run_repair
├── tests/                      # unit tests, no GPU needed
├── notebooks/
│   └── colab_runner.ipynb      # runs everything on Colab
├── requirements.txt
└── README.md
```

`data/` (index) and `outputs/` (results) are created when you run the
scripts and are not in git. A new folder is added to `factcheck_rag/` when
its phase starts.

Run the unit tests (no GPU needed): `python tests/test_text_parsing.py` and `python tests/test_repair.py`

**On Colab:** open `notebooks/colab_runner.ipynb` (*File → Upload notebook* or *Open from GitHub*). It clones this repo, links `data/` and `outputs/` to `MyDrive/factcheck_rag/` so the index and results persist, and runs every step.

---

## Phase 1 — Baseline RAG ✅

### Goal
Build a working single-pass RAG pipeline end to end
before adding any verification or repair logic.

### What was built

`factcheck_rag/retrieval/`, `factcheck_rag/generation/`,
`factcheck_rag/pipeline.py`, `scripts/build_index.py`, `scripts/run_baseline.py`.

### How Phase 1 works step by step

```
Step 1 — indexer.py   (ASQA-targeted corpus)
    Collect the Wikipedia pages ASQA annotators used, for all 948 dev
      questions (2016 titles; the 748 non-evaluated questions = distractors)
    Download their wikitext, 50 pages per API request (one-by-one requests
      get rate-limited with HTTP 429); convert to plain text, keeping tables
    Chunk into 100-word passages (ALCE convention) → 69,665 passages
    Embed with bge-large-en-v1.5 (fp16) → FAISS IndexFlatIP (cosine)
    Save → data/wiki_pages_v2.jsonl, data/passages.jsonl, data/faiss.index
    `--no-embed` runs the download on a CPU runtime; embedding needs a GPU

Step 2 — retriever.py
    Load FAISS index + passages
    Encode question with bge-large-en-v1.5 (+ bge query instruction prefix)
    Search index → return top-5 passages with scores

Step 3 — generator.py
    Load Qwen2.5-7B-Instruct (4-bit, bitsandbytes)
    ALCE instruction prompt: system + passages + question
    Generate answer with citation tags [1][2] at the end of each sentence

Step 4 — pipeline.py
    question → retriever → generator → result dict
    result = { question, passages, answer, latency_s }
```

### Tech Stack

| Component | Tool |
|---|---|
| Vector index | FAISS (IndexFlatIP) |
| Embeddings | BAAI/bge-large-en-v1.5 (1024 dim) |
| Knowledge base | Wikipedia pages cited by ASQA dev (current version, via API) |
| Generator | Qwen2.5-7B-Instruct (4-bit bitsandbytes) |
| Dataset | ASQA dev split (200 questions) |
| Environment | Google Colab T4 GPU (15 GB VRAM) |

### Phase 1 Results

| Metric | Value |
|---|---|
| Pages / passages indexed | 1,974 pages (18 titles no longer exist) / 69,665 passages |
| Wikipedia download | ~9 min (CPU) |
| Embedding time | ~14 min (T4, fp16) |
| Latency per question | ~6 seconds |
| GPU | Tesla T4 |

> The first version indexed the first 100k passages of the Wikipedia dump,
> which almost never contained the ASQA answers (the model answered "the
> passages do not contain the information"). The ASQA-targeted corpus fixed
> this.
>
> Known limits: pages are today's Wikipedia, while ASQA was annotated on a
> 2018 dump, so a few answers have changed (e.g. goal records).

### Example Output

```
Question : Who was the first person to walk on the moon?

Answer   : The first person to walk on the Moon was Neil Armstrong [1].
           Specifically, Armstrong became the first person to step onto
           the Moon's surface six hours and 39 minutes after landing
           on July 20, 1969.

Top passage: Apollo 11 (Wikipedia)
```

### How to Reproduce

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. build the FAISS index (runs once, ~25 min; cached steps are skipped)
python scripts/build_index.py --no-embed   # optional: download on a CPU runtime
python scripts/build_index.py              # GPU runtime: embedding + FAISS

# 3. test one question (from the repo root)
python -c "
from factcheck_rag.pipeline import RAGPipeline
rag = RAGPipeline()
result = rag.run('Who was the first person to walk on the moon?')
print(result['answer'])
"

# 4. run on 200 ASQA questions
python scripts/run_baseline.py
```

---

## Phase 2 — Claim decomposer + verifiers ✅

### What was built

```
factcheck_rag/verification/
├── decomposer.py    # answer -> sentences (with their [n] tags) -> atomic claims
├── verifier.py      # NLIVerifier (DeBERTa-v3) and LLMJudgeVerifier, same interface
└── factchecker.py   # ClaimChecker: decompose + verify, claim precision
scripts/run_verify.py  # verify the baseline answers, report Claim P
```

`generator.py` now exposes `chat(system, user)`, so the decomposer and the
LLM judge reuse the already-loaded Qwen model (one 7B model in memory).

### How it works

```
Step 1 — decomposer.py
    Split the draft answer into sentences, keep each sentence's citations
    LLM rewrites each sentence as atomic claims ("- ..." lines); it also sees
      the previous sentences, so pronouns ("it", "their") are replaced by names
    Each claim inherits its sentence's citations

Step 2 — evidence (factchecker.py)
    Evidence = passages cited by the claim
    No citation -> all retrieved passages, claim marked cited=False

Step 3 — verifier.py  (config.VERIFIER = "nli" | "judge")
    nli   : DeBERTa-v3-large NLI; premises = each cited passage + their
            concatenation; supported if max P(entailment) >= 0.5
    judge : Qwen first quotes the evidence sentence, then gives
            "Verdict: SUPPORTED / UNSUPPORTED" (quote saved for manual checks)

Output per answer: claims[], n_claims, n_supported, claim_precision
```

### Phase 2 results (smoke test, 5 ASQA questions)

| Verifier | Claims | Claim P | Time / question |
|---|---|---|---|
| NLI (DeBERTa-v3-large) | 15 (3.0 / answer) | 1.000 | ~5 s |
| LLM judge (Qwen, quote + verdict) | 15 | 0.933 | ~15 s |

The two verifiers agree on 14 of 15 claims. Lessons from the iterations:

- A one-word judge ("SUPPORTED/UNSUPPORTED") was far too strict (Claim P
  0.53, it rejected "44 official state parks" although the passage says it
  verbatim). Asking for the supporting quote first fixed it.
- Without the previous sentences as context, claims kept pronouns
  ("It was announced by Steve Jobs") and could not be re-queried.
- Both verifiers can still be lenient (e.g. "Simon & Garfunkel wrote the
  song" passes, although Paul Simon wrote it). This is measured in Phase 5
  (manual check of 50 claims).

### How to run

```bash
python scripts/run_baseline.py                 # -> outputs/baseline.jsonl
python scripts/run_verify.py --verifier nli    # -> outputs/verified_nli.jsonl
python scripts/run_verify.py --verifier judge  # -> outputs/verified_judge.jsonl
```

---

## Phase 3 — Repair loop 🔄

### What was built

```
factcheck_rag/repair/
└── repairer.py      # Repairer: sub-query -> retrieve -> rewrite -> verify, max T rounds
scripts/run_repair.py  # repair the answers checked by run_verify.py
tests/test_repair.py   # GPU-free tests with a fake LLM, retriever and verifier
```

It reuses the loaded Qwen (sub-queries and rewrites), the Phase 1 retriever,
and `ClaimChecker.verify_claims` (same evidence rule and verifier as Phase 2).

### How it works

```
For each round r = 1 .. T (T = config.MAX_ROUNDS = 3):
    for each claim still unsupported:
        Qwen writes one short sub-query for that claim
          (it sees the queries already tried for it, so it does not repeat them)
        Retrieve top-5 passages; new ones are appended to the answer's passage
          list, so [n] keeps meaning passages[n-1]
        Qwen rewrites ONLY that claim from those passages:
          unchanged if they state it, corrected if they state another fact,
          NONE if they say nothing -> claim dropped
        Verify the rewritten claim against the passages it cites
    stop early when no claim is left to repair

Claims still unsupported after round T are dropped.
Final answer: sentences whose claims were all supported are kept word for
word; any other sentence is replaced by its surviving claims, each with its
own citations. Nothing left -> "The search results do not contain enough
information to answer the question."

Output per answer: answer, passages (extended), claims[] with status
kept / repaired / dropped and per-round history, rounds, n_kept,
n_repaired, n_dropped, repair_s
```

> For Phase 4: the loop only keeps claims its verifier accepted, so the
> final answer's claim precision is 1.0 by construction under that same
> verifier. Measure it with the other verifier, the ALCE citation metrics or
> the manual check, and report `n_dropped` next to it.

### How to run

```bash
python scripts/run_repair.py --verifier nli                 # -> outputs/repaired_nli_T3.jsonl
python scripts/run_repair.py --verifier judge               # -> outputs/repaired_judge_T3.jsonl
python scripts/run_repair.py --verifier nli --max-rounds 1  # rounds ablation (Phase 5)
```

---

## Phases Overview

| Phase | Description | Status |
|---|---|---|
| 1 | Baseline RAG pipeline | ✅ Done |
| 2 | Claim decomposer + verifiers | ✅ Done |
| 3 | Repair loop | 🔄 Code + unit tests done, first GPU run pending |
| 4 | Evaluation on 200 ASQA questions | 🔄 Next |
| 5 | Ablations and manual validation | ⏳ Pending |

---

## Research Question

> Does claim-level verification with targeted re-retrieval raise
> faithfulness over single-pass RAG,
> and is the extra latency worth it?

---

## References

- RARR — Gao et al., 2023
- Chain-of-Verification — Dhuliawala et al., 2023
- FLARE — Jiang et al., 2023
- Self-RAG — Asai et al., 2023
- ASQA — Stelmakh et al., 2022
- ALCE — Gao et al., 2023
