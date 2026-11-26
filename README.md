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
│   └── repair/                 # subquery.py, rewriter.py, loop.py           (Phase 3)
├── scripts/                    # what you run: build_index, run_baseline, run_verify, run_factcheck
├── tests/                      # unit tests, no GPU needed
├── notebooks/
│   └── colab_runner.ipynb      # runs everything on Colab
├── requirements.txt
└── README.md
```

`data/` (index) and `outputs/` (results) are created when you run the
scripts and are not in git. A new folder is added to `factcheck_rag/` when
its phase starts (e.g. `evaluation/` for Phase 4).

Run the unit tests (no GPU needed): `python tests/test_text_parsing.py` and `python tests/test_repair_loop.py`

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

## Phase 3 — Repair loop ✅

### What was built

```
factcheck_rag/repair/
├── subquery.py      # LLM writes one short search query per unsupported claim
├── rewriter.py      # LLM rewrites ONE sentence: fix or delete its unsupported facts
└── loop.py          # RepairLoop (repair a draft) and FactCheckRAG (end to end)
scripts/run_factcheck.py   # repair the baseline drafts, report Claim P per round
```

### How it works

```
round 0   decompose the draft + verify every claim            (Phase 2)
round t   for each sentence holding unsupported claims:
            for each unsupported claim:
              sub-query  -> retrieve top-3 new passages
              new passages are appended: [1..5] original, [6..] new
            rewrite ONLY that sentence with the new evidence
              (fix the fact, or delete it if the evidence says nothing)
            re-decompose + re-verify ONLY that sentence
stop      when every claim is supported, or after T = 3 rounds
```

- Sentences without unsupported claims are kept verbatim.
- Safeguards (found when testing on Colab): a rewrite is **rejected** if the
  sentence loses supported facts (Qwen sometimes changes a correct fact), and
  a sentence the rewriter returns **unchanged** is not retried in later rounds.
- Each sub-query brings up to 3 passages that are not already in the answer.
- FactCheck-RAG starts from the same drafts and passages as standard RAG
  (`outputs/baseline.jsonl`), so the comparison isolates the repair loop.
- One snapshot is stored per round. Since rounds are sequential, the
  snapshot after round t is the output of a run with T = t, so a single
  T = 3 run also gives the T = 1 and T = 2 results (Phase 5 ablation).
- Each output line keeps the draft, the final answer, every sub-query and
  rewrite, the claims, the number of rounds and the latency.

### How to run

```bash
python scripts/run_factcheck.py --verifier nli   --limit 5   # -> outputs/factcheck_nli.jsonl
python scripts/run_factcheck.py --verifier judge --limit 5   # -> outputs/factcheck_judge.jsonl
```

### Phase 3 results (first 20 ASQA dev questions, same drafts as standard RAG)

| Verifier | Claim P before → after | T = 1 | T = 2 | T = 3 | Latency / question | Rewrites applied |
|---|---|---|---|---|---|---|
| NLI | 0.922 → 0.978 | 0.978 | 0.978 | 0.978 | 12.0 s | 1 of 3 attempts |
| LLM judge | 0.859 → 0.959 | 0.959 | 0.959 | 0.959 | 22.6 s | 4 of 6 attempts |

Standard RAG alone takes ~6 s per question.

- The whole gain comes from round 1; rounds 2 and 3 add nothing on this sample.
- The stricter judge triggers twice as many repairs, at about 3x the cost.
- Repairs seen: wrong citations fixed (`[2]` → `[2][5]`), invented
  conclusions deleted ("all 28 states were under Congress"), unsupported
  details removed ("fifth Finals win in eleven seasons").
- Bad rewrites are now rejected (e.g. Qwen turned "after the 2004
  amalgamation" into "prior to"), keeping the original sentence.
- Trade-off: faithfulness to the retrieved passages can remove a true fact
  the passages do not state (Dua Lipa at the 2018 final, judge run). Phase 4
  measures this with Str-EM.
- Known limitation: the decomposer still returns no claim for some
  sentences (e.g. the Congress answer); checked by hand in Phase 5.

---

## Phases Overview

| Phase | Description | Status |
|---|---|---|
| 1 | Baseline RAG pipeline | ✅ Done |
| 2 | Claim decomposer + verifiers | ✅ Done |
| 3 | Repair loop | ✅ Done |
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
