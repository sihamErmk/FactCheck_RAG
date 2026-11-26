"""All settings in one place. Paths are relative to the repository root,
so scripts work from any working directory (Colab, Kaggle, local)."""
import os

ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
INDEX_PATH      = os.path.join(DATA_DIR, "faiss.index")
PASSAGES_PATH   = os.path.join(DATA_DIR, "passages.jsonl")
RESULTS_DIR     = os.path.join(ROOT_DIR, "outputs")

EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
EMBED_BATCH     = 256
TOP_K           = 5

GEN_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.0

ASQA_DATASET    = "din0s/asqa"
ASQA_SPLIT      = "dev"
ASQA_SUBSET     = 200

# corpus = Wikipedia pages used by ASQA annotators (see indexer.py)
CHUNK_WORDS     = 100            # ALCE uses 100-word passages
WIKI_DELAY      = 0.5            # seconds between Wikipedia API calls (429 if faster)

# bge recommends this instruction in front of queries (not passages)
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "

# ---- Phase 2: claim decomposition + verification ----
DECOMP_MAX_TOKENS = 256          # max tokens for one sentence -> claims call

# "nli" (DeBERTa-v3) or "judge" (LLM judge, reuses the generator model)
VERIFIER        = "nli"

NLI_MODEL       = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_THRESHOLD   = 0.5            # entailment prob needed to call a claim supported
NLI_MAX_LENGTH  = 512            # DeBERTa max input tokens (premise is truncated)
NLI_BATCH       = 8

JUDGE_MAX_TOKENS = 160          # room for "Quote: ..." + "Verdict: ..."

# ---- Phase 3: repair loop ----
REPAIR_MAX_ROUNDS   = 3          # T: stop after this many repair rounds
REPAIR_TOP_K        = 3          # passages retrieved per sub-query
SUBQUERY_MAX_TOKENS = 32
REWRITE_MAX_TOKENS  = 200
REWRITE_MAX_EVIDENCE = 8         # max passages shown to the rewriter for one sentence
