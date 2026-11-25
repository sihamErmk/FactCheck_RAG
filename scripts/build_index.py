"""Build the ASQA-targeted FAISS index (Phase 1).

    python scripts/build_index.py --no-embed   # CPU runtime: download + chunk only
    python scripts/build_index.py              # GPU runtime: everything (cached steps skipped)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from factcheck_rag.retrieval.indexer import build_index

if __name__ == "__main__":
    build_index(embed="--no-embed" not in sys.argv)
