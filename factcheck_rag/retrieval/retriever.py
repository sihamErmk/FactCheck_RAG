
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from factcheck_rag import config

class Retriever:
    def __init__(self):
        print("Loading FAISS index ...")
        self._index = faiss.read_index(config.INDEX_PATH)
        print("Loading passages ...")
        self._passages = [json.loads(l) for l in open(config.PASSAGES_PATH)]
        print("Loading embedding model ...")
        self._model = SentenceTransformer(config.EMBED_MODEL)
        print("Retriever ready.")

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list:
        vec = self._model.encode(
            [config.QUERY_PREFIX + query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, indices = self._index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            passage = dict(self._passages[idx])
            passage["score"] = float(score)
            results.append(passage)
        return results
