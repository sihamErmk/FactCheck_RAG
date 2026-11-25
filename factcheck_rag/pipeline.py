
import time
from factcheck_rag.retrieval.retriever import Retriever
from factcheck_rag.generation.generator import Generator
from factcheck_rag import config

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(self, question: str, top_k: int = config.TOP_K) -> dict:
        t0       = time.perf_counter()
        passages = self.retriever.retrieve(question, top_k=top_k)
        answer   = self.generator.generate(question, passages)
        return {
            "question":  question,
            "passages":  passages,
            "answer":    answer,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
