"""Two interchangeable claim verifiers.

Both expose the same method:

    verify(claim_text: str, evidence: list[passage]) -> {
        "supported": bool,
        "score":     float,   # NLI: max entailment prob; judge: 1.0 / 0.0
        "verifier":  "nli" | "judge",
    }

`evidence` is the list of passages the claim cites (see ClaimChecker in
factchecker.py for how evidence is chosen).
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from factcheck_rag import config


def _passage_text(p: dict) -> str:
    title = p.get("title", "")
    return f"{title}. {p['text']}" if title else p["text"]


# --------------------------------------------------------------------- NLI
class NLIVerifier:
    """DeBERTa-v3 NLI. A claim is supported if some premise entails it with
    probability >= threshold. Premises = each cited passage alone, plus the
    concatenation of all cited passages (for claims that combine facts from
    two passages)."""

    name = "nli"

    def __init__(self, model_name: str = config.NLI_MODEL,
                 threshold: float = config.NLI_THRESHOLD):
        print(f"Loading NLI model {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.threshold = threshold
        labels = {i: l.lower() for i, l in self._model.config.id2label.items()}
        self._entail_idx = next(i for i, l in labels.items() if "entail" in l)
        print("NLI verifier ready.")

    @torch.no_grad()
    def entailment_probs(self, premises: list, hypothesis: str) -> list:
        probs = []
        for i in range(0, len(premises), config.NLI_BATCH):
            batch = premises[i : i + config.NLI_BATCH]
            enc = self._tok(
                batch,
                [hypothesis] * len(batch),
                truncation="only_first",          # cut the premise, never the claim
                max_length=config.NLI_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**enc).logits.float()
            probs.extend(torch.softmax(logits, dim=-1)[:, self._entail_idx].tolist())
        return probs

    def verify(self, claim: str, evidence: list) -> dict:
        if not evidence:
            return {"supported": False, "score": 0.0, "verifier": self.name}
        premises = [_passage_text(p) for p in evidence]
        if len(premises) > 1:
            premises.append(" ".join(premises))
        score = max(self.entailment_probs(premises, claim))
        return {
            "supported": score >= self.threshold,
            "score":     round(score, 4),
            "verifier":  self.name,
        }


# --------------------------------------------------------------- LLM judge
_JUDGE_SYSTEM = (
    "You are a fact-checking judge. You are given evidence passages and a "
    "claim. Decide whether the evidence supports the claim. Use only the "
    "evidence, not your own knowledge.\n"
    "First, copy the sentence(s) from the evidence that state the claim's "
    "fact (paraphrases and different wording are fine). Then give the "
    "verdict: SUPPORTED if the quoted text states the claim, UNSUPPORTED if "
    "no sentence states it or the evidence contradicts it.\n"
    "Reply in exactly this format:\n"
    "Quote: <copied sentence(s), or NONE>\n"
    "Verdict: SUPPORTED or UNSUPPORTED"
)


def _is_supported(text: str) -> bool:
    text = text.upper()
    if "UNSUPPORTED" in text or "NOT SUPPORTED" in text or "NOT_SUPPORTED" in text:
        return False
    return "SUPPORTED" in text


def parse_judgement(raw: str):
    """Return (supported, quote). Reads the 'Verdict:' line when present."""
    quote = ""
    verdict = None
    for line in raw.splitlines():
        low = line.strip().lower()
        if low.startswith("quote:"):
            quote = line.split(":", 1)[1].strip()
        elif low.startswith("verdict:"):
            verdict = _is_supported(line.split(":", 1)[1])
    if verdict is None:                       # model ignored the format
        verdict = _is_supported(raw.splitlines()[-1] if raw.strip() else "")
    return verdict, quote


class LLMJudgeVerifier:
    """Uses the generator LLM as a judge (no extra model in memory)."""

    name = "judge"

    def __init__(self, generator):
        self._gen = generator

    def verify(self, claim: str, evidence: list) -> dict:
        if not evidence:
            return {"supported": False, "score": 0.0, "verifier": self.name}
        block = "\n\n".join(
            f"[{i+1}] {_passage_text(p)}" for i, p in enumerate(evidence)
        )
        user = f"Evidence:\n{block}\n\nClaim: {claim}"
        raw = self._gen.chat(_JUDGE_SYSTEM, user, max_new_tokens=config.JUDGE_MAX_TOKENS)
        ok, quote = parse_judgement(raw)
        return {
            "supported": ok,
            "score":     1.0 if ok else 0.0,
            "verifier":  self.name,
            "quote":     quote,
            "raw":       raw,
        }


def build_verifier(kind: str = config.VERIFIER, generator=None):
    if kind == "nli":
        return NLIVerifier()
    if kind == "judge":
        if generator is None:
            raise ValueError("the LLM judge needs the loaded Generator")
        return LLMJudgeVerifier(generator)
    raise ValueError(f"unknown verifier: {kind!r} (use 'nli' or 'judge')")
