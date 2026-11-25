
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from factcheck_rag import config

# ALCE instruction (Gao et al., 2023), so our baseline matches the protocol
_SYSTEM = (
    "Write an accurate, engaging, and concise answer for the given question "
    "using only the provided search results (some of which might be "
    "irrelevant) and cite them properly. Use an unbiased and journalistic "
    "tone. Always cite for any factual claim. When citing several search "
    "results, use [1][2][3]. Cite at least one document and at most three "
    "documents in each sentence, placing the citations at the end of that "
    "sentence. Never list citations separately at the end of the answer. "
    "If the question is ambiguous, cover each plausible interpretation. "
    "If the search results do not contain the answer, say so in one sentence."
)

def _build_prompt(question, passages):
    block = "\n\n".join(
        f"[{i+1}] (Title: {p.get('title','')}) {p['text']}"
        for i, p in enumerate(passages)
    )
    return (
        f"Passages:\n{block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

class Generator:
    """Wraps the LLM. `chat` is shared by the answer generator, the claim
    decomposer and the LLM judge, so the 7B model is loaded only once."""

    def __init__(self):
        print(f"Loading {config.GEN_MODEL} ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            config.GEN_MODEL,
            quantization_config=bnb,
            device_map="auto",
        )
        print("Generator ready.")

    def chat(self, system: str, user: str,
             max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy = deterministic
                pad_token_id=self._tokenizer.eos_token_id,
            )
        decoded = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return decoded.strip()

    def generate(self, question: str, passages: list) -> str:
        return self.chat(_SYSTEM, _build_prompt(question, passages))
