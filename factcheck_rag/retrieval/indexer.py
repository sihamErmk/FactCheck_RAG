"""Build the FAISS index over an ASQA-targeted Wikipedia corpus.

Corpus = the Wikipedia pages that ASQA annotators used (`wikipages` and
`qa_pairs[].wikipage`) for EVERY question of the split (948 in dev).
Our 200 evaluation questions are the first 200; the pages of the other
748 questions act as realistic distractors, so retrieval is not trivial.

Steps (each one is cached, so a crash does not restart from zero):
    1. collect page titles from ASQA         -> data/wiki_titles.json
    2. download wikitext (50 pages/request)   -> data/wiki_pages_v2.jsonl
    3. chunk into CHUNK_WORDS-word passages   -> data/passages.jsonl
    4. embed with bge-large + FAISS           -> data/faiss.index

    python scripts/build_index.py --no-embed   # CPU runtime: steps 1-3 only
    python scripts/build_index.py              # GPU runtime: everything (cached steps are skipped)
"""
import json, os, time
from urllib.parse import unquote
import requests
from tqdm import tqdm
from factcheck_rag import config

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "FactCheckRAG-course-project/0.1 (academic, non-commercial)"}


# ------------------------------------------------------------ 1. titles
def _title_from_url(url: str) -> str:
    return unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ").strip()


def collect_titles() -> list:
    from datasets import load_dataset
    ds = load_dataset(config.ASQA_DATASET, split=config.ASQA_SPLIT)
    titles = set()
    for row in ds:
        for wp in row["wikipages"] or []:
            if wp.get("url"):
                titles.add(_title_from_url(wp["url"]))
            elif wp.get("title") and not wp["title"].endswith("..."):
                titles.add(wp["title"].strip())
        for qa in row["qa_pairs"] or []:
            if qa.get("wikipage"):
                titles.add(qa["wikipage"].strip())
    return sorted(t for t in titles if t)


# ------------------------------------------------------------ 2. download
# We ask for the raw wikitext of 50 pages per request (one request per page
# got rate-limited with HTTP 429), then convert wikitext -> plain text with
# mwparserfromhell. Unlike the TextExtracts API this keeps table cells, which
# matters for ASQA "List of ..." pages.
_BATCH = 50
_DROP_SECTIONS = {"references", "see also", "external links", "notes",
                  "further reading", "bibliography", "sources", "citations",
                  "footnotes"}


def wikitext_to_text(wikitext: str) -> str:
    import mwparserfromhell as mw
    code = mw.parse(wikitext)
    for sec in code.get_sections(levels=[2]):
        heading = sec.filter_headings()[0].title.strip_code().strip().lower()
        if heading in _DROP_SECTIONS:
            try:
                code.remove(sec)
            except ValueError:
                pass
    for link in code.filter_wikilinks():
        if str(link.title).strip().lower().startswith(("file:", "image:", "category:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    for tag in code.filter_tags():
        if str(tag.tag).lower() in ("ref", "gallery", "references"):
            try:
                code.remove(tag)
            except ValueError:
                pass
    return code.strip_code(normalize=True, collapse=True).strip()


def _get(params: dict) -> dict:
    wait = 5
    for _ in range(8):
        try:
            r = requests.get(_API, params=params, headers=_HEADERS, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(int(r.headers.get("Retry-After", wait)))
                wait = min(wait * 2, 120)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry after error: {e}")
            time.sleep(wait)
            wait = min(wait * 2, 120)
    raise RuntimeError("Wikipedia API keeps failing; re-run scripts/build_index.py later")


def fetch_batch(titles: list) -> dict:
    """Return {query_title: {"status", "title", "text"}} for up to 50 titles."""
    params = {
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "redirects": 1, "titles": "|".join(titles),
    }
    pages, alias = {}, {}
    cont = {}
    while True:                      # big batches are split with "continue"
        j = _get({**params, **cont})
        q = j.get("query", {})
        for m in q.get("normalized", []) + q.get("redirects", []):
            alias[m["from"]] = m["to"]
        for p in q.get("pages", []):
            content = (p.get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("content")
            if content is not None or p["title"] not in pages:
                pages[p["title"]] = {"missing": bool(p.get("missing") or p.get("invalid")),
                                     "content": content}
        if "continue" not in j:
            break
        cont = j["continue"]

    out = {}
    for t in titles:
        final, seen = t, set()
        while final in alias and final not in seen:
            seen.add(final)
            final = alias[final]
        p = pages.get(final)
        if not p or p["missing"] or not p["content"]:
            out[t] = {"status": "missing", "title": final, "text": ""}
        else:
            out[t] = {"status": "ok", "title": final, "text": wikitext_to_text(p["content"])}
    return out


def download_pages(titles: list, path: str) -> list:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for l in f:
                p = json.loads(l)
                done[p["query_title"]] = p
    todo = [t for t in titles if t not in done]
    print(f"  {len(done)} pages cached, {len(todo)} to download")
    batches = [todo[i : i + _BATCH] for i in range(0, len(todo), _BATCH)]
    for batch in tqdm(batches, desc="Wikipedia batches"):
        res = fetch_batch(batch)
        with open(path, "a", encoding="utf-8") as f:
            for t in batch:
                rec = {"query_title": t, **res[t]}
                done[t] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(config.WIKI_DELAY)
    n_missing = sum(p["status"] == "missing" for p in done.values())
    print(f"  {n_missing} titles do not exist on Wikipedia")
    # several query titles can redirect to the same page: keep one copy
    pages = {}
    for p in done.values():
        if p["text"]:
            pages.setdefault(p["title"], p)
    return list(pages.values())


# ------------------------------------------------------------ 3. chunk
def chunk_pages(pages: list) -> list:
    passages = []
    n = config.CHUNK_WORDS
    for page in pages:
        # drop the reference-like tail sections
        text = page["text"]
        for cut in ("\n== References ==", "\n== See also ==", "\n== External links ==", "\n== Notes =="):
            text = text.split(cut)[0]
        words = text.replace("=", " ").split()
        for i in range(0, len(words), n):
            chunk = " ".join(words[i : i + n])
            if len(chunk) > 30:
                passages.append({"id": len(passages), "title": page["title"], "text": chunk})
    return passages


# ------------------------------------------------------------ 4. embed
def embed_and_index(passages: list):
    import faiss, torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    if torch.cuda.is_available():
        model = model.half()                 # ~2x faster on T4, same quality
    vecs = model.encode(
        [p["text"] for p in passages],
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"  Embedding matrix: {vecs.shape}")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, config.INDEX_PATH)
    print(f"  Saved {config.INDEX_PATH} ({index.ntotal:,} vectors)")


def build_index(embed: bool = True):
    os.makedirs(config.DATA_DIR, exist_ok=True)

    titles_path = os.path.join(config.DATA_DIR, "wiki_titles.json")
    print("1/4 Collecting ASQA page titles ...")
    if os.path.exists(titles_path):
        titles = json.load(open(titles_path, encoding="utf-8"))
    else:
        titles = collect_titles()
        json.dump(titles, open(titles_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  {len(titles)} unique titles")

    print("2/4 Downloading pages ...")
    pages = download_pages(titles, os.path.join(config.DATA_DIR, "wiki_pages_v2.jsonl"))
    print(f"  {len(pages)} pages with text")

    print(f"3/4 Chunking into {config.CHUNK_WORDS}-word passages ...")
    passages = chunk_pages(pages)
    with open(config.PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {len(passages):,} passages -> {config.PASSAGES_PATH}")

    print(f"4/4 Embedding with {config.EMBED_MODEL} ...")
    if not embed:
        print("Stopping before embedding (--no-embed). Switch to a GPU runtime "
              "and run: python scripts/build_index.py")
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU: embedding on CPU would take hours. Switch the "
                         "runtime to T4 GPU and run python scripts/build_index.py again "
                         "(downloaded pages are cached).")
    embed_and_index(passages)
