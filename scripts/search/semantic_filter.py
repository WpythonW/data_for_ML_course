"""
Multi-stage dataset filtering pipeline.

Stages:
  1. BM25 — broad keyword ranking
  2. OpenRouter — summarize + filter candidates (lаконично, экономно)
  3. OpenRouter — rerank survivors
  4. Haiku step 1 — review BM25+rerank scores, pick which datasets warrant questions,
                    generate targeted questions per dataset
  5. OpenRouter — answer Haiku's questions (laconic)
  6. Haiku step 2 — read answers, make final selection (narrow bottleneck)

Global deduplication: rejected IDs are written to --rejected-ids-file and
never passed to OpenRouter or Haiku again.

Usage:
    uv run scripts/search/semantic_filter.py \
        --input data/raw_results.json \
        --goal "car brand/make image classification dataset" \
        --queries "car brand,car make,..." \
        --keywords "Toyota,BMW,Ford,..." \
        --bm25-top 60 \
        --output data/filtered_results.json \
        --rejected-ids-file data/rejected_ids.txt
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/dataset-agent",
}
HAIKU_MODEL = "anthropic/claude-haiku-4-5"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return key


async def openrouter_call_async(session, model: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
    resp = await session.post(
        OPENROUTER_URL,
        headers={**OPENROUTER_HEADERS, "Authorization": f"Bearer {get_api_key()}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def openrouter_call(model: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
    """Sync wrapper for backward compatibility."""
    import httpx
    with httpx.Client() as client:
        resp = client.post(
            OPENROUTER_URL,
            headers={**OPENROUTER_HEADERS, "Authorization": f"Bearer {get_api_key()}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_response(content: str) -> list | dict:
    """Strip markdown fences and parse JSON."""
    content = content.strip()
    if "```" in content:
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith(("[", "{")):
                content = part
                break
    return json.loads(content)


def load_rejected_ids(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    return set(line.strip() for line in p.read_text().splitlines() if line.strip())


def save_rejected_ids(path: str, ids: set):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(sorted(ids)))


def compact_dataset(d: dict) -> str:
    """One-line compact representation for large batch prompts."""
    tags = ", ".join((d.get("tags") or [])[:8])
    return (
        f"ID: {d['id']} | DL: {d.get('downloads',0)} | "
        f"License: {d.get('license','?')} | Size: {', '.join(d.get('size_categories',[]) or ['?'])} | "
        f"Tags: {tags} | Desc: {(d.get('description') or '')[:300]}"
    )


# ── Stage 1: BM25 ─────────────────────────────────────────────────────────────

def bm25_filter(datasets: list[dict], queries: list[str], keywords: list[str], top_n: int) -> tuple[list[dict], dict]:
    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    corpus = [tokenize(d.get("corpus_text", "") or "") for d in datasets]
    bm25 = BM25Okapi(corpus)
    all_tokens = tokenize(" ".join(queries + keywords))
    scores = bm25.get_scores(all_tokens)

    for i, ds in enumerate(datasets):
        scores[i] += ds.get("query_match_count", 1) * 0.05

    ranked = sorted(zip(scores, datasets), key=lambda x: x[0], reverse=True)
    top = ranked[:top_n]

    score_map = {d["id"]: float(s) for s, d in ranked}
    result = [d for _, d in top]

    if top:
        print(f"  BM25: {len(datasets)} → {len(result)} (top: {top[0][0]:.2f}, cutoff: {top[-1][0]:.2f})")
    return result, score_map


# ── Stage 2: OpenRouter summarize + filter ────────────────────────────────────

def openrouter_filter(datasets: list[dict], goal: str, or_model: str) -> list[dict]:
    """
    OpenRouter reads all BM25 survivors, summarizes each to the necessary detail level,
    and drops clearly irrelevant ones. Returns survivors with added 'or_summary' field.
    It decides itself how many to keep — be economical but don't lose relevant datasets.
    """
    batch = "\n\n".join(
        f"[{i+1}] {compact_dataset(d)}" for i, d in enumerate(datasets)
    )

    prompt = f"""You are an ML dataset curator assistant. Be concise and token-efficient.

GOAL: {goal}

Below are {len(datasets)} dataset candidates. For each:
1. Write a summary preserving ALL information relevant to the goal (size, modality, annotations, license, task). Length: as needed — do NOT lose relevant details.
2. Decide: KEEP or REJECT. Reject only if clearly irrelevant to the goal.

Return ONLY valid JSON array:
[
  {{"id": "owner/name", "keep": true, "summary": "..."}},
  ...
]

CANDIDATES:
{batch}"""

    print(f"  OpenRouter filter: sending {len(datasets)} datasets...", end=" ", flush=True)
    try:
        raw = openrouter_call(or_model, prompt, max_tokens=4000)
        items = parse_json_response(raw)
    except Exception as e:
        print(f"\n  Warning: OpenRouter filter failed: {e}", file=sys.stderr)
        return datasets  # pass-through on failure

    id_to_ds = {d["id"]: d for d in datasets}
    survivors = []
    rejected = []
    for item in items:
        ds_id = item.get("id", "")
        if ds_id not in id_to_ds:
            continue
        if item.get("keep", False):
            ds = dict(id_to_ds[ds_id])
            ds["or_summary"] = item.get("summary", "")
            survivors.append(ds)
        else:
            rejected.append(ds_id)

    print(f"kept {len(survivors)}, rejected {len(rejected)}")
    return survivors, set(rejected)


# ── Stage 3: OpenRouter rerank ────────────────────────────────────────────────

def openrouter_rerank(datasets: list[dict], goal: str, or_model: str) -> list[dict]:
    """
    Rerank survivors by relevance. Uses or_summary if available.
    Returns datasets sorted by or_rank with score attached.
    """
    batch = "\n\n".join(
        f"[{i+1}] ID: {d['id']}\n{d.get('or_summary') or compact_dataset(d)}"
        for i, d in enumerate(datasets)
    )

    prompt = f"""Rank these datasets by relevance to the goal. Be concise.

GOAL: {goal}

Return ONLY valid JSON array sorted best-first:
[{{"id": "owner/name", "score": 9, "one_line": "why it fits in one sentence"}}, ...]

DATASETS:
{batch}"""

    print(f"  OpenRouter rerank: {len(datasets)} datasets...", end=" ", flush=True)
    try:
        raw = openrouter_call(or_model, prompt, max_tokens=2000)
        items = parse_json_response(raw)
    except Exception as e:
        print(f"\n  Warning: OpenRouter rerank failed: {e}", file=sys.stderr)
        return datasets

    id_to_ds = {d["id"]: d for d in datasets}
    result = []
    for rank, item in enumerate(items, 1):
        ds_id = item.get("id", "")
        if ds_id in id_to_ds:
            ds = dict(id_to_ds[ds_id])
            ds["or_rank"] = rank
            ds["or_score"] = item.get("score", 0)
            ds["or_one_line"] = item.get("one_line", "")
            result.append(ds)

    # append any datasets OpenRouter missed (keep at end)
    ranked_ids = {d["id"] for d in result}
    for ds in datasets:
        if ds["id"] not in ranked_ids:
            ds["or_rank"] = 999
            ds["or_score"] = 0
            result.append(ds)

    result.sort(key=lambda x: x.get("or_rank", 999))
    print(f"reranked {len(result)}")
    return result


# ── Stage 3.5: Fetch full cards for datasets Haiku will investigate ──────────

async def fetch_one_card(session, ds: dict, token: str) -> None:
    """Fetch README card for one dataset, mutates ds in place."""
    if ds.get("card_text"):
        return
    url = f"https://huggingface.co/datasets/{ds['id']}/resolve/main/README.md"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = await session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            ds["card_text"] = resp.text[:5000]
            ds["corpus_text"] = ds.get("corpus_text", "") + " " + ds["card_text"]
    except Exception:
        pass


async def fetch_cards_async(datasets: list[dict]) -> list[dict]:
    import httpx
    token = os.getenv("HF_TOKEN", "")
    print(f"  Fetching README cards for {len(datasets)} candidates (async)...", end=" ", flush=True)
    async with httpx.AsyncClient() as session:
        await asyncio.gather(*[fetch_one_card(session, ds, token) for ds in datasets])
    fetched = sum(1 for ds in datasets if ds.get("card_text"))
    print(f"fetched {fetched} new cards")
    return datasets


def fetch_cards_for_candidates(datasets: list[dict]) -> list[dict]:
    return asyncio.run(fetch_cards_async(datasets))


# ── Stage 4: Haiku step 1 — select which to question, generate questions ──────

def haiku_generate_questions(datasets: list[dict], goal: str, bm25_scores: dict) -> dict:
    """
    Haiku reviews the ranked candidates. It decides:
    - Which datasets to investigate further (not all — be selective)
    - What specific questions to ask about each chosen dataset

    Returns: {dataset_id: [question1, question2, ...]}
    """
    # Build compact view with BM25 + OR scores for Haiku to reason about
    batch = "\n\n".join(
        f"[{i+1}] ID: {d['id']}\n"
        f"  BM25: {bm25_scores.get(d['id'], 0):.2f} | OR_rank: {d.get('or_rank','?')} | OR_score: {d.get('or_score','?')}\n"
        f"  {d.get('or_summary') or compact_dataset(d)}"
        for i, d in enumerate(datasets)
    )

    prompt = f"""You are a focused ML dataset researcher. Your goal: find the BEST datasets for:

GOAL: {goal}

You have {len(datasets)} candidates ranked by BM25 and OpenRouter scores.
Token budget is limited — be selective.

Your tasks:
1. Choose ONLY the datasets worth investigating further (those where key info is missing or ambiguous). Skip datasets that are clearly good or clearly bad based on available info.
2. For each chosen dataset, write 1-3 targeted questions that would confirm or deny its fit.
   Questions must be specific and answerable from dataset metadata/card.

Return ONLY valid JSON:
{{
  "selected": [
    {{"id": "owner/name", "questions": ["Q1?", "Q2?"]}},
    ...
  ],
  "reasoning": "brief note on your selection strategy"
}}"""

    print(f"  Haiku step 1: reviewing {len(datasets)} candidates...", end=" ", flush=True)
    try:
        raw = openrouter_call(HAIKU_MODEL, prompt, max_tokens=1500)
        parsed = parse_json_response(raw)
        selected = parsed.get("selected", [])
        reasoning = parsed.get("reasoning", "")
        print(f"chose {len(selected)} to question. ({reasoning[:80]})")
        return {item["id"]: item["questions"] for item in selected if "id" in item and "questions" in item}
    except Exception as e:
        print(f"\n  Warning: Haiku step 1 failed: {e}", file=sys.stderr)
        return {}


# ── Stage 5: OpenRouter answers Haiku's questions ─────────────────────────────

def openrouter_answer_questions(questions_map: dict, datasets: list[dict], or_model: str) -> dict:
    """
    For each dataset Haiku selected, answer its questions using available metadata.
    Returns {dataset_id: {question: answer}}
    """
    if not questions_map:
        return {}

    id_to_ds = {d["id"]: d for d in datasets}
    answers = {}

    # Batch all questions in one call to save tokens
    items = []
    for ds_id, qs in questions_map.items():
        ds = id_to_ds.get(ds_id)
        if not ds:
            continue
        context = (
            f"ID: {ds_id}\n"
            f"Description: {(ds.get('description') or '')[:600]}\n"
            f"Summary: {ds.get('or_summary','')}\n"
            f"Tags: {', '.join((ds.get('tags') or [])[:15])}\n"
            f"Card: {(ds.get('card_text') or '')[:400]}"
        )
        items.append({"id": ds_id, "context": context, "questions": qs})

    prompt = f"""Answer these questions about ML datasets. Be LACONIC — one sentence per answer max.
If unknown, say "unknown".

Return ONLY valid JSON:
[
  {{
    "id": "owner/name",
    "answers": {{"Q1?": "answer", "Q2?": "answer"}}
  }},
  ...
]

DATASETS AND QUESTIONS:
{json.dumps(items, ensure_ascii=False, indent=2)}"""

    print(f"  OpenRouter answers: {len(items)} datasets × questions...", end=" ", flush=True)
    try:
        raw = openrouter_call(or_model, prompt, max_tokens=2000)
        parsed = parse_json_response(raw)
        for item in parsed:
            ds_id = item.get("id", "")
            if ds_id:
                answers[ds_id] = item.get("answers", {})
        print(f"got answers for {len(answers)}")
    except Exception as e:
        print(f"\n  Warning: OpenRouter Q&A failed: {e}", file=sys.stderr)

    return answers


# ── Stage 6: Haiku step 2 — final selection ───────────────────────────────────

def haiku_final_selection(datasets: list[dict], goal: str, answers: dict, bm25_scores: dict) -> list[dict]:
    """
    Haiku reads all information including Q&A answers and makes final narrow selection.
    It decides the top-N itself — but must be conservative (small N).
    """
    id_to_ds = {d["id"]: d for d in datasets}

    # Build rich view per dataset
    entries = []
    for i, d in enumerate(datasets, 1):
        ds_answers = answers.get(d["id"], {})
        qa_text = ""
        if ds_answers:
            qa_text = "\n  Q&A: " + " | ".join(f"{q} → {a}" for q, a in ds_answers.items())
        entries.append(
            f"[{i}] ID: {d['id']}\n"
            f"  BM25: {bm25_scores.get(d['id'],0):.2f} | OR_rank: {d.get('or_rank','?')} | OR_score: {d.get('or_score','?')}\n"
            f"  {d.get('or_summary') or compact_dataset(d)}"
            f"{qa_text}"
        )

    batch = "\n\n".join(entries)

    prompt = f"""You are a precise ML dataset curator. Final selection task.

GOAL: {goal}

Review all candidates below with their BM25 scores, OpenRouter rankings, and Q&A answers.
Select ONLY the genuinely best datasets — be conservative, narrow bottleneck.
Quality over quantity. If fewer than 3 truly fit, return fewer.

For each selected dataset provide:
- relevance_score (1-10)
- reason: specific explanation mentioning modality, annotations, size, license
- needs_verification: true if key info was "unknown" in Q&A

Return ONLY valid JSON array sorted best-first:
[
  {{
    "rank": 1,
    "id": "owner/name",
    "relevance_score": 9,
    "reason": "...",
    "needs_verification": false
  }},
  ...
]

CANDIDATES:
{batch}"""

    print(f"  Haiku step 2: final selection from {len(datasets)} candidates...", end=" ", flush=True)
    try:
        raw = openrouter_call(HAIKU_MODEL, prompt, max_tokens=2000)
        items = parse_json_response(raw)
    except Exception as e:
        print(f"\n  Warning: Haiku final selection failed: {e}", file=sys.stderr)
        return datasets[:3]

    result = []
    for item in items:
        ds_id = item.get("id", "")
        if ds_id in id_to_ds:
            ds = dict(id_to_ds[ds_id])
            ds["llm_rank"] = item.get("rank", 0)
            ds["llm_reason"] = item.get("reason", "")
            ds["llm_relevance_score"] = item.get("relevance_score", 0)
            ds["needs_verification"] = item.get("needs_verification", False)
            result.append(ds)

    result.sort(key=lambda x: x.get("llm_relevance_score", 0), reverse=True)
    print(f"selected {len(result)}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-stage dataset filtering pipeline")
    parser.add_argument("--input", required=True, help="Input JSON from hf_bulk_search.py")
    parser.add_argument("--goal", required=True, help="Natural language goal description")
    parser.add_argument("--keywords", default="", help="Comma-separated keywords for BM25")
    parser.add_argument("--queries", default="", help="Comma-separated queries for BM25")
    parser.add_argument("--bm25-top", type=int, default=60,
                        help="Keep top-N after BM25 (agent sets this, default 60)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--exclude-ids", default="",
                        help="Comma-separated IDs to exclude before pipeline")
    parser.add_argument("--exclude-ids-file", default="",
                        help="Path to file with IDs to exclude, one per line (e.g. data/seen_ids.txt)")
    parser.add_argument("--rejected-ids-file", default="data/rejected_ids.txt",
                        help="File to persist rejected IDs across waves (default: data/rejected_ids.txt)")
    args = parser.parse_args()

    or_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-30b-a3b")

    with open(args.input, encoding="utf-8") as f:
        datasets = json.load(f)

    # ── Global deduplication ──
    rejected_ids = load_rejected_ids(args.rejected_ids_file)

    exclude_ids = set(e.strip() for e in args.exclude_ids.split(",") if e.strip())
    if args.exclude_ids_file:
        ids_path = Path(args.exclude_ids_file)
        if ids_path.exists():
            exclude_ids |= set(line.strip() for line in ids_path.read_text().splitlines() if line.strip())
    all_exclude = exclude_ids | rejected_ids

    if all_exclude:
        before = len(datasets)
        datasets = [d for d in datasets if d["id"] not in all_exclude]
        print(f"  Pre-filter: excluded {before - len(datasets)} already-seen/rejected IDs ({len(datasets)} remain)")

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    print(f"\n{'='*60}")
    print(f"Multi-stage filtering pipeline")
    print(f"  Input:    {len(datasets)} datasets")
    print(f"  Goal:     {args.goal}")
    print(f"  BM25 top: {args.bm25_top}")
    print(f"  OR model: {or_model}  |  Haiku: {HAIKU_MODEL}")
    print(f"{'='*60}\n")

    # ── Stage 1: BM25 ──
    print("[Stage 1] BM25...")
    datasets, bm25_scores = bm25_filter(datasets, queries, keywords, args.bm25_top)
    new_rejected = set()

    # ── Stage 2: OpenRouter filter + summarize ──
    print("\n[Stage 2] OpenRouter filter + summarize...")
    result = openrouter_filter(datasets, args.goal, or_model)
    if isinstance(result, tuple):
        datasets, stage2_rejected = result
        new_rejected |= stage2_rejected
    else:
        datasets = result

    if not datasets:
        print("  No survivors after OpenRouter filter. Exiting.")
        _save_and_exit([], args.output, rejected_ids | new_rejected, args.rejected_ids_file)
        return

    # ── Stage 3: OpenRouter rerank ──
    print("\n[Stage 3] OpenRouter rerank...")
    datasets = openrouter_rerank(datasets, args.goal, or_model)

    # ── Stage 3.5 + 4 in parallel: fetch cards AND Haiku step 1 ──
    print("\n[Stage 3.5 + 4] Fetching cards & Haiku questions in parallel (async)...")

    async def stages_3_5_and_4():
        import httpx
        token = os.getenv("HF_TOKEN", "")
        async with httpx.AsyncClient() as session:
            cards_task = fetch_cards_async(datasets)
            haiku_task = asyncio.to_thread(haiku_generate_questions, datasets, args.goal, bm25_scores)
            updated_datasets, questions_map = await asyncio.gather(cards_task, haiku_task)
        return updated_datasets, questions_map

    datasets, questions_map = asyncio.run(stages_3_5_and_4())

    # ── Stage 5: OpenRouter answers questions ──
    print("\n[Stage 5] OpenRouter: answer Haiku's questions...")
    answers = openrouter_answer_questions(questions_map, datasets, or_model)

    # ── Stage 6: Haiku step 2 — final selection ──
    print("\n[Stage 6] Haiku: final selection (narrow bottleneck)...")
    final = haiku_final_selection(datasets, args.goal, answers, bm25_scores)

    # Track rejected by Haiku (all datasets not in final)
    final_ids = {d["id"] for d in final}
    haiku_rejected = {d["id"] for d in datasets if d["id"] not in final_ids}
    new_rejected |= haiku_rejected

    _save_and_exit(final, args.output, rejected_ids | new_rejected, args.rejected_ids_file)


def _save_and_exit(datasets: list[dict], output: str, all_rejected: set, rejected_file: str):
    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    # Persist rejected IDs
    save_rejected_ids(rejected_file, all_rejected)

    print(f"\n{'='*60}")
    print(f"Final: {len(datasets)} datasets")
    print(f"Rejected (persisted): {len(all_rejected)} total IDs in {rejected_file}")
    print(f"Saved to: {output}\n")

    for i, ds in enumerate(datasets, 1):
        score = ds.get("llm_relevance_score", "?")
        verify = " ⚠️ verify" if ds.get("needs_verification") else ""
        print(f"{i:>2}. [{score}/10]{verify} {ds['id']}")
        print(f"    {ds['url']}")
        print(f"    {ds.get('llm_reason', ds.get('or_one_line', ''))}")
        print()


if __name__ == "__main__":
    main()
