"""Concurrent version of predict_tags.py: same behavior, but scores multiple video ids in
parallel worker threads instead of one at a time. Each call still goes through the same
retry logic; the thread pool just lets multiple in-flight requests overlap.

Supports Gemini (Vertex AI), Anthropic Claude (Vertex AI), and xAI Grok (Vertex AI's
OpenAI-compatible endpoint) as providers, all reading the same tag list / instructions
template / JSON schema.

One output file per video id: results/gemini_eval/<model>/<id>.json

Usage:
    # Gemini, default 5 worker threads
    python predict_tags_concurrent.py --all

    # Claude on Vertex, 10 worker threads
    python predict_tags_concurrent.py --all --provider anthropic --model claude-opus-5 --workers 10

    # Grok on Vertex, 10 worker threads
    python predict_tags_concurrent.py --all --provider grok --model xai/grok-4.1-fast-reasoning --workers 10

    # movie-aware prompt variant (adds identified_title to the stored response)
    python predict_tags_concurrent.py --all --instructions instructions-movie-expert.md --with-title

    # ablation: fixed id list (same ids across every model run), separate output dir
    python predict_tags_concurrent.py --ids-file data/ablation_sample_ids.txt \\
        --instructions instructions-movie-expert.md --with-title --results-dir results/ablation_movie_expert
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TEXTS_DIR = ROOT / "extracted_data" / "texts"
TAGS_PATH = ROOT / "data" / "simple_labels.txt"
ANNOTATIONS_PATH = ROOT / "data" / "annotations.txt"
RESULTS_DIR = ROOT / "results" / "gemini_eval"

TAG_LIST_PLACEHOLDER = "{{TAG_LIST}}"
ANTHROPIC_STRUCTURED_OUTPUT_BETA = "structured-outputs-2025-12-15"


def load_tags() -> list[str]:
    return [line.strip() for line in TAGS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_instructions(template_path: Path, tags: list[str]) -> str:
    template = template_path.read_text(encoding="utf-8")
    tag_list_md = "\n".join(f"- {tag}" for tag in tags)
    return template.replace(TAG_LIST_PLACEHOLDER, tag_list_md)


def build_schema(tags: list[str], with_title: bool) -> dict:
    tag_scores_schema = {
        "type": "object",
        "properties": {tag: {"type": "number", "minimum": 0.0, "maximum": 1.0} for tag in tags},
        "required": tags,
        "additionalProperties": False,
    }
    if not with_title:
        return tag_scores_schema
    return {
        "type": "object",
        "properties": {
            "identified_title": {"type": ["string", "null"]},
            "tag_scores": tag_scores_schema,
        },
        "required": ["identified_title", "tag_scores"],
        "additionalProperties": False,
    }


def all_annotated_ids() -> list[str]:
    ids = []
    seen = set()
    for line in ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vid = line.split(":", 1)[0].strip()
        if vid in seen:
            continue
        seen.add(vid)
        if (TEXTS_DIR / f"{vid}.txt").exists():
            ids.append(vid)
    return ids


# --- Gemini (Vertex AI) ---------------------------------------------------
# Vertex AI returns 429 (quota) and transient 5xx errors under concurrent load far more often
# than under a serial run, so give it more attempts and a longer max backoff than the serial script.
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=2, max=90),
    stop=stop_after_attempt(8),
)
def call_gemini(client, model: str, system_instruction: str, schema: dict, transcript: str) -> dict:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=transcript,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
        ),
    )
    return json.loads(response.text)


def make_gemini_client():
    from google import genai

    return genai.Client()


# --- Anthropic Claude (Vertex AI) -----------------------------------------
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=2, max=90),
    stop=stop_after_attempt(8),
)
def call_anthropic(client, model: str, system_instruction: str, schema: dict, transcript: str, max_tokens: int) -> dict:
    message = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_instruction,
        messages=[{"role": "user", "content": transcript}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
        betas=[ANTHROPIC_STRUCTURED_OUTPUT_BETA],
        temperature=0,
    )
    text_block = next(block for block in message.content if block.type == "text")
    return json.loads(text_block.text)


def make_anthropic_client(region: str):
    from anthropic import AnthropicVertex

    return AnthropicVertex(project_id=os.environ["GOOGLE_CLOUD_PROJECT"], region=region)


# --- xAI Grok (Vertex AI OpenAI-compatible endpoint) -----------------------
# Vertex's OpenAI-compatible endpoint authenticates with a plain bearer access token (no
# separate API key), which expires ~1hr. We fetch/refresh it lazily and pass it fresh on
# every call via extra_headers rather than baking it into the client at construction time,
# since a single client is shared across all worker threads for the life of a long batch run.
_token_lock = threading.Lock()
_credentials = None


def get_access_token() -> str:
    global _credentials
    import google.auth
    import google.auth.transport.requests

    with _token_lock:
        if _credentials is None:
            _credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not _credentials.valid:
            _credentials.refresh(google.auth.transport.requests.Request())
        return _credentials.token


class ContentFilteredError(Exception):
    """Model refused to answer (e.g. its own safety filter tripped) - retrying won't help."""


@retry(
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ContentFilteredError),
    wait=wait_exponential(multiplier=2, min=2, max=90),
    stop=stop_after_attempt(8),
)
def call_grok(client, model: str, system_instruction: str, schema: dict, transcript: str, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "tag_scores", "schema": schema},
        },
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": transcript},
        ],
        extra_headers={"Authorization": f"Bearer {get_access_token()}"},
    )
    choice = response.choices[0]
    if choice.message.content is None:
        raise ContentFilteredError(f"empty content, finish_reason={choice.finish_reason}")
    return json.loads(choice.message.content)


def make_grok_client(region: str):
    from openai import OpenAI

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    # the "global" location is served from the bare host, with no region prefix
    host = "https://aiplatform.googleapis.com" if region == "global" else f"https://{region}-aiplatform.googleapis.com"
    base_url = f"{host}/v1/projects/{project}/locations/{region}/endpoints/openapi"
    return OpenAI(api_key=get_access_token(), base_url=base_url)


# --- provider dispatch ------------------------------------------------------
def make_client(provider: str, anthropic_region: str, grok_region: str):
    if provider == "gemini":
        return make_gemini_client()
    if provider == "anthropic":
        return make_anthropic_client(anthropic_region)
    if provider == "grok":
        return make_grok_client(grok_region)
    raise ValueError(f"unknown provider: {provider}")


def call_model(provider: str, client, model: str, system_instruction: str, schema: dict, transcript: str, max_tokens: int) -> dict:
    if provider == "gemini":
        return call_gemini(client, model, system_instruction, schema, transcript)
    if provider == "anthropic":
        return call_anthropic(client, model, system_instruction, schema, transcript, max_tokens)
    if provider == "grok":
        return call_grok(client, model, system_instruction, schema, transcript, max_tokens)
    raise ValueError(f"unknown provider: {provider}")


def score_one(provider, client, model, system_instruction, schema, vid, instructions_arg, out_dir, max_tokens) -> tuple[str, Exception | None]:
    out_path = out_dir / f"{vid}.json"
    transcript = (TEXTS_DIR / f"{vid}.txt").read_text(encoding="utf-8")
    try:
        response = call_model(provider, client, model, system_instruction, schema, transcript, max_tokens)
    except ContentFilteredError as exc:
        record = {"id": vid, "model": model, "instructions": instructions_arg, "filtered": True, "reason": str(exc)}
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return vid, None
    except Exception as exc:  # noqa: BLE001 - reported by the caller, one bad clip shouldn't kill the run
        return vid, exc
    record = {"id": vid, "model": model, "instructions": instructions_arg, "filtered": False, **response}
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return vid, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["gemini", "anthropic", "grok"], default="gemini")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--anthropic-region", default="global", help="Vertex region for Claude models (anthropic provider only)")
    parser.add_argument("--grok-region", default="global", help="Vertex region for Grok models (grok provider only)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max output tokens (anthropic/grok providers only)")
    parser.add_argument("--instructions", default="instructions.md", help="prompt template, relative to repo root")
    parser.add_argument("--with-title", action="store_true", help="use the {identified_title, tag_scores} schema")
    parser.add_argument("--ids", nargs="+", help="one or more video ids to score")
    parser.add_argument("--ids-file", help="path to a text file of ids (one per line), relative to repo root")
    parser.add_argument("--all", action="store_true", help="score every id in data/annotations.txt")
    parser.add_argument("--limit", type=int, default=None, help="cap number of ids in --all mode")
    parser.add_argument("--overwrite", action="store_true", help="re-score ids that already have a stored response")
    parser.add_argument("--workers", type=int, default=5, help="number of concurrent worker threads (default 5)")
    parser.add_argument("--results-dir", default="results/gemini_eval", help="output root, relative to repo root; one subfolder per model is created under it")
    args = parser.parse_args()

    if not args.ids and not args.ids_file and not args.all:
        parser.error("pass --ids ID [ID ...], --ids-file PATH, or --all for every annotated id")

    load_dotenv(ROOT / ".env")

    tags = load_tags()
    system_instruction = render_instructions(ROOT / args.instructions, tags)
    schema = build_schema(tags, args.with_title)
    client = make_client(args.provider, args.anthropic_region, args.grok_region)

    if args.ids:
        ids = args.ids
    elif args.ids_file:
        ids = [line.strip() for line in (ROOT / args.ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        ids = all_annotated_ids()
    if args.limit:
        ids = ids[: args.limit]

    out_dir = ROOT / args.results_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        ids = [vid for vid in ids if not (out_dir / f"{vid}.json").exists()]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                score_one, args.provider, client, args.model, system_instruction, schema, vid, args.instructions, out_dir, args.max_tokens
            ): vid
            for vid in ids
        }
        with tqdm(total=len(futures), desc=f"scoring with {args.model} ({args.workers} workers)") as pbar:
            for future in as_completed(futures):
                vid, error = future.result()
                if error is not None:
                    tqdm.write(f"[{vid}] failed: {error}")
                pbar.update(1)


if __name__ == "__main__":
    main()
