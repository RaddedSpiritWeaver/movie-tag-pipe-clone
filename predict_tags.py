"""Call a Gemini model on Vertex AI to score transcripts against the movietag label set,
and store each model response to disk for later processing (metrics, comparisons, etc).

One output file per video id: results/gemini_eval/<model>/<id>.json

Usage:
    # single test call
    python predict_tags.py --ids xNu0qNfOoGY

    # every id in data/annotations.txt that has a transcript file
    python predict_tags.py --all

    # movie-aware prompt variant (adds identified_title to the stored response)
    python predict_tags.py --all --instructions instructions-movie-expert.md --with-title
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TEXTS_DIR = ROOT / "extracted_data" / "texts"
TAGS_PATH = ROOT / "data" / "simple_labels.txt"
ANNOTATIONS_PATH = ROOT / "data" / "annotations.txt"
RESULTS_DIR = ROOT / "results" / "gemini_eval"

TAG_LIST_PLACEHOLDER = "{{TAG_LIST}}"


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


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
)
def call_model(client: genai.Client, model: str, system_instruction: str, schema: dict, transcript: str) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--instructions", default="instructions.md", help="prompt template, relative to repo root")
    parser.add_argument("--with-title", action="store_true", help="use the {identified_title, tag_scores} schema")
    parser.add_argument("--ids", nargs="+", help="one or more video ids to score")
    parser.add_argument("--all", action="store_true", help="score every id in data/annotations.txt")
    parser.add_argument("--limit", type=int, default=None, help="cap number of ids in --all mode")
    parser.add_argument("--overwrite", action="store_true", help="re-score ids that already have a stored response")
    args = parser.parse_args()

    if not args.ids and not args.all:
        parser.error("pass --ids ID [ID ...] for specific ids, or --all for every annotated id")

    load_dotenv(ROOT / ".env")

    tags = load_tags()
    system_instruction = render_instructions(ROOT / args.instructions, tags)
    schema = build_schema(tags, args.with_title)
    client = genai.Client()

    ids = args.ids if args.ids else all_annotated_ids()
    if args.limit:
        ids = ids[: args.limit]

    out_dir = RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in tqdm(ids, desc=f"scoring with {args.model}"):
        out_path = out_dir / f"{vid}.json"
        if out_path.exists() and not args.overwrite:
            continue

        transcript = (TEXTS_DIR / f"{vid}.txt").read_text(encoding="utf-8")
        try:
            response = call_model(client, args.model, system_instruction, schema, transcript)
        except Exception as exc:  # noqa: BLE001 - log and keep going, one bad clip shouldn't kill the run
            tqdm.write(f"[{vid}] failed: {exc}")
            continue

        record = {"id": vid, "model": args.model, "instructions": args.instructions, **response}
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if len(ids) == 1:
            print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
