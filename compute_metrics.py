"""Compute classification metrics for every model scored by predict_tags(_concurrent).py,
plus the movie-expert-prompt ablation, and write it all up as report.md.

Reads:
    data/annotations.txt                          ground truth tags per id
    data/simple_labels.txt                        the fixed 91-tag label space
    data/ablation_sample_ids.txt                  the fixed 98-id ablation sample
    results/gemini_eval/<model>/<id>.json         neutral-prompt tag scores per model
    results/ablation_movie_expert/<model>/<id>.json   movie-expert-prompt scores (98-id sample)

Writes:
    report.md
    results/report_assets/title_agreement_histogram.png

Usage:
    python compute_metrics.py
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
ANNOTATIONS_PATH = ROOT / "data" / "annotations.txt"
TAGS_PATH = ROOT / "data" / "simple_labels.txt"
ABLATION_IDS_PATH = ROOT / "data" / "ablation_sample_ids.txt"
NEUTRAL_RESULTS_ROOT = ROOT / "results" / "gemini_eval"
ABLATION_RESULTS_ROOT = ROOT / "results" / "ablation_movie_expert"
ASSETS_DIR = ROOT / "results" / "report_assets"
REPORT_PATH = ROOT / "report.md"

THRESHOLDS = ["max", 0.7, 0.3]
THRESHOLD_LABELS = {"max": "max (top-1)", 0.7: "0.7", 0.3: "0.3"}

MODELS = [
    {
        "name": "Gemini 2.5 Flash",
        "dir": NEUTRAL_RESULTS_ROOT / "gemini-2.5-flash",
        "ablation_dir": ABLATION_RESULTS_ROOT / "gemini-2.5-flash",
        "description": (
            "Google's mid-tier Gemini model on Vertex AI, balancing quality, latency, and cost. "
            "Called via the native Gemini API with `response_json_schema` structured output, so "
            "every call is schema-constrained at the API level rather than relying on prompted "
            "JSON formatting."
        ),
    },
    {
        "name": "Grok 4.1 Fast Reasoning",
        "dir": NEUTRAL_RESULTS_ROOT / "xai" / "grok-4.1-fast-reasoning",
        "ablation_dir": ABLATION_RESULTS_ROOT / "xai" / "grok-4.1-fast-reasoning",
        "description": (
            "xAI's fast, reasoning-tuned model, accessed through Vertex AI's OpenAI-compatible "
            "endpoint. One clip in the main run triggered Grok's own content-safety filter and "
            "returned no output at all (a deterministic refusal, not a transient error) - it's "
            "excluded from this model's metrics rather than counted as a miss, so Grok's coverage "
            "is 982/983 clips instead of 983/983."
        ),
    },
    {
        "name": "GPT-OSS 120B",
        "dir": NEUTRAL_RESULTS_ROOT / "openai" / "gpt-oss-120b-maas",
        "ablation_dir": ABLATION_RESULTS_ROOT / "openai" / "gpt-oss-120b-maas",
        "description": (
            "OpenAI's open-weight 120B model, hosted directly on Vertex AI as a managed open model "
            "(MaaS) rather than through OpenAI's own API - billed through GCP, no OpenAI account "
            "needed. Called via the same OpenAI-compatible endpoint as Grok."
        ),
    },
    {
        "name": "DeepSeek V3.2",
        "dir": NEUTRAL_RESULTS_ROOT / "deepseek-ai" / "deepseek-v3.2-maas",
        "ablation_dir": ABLATION_RESULTS_ROOT / "deepseek-ai" / "deepseek-v3.2-maas",
        "description": (
            "DeepSeek's large open-weight model, hosted on Vertex AI as a managed open model, "
            "reachable only via the endpoint's `global` location rather than a specific region."
        ),
    },
    {
        "name": "Qwen3-Next-80B Instruct",
        "dir": NEUTRAL_RESULTS_ROOT / "qwen" / "qwen3-next-80b-a3b-instruct-maas",
        "ablation_dir": ABLATION_RESULTS_ROOT / "qwen" / "qwen3-next-80b-a3b-instruct-maas",
        "description": (
            "Alibaba's Qwen3-Next 80B model, standard instruction-tuned (non-reasoning) variant. "
            "The sibling \"Thinking\" (reasoning) variant was tried first but got stuck in a long "
            "repetitive reasoning loop and never produced output even on a trivial prompt, so it "
            "was dropped in favor of this Instruct variant."
        ),
    },
]


def load_tags() -> list[str]:
    return [line.strip() for line in TAGS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ground_truth() -> dict[str, set[str]]:
    gt = {}
    for line in ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vid, rest = line.split(":", 1)
        tags = {t.strip() for t in rest.split(",") if t.strip()}
        gt[vid.strip()] = tags
    return gt


def load_neutral_scores(model_dir: Path, tags: list[str]) -> dict[str, dict[str, float]]:
    """id -> {tag: score}, skipping filtered/refused clips."""
    tag_set = set(tags)
    scores = {}
    if not model_dir.exists():
        return scores
    for path in model_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("filtered"):
            continue
        scores[path.stem] = {k: v for k, v in record.items() if k in tag_set}
    return scores


def load_ablation_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("filtered"):
        return None
    return record


def predicted_set(scores: dict[str, float], threshold) -> set[str]:
    if not scores:
        return set()
    if threshold == "max":
        top_tag = max(scores, key=scores.get)
        return {top_tag}
    return {tag for tag, score in scores.items() if score >= threshold}


def micro_prf(y_true: dict[str, set[str]], y_pred: dict[str, set[str]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for vid in y_true:
        true_set = y_true[vid]
        pred_set = y_pred[vid]
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def label_accuracy(y_true: dict[str, set[str]], y_pred: dict[str, set[str]], tags: list[str]) -> float:
    correct = 0
    total = 0
    for vid in y_true:
        true_set = y_true[vid]
        pred_set = y_pred[vid]
        for tag in tags:
            total += 1
            if (tag in true_set) == (tag in pred_set):
                correct += 1
    return correct / total if total else 0.0


def main():
    tags = load_tags()
    ground_truth = load_ground_truth()
    ablation_ids = [line.strip() for line in ABLATION_IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # --- per-model neutral-prompt metrics ---------------------------------
    for model in MODELS:
        neutral_scores = load_neutral_scores(model["dir"], tags)
        coverage_ids = [vid for vid in neutral_scores if vid in ground_truth]
        model["coverage"] = len(coverage_ids)
        model["total_unique_clips"] = len(ground_truth)
        rows = []
        for threshold in THRESHOLDS:
            y_true = {vid: ground_truth[vid] for vid in coverage_ids}
            y_pred = {vid: predicted_set(neutral_scores[vid], threshold) for vid in coverage_ids}
            precision, recall, f1 = micro_prf(y_true, y_pred)
            accuracy = label_accuracy(y_true, y_pred, tags)
            rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy})
        model["metric_rows"] = rows

    # --- per-model ablation: neutral vs movie-expert on the 98-id sample --
    for model in MODELS:
        neutral_scores = load_neutral_scores(model["dir"], tags)
        neutral_deltas = []
        neutral_vals = []
        expert_vals = []
        for vid in ablation_ids:
            neutral = neutral_scores.get(vid)
            expert_record = load_ablation_record(model["ablation_dir"] / f"{vid}.json")
            if neutral is None or expert_record is None:
                continue
            expert_tag_scores = expert_record.get("tag_scores", {})
            for tag in tags:
                n = neutral.get(tag)
                e = expert_tag_scores.get(tag)
                if n is None or e is None:
                    continue
                neutral_vals.append(n)
                expert_vals.append(e)
                neutral_deltas.append(e - n)

        model["ablation_n_pairs"] = len(neutral_deltas)
        if neutral_deltas:
            model["ablation_mean_abs_diff"] = statistics.fmean(abs(d) for d in neutral_deltas)
            model["ablation_mean_signed_diff"] = statistics.fmean(neutral_deltas)
            corr, _ = pearsonr(neutral_vals, neutral_deltas)
            model["ablation_corr_neutral_vs_delta"] = corr
        else:
            model["ablation_mean_abs_diff"] = None
            model["ablation_mean_signed_diff"] = None
            model["ablation_corr_neutral_vs_delta"] = None

    # --- title identification + cross-model agreement ---------------------
    def normalize_title(title: str) -> str:
        return " ".join(title.lower().split())

    per_clip_titles = {}  # vid -> {model_name: raw_title}
    for vid in ablation_ids:
        per_clip_titles[vid] = {}
        for model in MODELS:
            record = load_ablation_record(model["ablation_dir"] / f"{vid}.json")
            title = record.get("identified_title") if record else None
            if title:
                per_clip_titles[vid][model["name"]] = title

    identification_counts = {model["name"]: 0 for model in MODELS}
    for vid, titles in per_clip_titles.items():
        for model_name in titles:
            identification_counts[model_name] += 1

    agreement_rows = []
    agreement_histogram = {n: 0 for n in range(len(MODELS) + 1)}
    for vid, titles in per_clip_titles.items():
        if not titles:
            agreement_histogram[0] += 1
            continue
        groups: dict[str, list[str]] = {}
        for model_name, title in titles.items():
            key = normalize_title(title)
            groups.setdefault(key, []).append(model_name)
        best_key, best_models = max(groups.items(), key=lambda kv: len(kv[1]))
        agreement_histogram[len(best_models)] += 1
        if len(best_models) >= 2:
            display_title = titles[best_models[0]]
            agreement_rows.append(
                {"id": vid, "title": display_title, "agree_count": len(best_models), "models": best_models}
            )
    agreement_rows.sort(key=lambda r: r["agree_count"], reverse=True)

    # histogram chart
    bins = list(range(len(MODELS) + 1))
    counts = [agreement_histogram[n] for n in bins]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(bins, counts, color="#2a78d6", width=0.6, zorder=3)
    for rect, count in zip(bars, counts):
        if count:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3, str(count),
                     ha="center", va="bottom", color="#0b0b0b", fontsize=9)
    ax.set_xticks(bins)
    ax.set_xlabel("number of models agreeing on the same title", color="#52514e")
    ax.set_ylabel("number of clips", color="#52514e")
    ax.set_title("Cross-model title agreement across the 98-clip sample", color="#0b0b0b", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    fig.tight_layout()
    chart_path = ASSETS_DIR / "title_agreement_histogram.png"
    fig.savefig(chart_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    write_report(tags, ground_truth, ablation_ids, identification_counts, agreement_rows, chart_path)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {chart_path}")


def write_report(tags, ground_truth, ablation_ids, identification_counts, agreement_rows, chart_path):
    lines = []
    lines.append("# Movie-tag prediction: model comparison report")
    lines.append("")
    lines.append(
        f"Ground truth: {len(ground_truth)} unique annotated clips, {len(tags)} valid tags "
        f"(`data/simple_labels.txt`). Ablation sample: {len(ablation_ids)} clips "
        f"(`data/ablation_sample_ids.txt`, fixed seed, identical across all models)."
    )
    lines.append("")
    lines.append(
        "**Methodology note:** Precision/recall/F1 are micro-averaged (pooled TP/FP/FN across all "
        "clips and tags, not averaged per-tag) - this is the standard choice for multi-label "
        "problems with many rare tags, where per-tag macro-averaging is dominated by noise on "
        "tags that appear only a handful of times. Accuracy is per-label accuracy (a tag counts "
        "as \"correct\" for a clip if its predicted presence/absence matches ground truth), "
        "computed across all clip x tag pairs - this is dominated by true negatives (most tags "
        "don't apply to most clips), so high accuracy numbers here are expected and not by "
        "themselves evidence of a good model; read them alongside F1."
    )
    lines.append("")
    lines.append(
        "Three thresholding schemes convert continuous 0-1 tag-likelihood scores into a predicted "
        "tag set: **max** takes only the single highest-scoring tag per clip (top-1); **0.7** and "
        "**0.3** take every tag scoring at or above that cutoff, regardless of count."
    )
    lines.append("")
    lines.append("## Model comparison (neutral prompt)")
    lines.append("")
    lines.append(
        "Head-to-head comparison of all five models on the full dataset, neutral prompt "
        "(`instructions.md`), one table per threshold. Best F1 per threshold is **bolded**."
    )
    lines.append("")
    for threshold in THRESHOLDS:
        label = THRESHOLD_LABELS[threshold]
        rows_for_threshold = []
        for model in MODELS:
            row = next(r for r in model["metric_rows"] if r["threshold"] == threshold)
            rows_for_threshold.append((model["name"], row))
        best_f1 = max(row["f1"] for _, row in rows_for_threshold)
        lines.append(f"**Threshold: {label}**")
        lines.append("")
        lines.append("| Model | Precision | Recall | F1 | Accuracy |")
        lines.append("|---|---|---|---|---|")
        for name, row in sorted(rows_for_threshold, key=lambda nr: nr[1]["f1"], reverse=True):
            f1_str = f"**{row['f1']:.3f}**" if row["f1"] == best_f1 else f"{row['f1']:.3f}"
            lines.append(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {f1_str} | {row['accuracy']:.3f} |")
        lines.append("")

    lines.append("## Models")
    lines.append("")

    for model in MODELS:
        lines.append(f"### {model['name']}")
        lines.append("")
        lines.append(model["description"])
        lines.append("")
        lines.append(f"Coverage: {model['coverage']}/{model['total_unique_clips']} clips scored.")
        lines.append("")
        lines.append("| Threshold | Precision | Recall | F1 | Accuracy |")
        lines.append("|---|---|---|---|---|")
        for row in model["metric_rows"]:
            label = THRESHOLD_LABELS[row["threshold"]]
            lines.append(
                f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['accuracy']:.3f} |"
            )
        lines.append("")

        lines.append(f"**Movie-expert ablation** (same {len(ablation_ids)}-clip sample, neutral prompt vs. movie-expert prompt):")
        lines.append("")
        if model["ablation_n_pairs"]:
            mean_abs = model["ablation_mean_abs_diff"]
            mean_signed = model["ablation_mean_signed_diff"]
            corr = model["ablation_corr_neutral_vs_delta"]
            direction = "higher" if mean_signed > 0 else "lower" if mean_signed < 0 else "unchanged"
            corr_note = (
                "a positive correlation - tags the model already scored high under the neutral "
                "prompt get pushed higher still under the movie-expert prompt (scores polarize)"
                if corr > 0.15
                else "a negative correlation - already-high scores get pulled down and already-low "
                "scores get pushed up under the movie-expert prompt (scores compress toward the middle)"
                if corr < -0.15
                else "close to no correlation - the shift is roughly independent of the tag's "
                "neutral-prompt score"
            )
            lines.append(f"- Mean absolute score change: **{mean_abs:.3f}** (on a 0-1 scale)")
            lines.append(
                f"- Mean signed score change: **{mean_signed:+.3f}** - scores run {direction} on "
                "average under the movie-expert prompt"
            )
            lines.append(
                f"- Correlation between a tag's neutral-prompt score and its change under the "
                f"movie-expert prompt: **{corr:+.3f}** ({corr_note})"
            )
        else:
            lines.append("- No overlapping scored clips between the two prompt variants for this model.")
        lines.append("")

    lines.append("## Key takeaways")
    lines.append("")
    lines.append("**Movie-title identification rate** (out of the same 98-clip ablation sample):")
    lines.append("")
    lines.append("| Model | Clips with a guessed title |")
    lines.append("|---|---|")
    for model in MODELS:
        count = identification_counts[model["name"]]
        lines.append(f"| {model['name']} | {count}/{len(ablation_ids)} ({count / len(ablation_ids):.0%}) |")
    lines.append("")

    lines.append(
        "**Cross-model agreement** is used as a correctness proxy in place of external lookups: "
        "if independent models converge on the same title for the same clip, that's stronger "
        "evidence of a real, recognizable movie than any single confident-sounding guess (a model "
        "guessing fluently doesn't mean it guessed *correctly* - we have no ground-truth titles to "
        "check against directly, and we did not perform external lookups to verify these guesses)."
    )
    lines.append("")
    rel_chart_path = chart_path.relative_to(ROOT).as_posix()
    lines.append(f"![Cross-model title agreement histogram]({rel_chart_path})")
    lines.append("")

    if agreement_rows:
        lines.append("Clips where 2 or more models agreed on the same title:")
        lines.append("")
        lines.append("| Clip id | Agreed title | # models agreeing | Models |")
        lines.append("|---|---|---|---|")
        for row in agreement_rows:
            models_str = ", ".join(row["models"])
            lines.append(f"| {row['id']} | {row['title']} | {row['agree_count']} | {models_str} |")
        lines.append("")
    else:
        lines.append("No clips had 2+ models agree on the same title.")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
