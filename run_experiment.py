"""
run_experiment.py — launch a full experiment suite sequentially.

Usage:
    uv run run_experiment.py                         # run all jobs for experiment
    uv run run_experiment.py --exp experiment6       # different experiment dir
    uv run run_experiment.py --dry-run               # print commands, don't run
    uv run run_experiment.py --only text,frame       # run a named subset
    uv run run_experiment.py --skip-done             # skip jobs whose save-dir already has final_metrics.json
"""

import argparse
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Job definitions
# Each entry: (label, script, extra_args)
#   label      — short name, also used as the subdirectory under results/<exp>/
#   script     — the Python file to run via `uv run`
#   extra_args — list of additional CLI arguments
# ---------------------------------------------------------------------------
JOBS = [
    ("text",          "train_text.py",        []),
    ("frame",         "train_frame.py",       []),
    ("audio",         "train_audio.py",       []),
    ("mm_crossattn",  "train_multimodal.py",  ["--fusion", "cross_attn"]),
    ("mm_concat",     "train_multimodal.py",  ["--fusion", "concat"]),
    ("mm_add",        "train_multimodal.py",  ["--fusion", "add"]),
    ("mm_film",       "train_multimodal.py",  ["--fusion", "film"]),
    ("mm_film_pw5",   "train_multimodal.py",  ["--fusion", "film", "--pos-weight-clip", "5"]),
    ("mm_film_pw20",  "train_multimodal.py",  ["--fusion", "film", "--pos-weight-clip", "20"]),
    ("mm_film_focal", "train_multimodal.py",  ["--fusion", "film", "--loss", "focal", "--focal-alpha", "0.6"]),
]


def build_cmd(script: str, save_dir: Path, extra_args: list[str]) -> list[str]:
    return ["uv", "run", script, "--save-dir", str(save_dir)] + extra_args


def is_done(save_dir: Path) -> bool:
    return (save_dir / "final_metrics.json").exists()


def run_job(label: str, cmd: list[str], save_dir: Path, log_path: Path) -> bool:
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  JOB: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_path}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()

    elapsed = timedelta(seconds=int(time.time() - t0))
    if proc.returncode == 0:
        print(f"  [OK]  {label} finished in {elapsed}")
        return True
    else:
        print(f"  [FAIL] {label} exited with code {proc.returncode} after {elapsed}")
        return False


def run_analyze(exp_dir: Path) -> None:
    cmd = ["uv", "run", "analyze_results.py", "--results-dir", str(exp_dir)]
    print(f"\n{'='*60}")
    print(f"  ANALYZE: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run a full experiment suite.")
    parser.add_argument("--exp", required=True, help="Experiment folder name under results/")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--skip-done", action="store_true", help="Skip jobs that already have final_metrics.json")
    parser.add_argument("--only", default="", help="Comma-separated subset of job labels to run (e.g. text,frame)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip the analyze_results.py step at the end")
    args = parser.parse_args()

    exp_dir = Path("results") / args.exp
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    jobs_to_run = [(label, script, extra) for label, script, extra in JOBS if not only or label in only]

    if args.dry_run:
        print(f"Dry run — experiment: {exp_dir}\n")
        for label, script, extra in jobs_to_run:
            save_dir = exp_dir / label
            cmd = build_cmd(script, save_dir, extra)
            status = " [DONE, would skip]" if args.skip_done and is_done(save_dir) else ""
            print(f"  {label:20s}  {' '.join(cmd)}{status}")
        print(f"\n  analyze_results.py --results-dir {exp_dir}")
        return

    results: dict[str, str] = {}
    total_t0 = time.time()

    for label, script, extra in jobs_to_run:
        save_dir = exp_dir / label
        if args.skip_done and is_done(save_dir):
            print(f"\n  [SKIP] {label} — final_metrics.json already exists")
            results[label] = "skipped"
            continue

        log_path = save_dir / "train.log"
        cmd = build_cmd(script, save_dir, extra)
        ok = run_job(label, cmd, save_dir, log_path)
        results[label] = "ok" if ok else "FAILED"

    # summary
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY  ({exp_dir})")
    print(f"{'='*60}")
    for label, status in results.items():
        mark = "OK" if status == "ok" else ("--" if status == "skipped" else "FAIL")
        print(f"  [{mark:4s}]  {label}")

    total_elapsed = timedelta(seconds=int(time.time() - total_t0))
    print(f"\n  Total elapsed: {total_elapsed}")

    failed = [l for l, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\n  Failed jobs: {', '.join(failed)}")

    if not args.no_analyze:
        run_analyze(exp_dir)


if __name__ == "__main__":
    main()
