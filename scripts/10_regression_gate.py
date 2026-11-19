"""Module 12: compare a scored run with the stored baseline; exit 1 on regression.

Run:
  python scripts/10_regression_gate.py --candidate results/<scored>.jsonl
  python scripts/10_regression_gate.py --candidate results/<scored>.jsonl --update-baseline
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.records import load_records  # noqa: E402
from evallab.regression import compare, load_baseline, per_item_scores, save_baseline  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default=str(settings.baseline_path))
    ap.add_argument("--max-drop", type=float, default=0.05)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    records = load_records(Path(args.candidate))
    if args.update_baseline:
        save_baseline(records, Path(args.baseline), meta={"git_sha": git_sha(), "source": args.candidate})
        print(f"baseline updated from {args.candidate}")
        return 0
    if not Path(args.baseline).exists():
        print(f"No baseline at {args.baseline}. Review this run, then re-run with "
              "--update-baseline to adopt it. Gate skipped.")
        return 0
    result = compare(load_baseline(Path(args.baseline)), per_item_scores(records), max_drop=args.max_drop)
    print(result.report())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
