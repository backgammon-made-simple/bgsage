"""Aggregate per-game PR stats across all .xg files in a folder.

For each .xg file, parses turns via ``bgsage.xg_compare.parse_xg_game``,
applies XG-style decision filters, then sums errors and decisions across
BOTH players (these are Sage-vs-Sage games).

Reports per-game total PR plus the across-games mean / std dev / SEM, and
the aggregate PR computed from summed errors and summed decisions.

PR = sum(equity errors) / decision count * 500.

Usage:

    python bgsage/scripts/aggregate_xg_pr.py [folder] [--pattern '*.xg']

The default folder is ``logs/sage_vs_sage`` under the parent project root,
matching where ``run_sage_vs_sage_games.py`` writes its .txt files. After
producing those files, run XG's Batch Analyze with "Save Games after analyze"
checked; XG writes one .xg file per .txt next to it. Then run this script.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# bgsage path setup (matches the other bgsage/scripts/* runners)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_BGSAGE_PYTHON = _PROJECT_ROOT / "bgsage" / "python"
_BUILD_DIR = _PROJECT_ROOT / "build"

for _p in (_BGSAGE_PYTHON, _BUILD_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))


from bgsage.xg_compare import compute_game_pr_stats, parse_xg_game  # noqa: E402

_DEFAULT_DIR = _PROJECT_ROOT / "logs" / "sage_vs_sage"


def _seed_sort_key(p: Path) -> tuple[int, str]:
    """Sort by trailing integer in the stem (``seed_<N>``), else by name."""
    parts = p.stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), p.name)
    return (10**9, p.name)


def _stats_for_xg(xg_path: Path) -> dict:
    """Parse one .xg file and return per-player error totals + decision counts."""
    turns = parse_xg_game(xg_path.read_bytes())
    return compute_game_pr_stats(turns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=_DEFAULT_DIR,
        help=f"Folder to scan for .xg files (default: {_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="*.xg",
        help="Glob pattern within folder (default: *.xg)",
    )
    args = parser.parse_args()

    xg_files = sorted(args.folder.glob(args.pattern), key=_seed_sort_key)
    if not xg_files:
        print(f"No .xg files found in {args.folder} matching {args.pattern!r}")
        return

    print(f"Scanning {len(xg_files)} .xg files in {args.folder}\n")
    header = (
        f"{'file':<16} "
        f"{'P1 err':>9} {'P1 dec':>7} "
        f"{'P2 err':>9} {'P2 dec':>7} "
        f"{'tot err':>9} {'tot dec':>7} "
        f"{'PR':>7}"
    )
    print(header)
    print("-" * len(header))

    per_game_pr: list[float] = []
    sum_err = 0.0
    sum_dec = 0
    for xg_path in xg_files:
        s = _stats_for_xg(xg_path)
        per_game_pr.append(s["pr"])
        sum_err += s["total_err"]
        sum_dec += s["total_dec"]
        pr_text = f"{s['pr']:.2f}" if not math.isnan(s["pr"]) else "  nan"
        print(
            f"{xg_path.name:<16} "
            f"{s['user_err']:>9.4f} {s['user_dec']:>7} "
            f"{s['bot_err']:>9.4f} {s['bot_dec']:>7} "
            f"{s['total_err']:>9.4f} {s['total_dec']:>7} "
            f"{pr_text:>7}"
        )

    n = len(per_game_pr)
    valid_pr = [p for p in per_game_pr if not math.isnan(p)]
    mean_pr = statistics.mean(valid_pr) if valid_pr else float("nan")
    std_pr = statistics.stdev(valid_pr) if len(valid_pr) > 1 else 0.0
    sem_pr = std_pr / math.sqrt(len(valid_pr)) if valid_pr else 0.0
    agg_pr = (sum_err / sum_dec * 500.0) if sum_dec > 0 else float("nan")

    print("-" * len(header))
    print()
    print(f"Games:                {n}")
    print(f"Per-game PR mean:     {mean_pr:.3f}")
    print(f"Per-game PR std dev:  {std_pr:.3f}")
    print(f"Per-game PR SEM:      {sem_pr:.3f}")
    print()
    print(f"Total errors summed:  {sum_err:.4f}")
    print(f"Total decisions:      {sum_dec}")
    print(f"Aggregate PR:         {agg_pr:.3f}")


if __name__ == "__main__":
    main()
