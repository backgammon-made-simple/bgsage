#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Generate Paskogammon training and benchmark position data.

Plays cubeless self-play money games from the Paskogammon starting position
(doubles ALLOWED on the opening roll; positive player always on roll) with the
production model at 1-ply, collecting positions that have back-game structure.

Filter — keep a position if ANY of these hold (checked from the on-roll
player's perspective; positive = P1, negative = P2):
  * P1 has >= 4 back checkers  (points 19-24 = P2's home, plus P1 bar at 25)
  * P2 has >= 4 back checkers  (points 1-6  = P1's home, plus P2 bar at 0)
  * P1 holds >= 2 anchors in P2's home board (>=2 P1 checkers on 2+ of points 19-24)
  * P2 holds >= 2 anchors in P1's home board (>=2 P2 checkers on 2+ of points 1-6)

Training and benchmark sets are generated in SEPARATE self-play runs with
disjoint RNG seeds; any benchmark position that also appears in the training
set is dropped, so the two files never share a position (the Paskogammon start
itself passes the filter and would otherwise leak into both).

Positions are stored exactly as encountered (on-roll perspective, one per line
as 26 space-separated ints) — the same convention as generate_backgame_data.py,
so the rollout and NN-eval steps treat every board identically.

Output, in bgsage/data/:
  pasko-train-data
  pasko-benchmark-data

Roll these out next (from the PARENT repo, via Parallelizor):
  python scripts/rollout_pasko_positions.py pasko-train-data --workers 200
  python scripts/rollout_pasko_positions.py pasko-benchmark-data --workers 50

Usage:
  python bgsage/scripts/generate_pasko_data.py
  python bgsage/scripts/generate_pasko_data.py --train-target 90000 --benchmark-target 10000
  python bgsage/scripts/generate_pasko_data.py --train-target 300 --benchmark-target 80 --workers 8  # quick test
"""

import os
import sys
import time
import random
import argparse
import multiprocessing as mp

# --- Path setup (runs in the main process AND every spawned worker) -------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../bgbot/bgsage
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)               # .../bgbot

if sys.platform == 'win32':
    _cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)

# Prefer bgsage/build (kept fresh by the MSVC copy step); the parent build/ may
# be held open by a running backend, so it is only a fallback.
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'python'))

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')

# Paskogammon starting position (positive player on roll). 15 checkers each side.
PASKO_START = [0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,
               5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2, 0]

# Disjoint seed namespaces for the two splits (Python ints are unbounded).
_SPLIT_SEED = {'train': 1 * 10**15, 'benchmark': 2 * 10**15}

# --- Worker-local strategy (set by the pool initializer) ------------------
_strategy = None


def passes_filter(board):
    """True if the position has qualifying Paskogammon back-game structure."""
    # Back checkers: own checkers deep in the opponent's home board, plus the bar.
    p1_back = sum(board[i] for i in range(19, 25) if board[i] > 0) + board[25]
    p2_back = sum(-board[i] for i in range(1, 7) if board[i] < 0) + board[0]
    if p1_back >= 4 or p2_back >= 4:
        return True
    # Anchors: points held with 2+ checkers in the opponent's home board.
    p1_anchors = sum(1 for i in range(19, 25) if board[i] >= 2)
    p2_anchors = sum(1 for i in range(1, 7) if board[i] <= -2)
    return p1_anchors >= 2 or p2_anchors >= 2


def _init_worker(weight_paths, hidden_sizes, n_plies):
    """Pool initializer — one production (stage9) strategy per worker."""
    global _strategy
    import bgbot_cpp
    _strategy = bgbot_cpp.create_multipy_stage9(
        weight_paths, hidden_sizes, n_plies=n_plies)


def play_games_worker(args):
    """Play n_games of cubeless self-play; return the qualifying positions."""
    seed, n_games = args
    import bgbot_cpp

    strategy = _strategy
    rng = random.Random(seed)
    found = set()

    for _ in range(n_games):
        board = list(PASKO_START)
        # Paskogammon opening: doubles ALLOWED (no re-roll), P1 always on roll.
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

        for _ in range(500):  # safety limit against pathological non-terminating play
            if passes_filter(board):
                found.add(tuple(board))

            candidates = bgbot_cpp.possible_moves(board, d1, d2)
            if candidates:
                best_idx = strategy.best_move_index(candidates, board)
                board = list(candidates[best_idx])

            if bgbot_cpp.check_game_over(board) != 0:
                break

            board = list(bgbot_cpp.flip_board(board))
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

    return [list(p) for p in found]


def collect(pool, split, target, n_workers, games_per_cycle, exclude):
    """Collect `target` unique qualifying positions for a split.

    Positions present in `exclude` (a set of tuples) are never added — used to
    keep the benchmark set disjoint from the training set.
    """
    seed_base = _SPLIT_SEED[split]
    collected = set()
    cycle = 0
    t0 = time.time()
    while len(collected) < target:
        # Prime-spread seeds within a large per-split namespace so train and
        # benchmark streams are disjoint and adjacent seeds are uncorrelated.
        seeds = [
            (seed_base + (cycle * n_workers + i) * 1000003, games_per_cycle)
            for i in range(n_workers)
        ]
        for positions in pool.map(play_games_worker, seeds):
            for p in positions:
                t = tuple(p)
                if t not in exclude:
                    collected.add(t)
        cycle += 1
        total_games = cycle * n_workers * games_per_cycle
        elapsed = time.time() - t0
        rate = total_games / elapsed if elapsed > 0 else 0
        print(f'  [{split}] cycle {cycle:3d}: {total_games:>9,} games | '
              f'unique={len(collected):>7,}/{target:,} | '
              f'{elapsed:6.1f}s ({rate:.0f} games/s)', flush=True)
    return collected


def write_positions(filepath, positions):
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(' '.join(str(x) for x in pos) + '\n')
    print(f'  {os.path.basename(filepath)}: {len(positions):,} positions')


def main():
    parser = argparse.ArgumentParser(
        description='Generate Paskogammon train/benchmark positions via cubeless self-play')
    parser.add_argument('--train-target', type=int, default=90000,
                        help='Unique training positions to collect (default: 90000)')
    parser.add_argument('--benchmark-target', type=int, default=10000,
                        help='Unique benchmark positions to collect (default: 10000)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Parallel self-play worker processes (default: 32)')
    parser.add_argument('--games-per-cycle', type=int, default=100,
                        help='Games per worker per cycle (default: 100)')
    parser.add_argument('--plies', type=int, default=1,
                        help='Self-play move-selection ply (default: 1)')
    args = parser.parse_args()

    from bgsage.weights import WeightConfigPair
    w = WeightConfigPair.from_model('stage9')  # production model
    w.validate()
    weight_paths, hidden_sizes = w.weight_args

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Model: production stage9 ({len(weight_paths)} NNs), {args.plies}-ply self-play')
    print(f'Targets: train={args.train_target:,}  benchmark={args.benchmark_target:,}')
    print(f'Workers: {args.workers}, games/worker/cycle: {args.games_per_cycle}')
    print(flush=True)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(weight_paths, hidden_sizes, args.plies)) as pool:
        train = set()
        if args.train_target > 0:
            print('Collecting TRAIN positions...')
            train = collect(pool, 'train', args.train_target, args.workers,
                            args.games_per_cycle, exclude=frozenset())
        benchmark = set()
        if args.benchmark_target > 0:
            note = 'excluding train' if args.train_target > 0 else 'no train set to exclude'
            print(f'Collecting BENCHMARK positions (disjoint seeds, {note})...')
            benchmark = collect(pool, 'benchmark', args.benchmark_target, args.workers,
                                args.games_per_cycle, exclude=train)

    # Deterministic shuffle so the on-disk order isn't correlated with game order.
    rng = random.Random(12345)
    train_list = [list(p) for p in train]
    bench_list = [list(p) for p in benchmark]
    rng.shuffle(train_list)
    rng.shuffle(bench_list)

    # Cap to the requested targets. The filter is generous for Paskogammon, so a
    # single collection cycle can overshoot by a wide margin; trimming keeps the
    # written sizes predictable (and the benchmark rollout cost bounded).
    train_list = train_list[:args.train_target]
    bench_list = bench_list[:args.benchmark_target]

    print(f'\nWriting to {DATA_DIR}:')
    if args.train_target > 0:
        write_positions(os.path.join(DATA_DIR, 'pasko-train-data'), train_list)
    if args.benchmark_target > 0:
        write_positions(os.path.join(DATA_DIR, 'pasko-benchmark-data'), bench_list)
    print(f'\nTotal time: {time.time() - t0:.1f}s')
    print('\nNext (roll out via Parallelizor, from the PARENT repo):')
    print('  python scripts/rollout_pasko_positions.py pasko-train-data --workers 200')
    print('  python scripts/rollout_pasko_positions.py pasko-benchmark-data --workers 50')


if __name__ == '__main__':
    mp.freeze_support()
    main()
