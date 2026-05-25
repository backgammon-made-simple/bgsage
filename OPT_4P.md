# 4P Optimization Journal

Target: speed up `post_move_analytics` and `cube_action` at 4P (16 threads).
Accuracy budget: ≤ 0.01 equity, ≤ 0.005 probability change.

## Baseline (16 threads, reference 8/6 8/3 position)

| Case               | Wall (s) | Leaves | Internal | Cache hits | Move gen |
|--------------------|----------|--------|----------|------------|----------|
| post_move_money    | 0.710    | 9724   | 463      | 0          | 9723     |
| post_move_match    | 0.717    | 9724   | 463      | 0          | 9723     |
| cube_action_money  | 0.199    | 8083   | 459      | 1557       | 9639     |
| cube_action_match  | 0.204    | 8198   | 459      | 1442       | 9639     |

Post-move 4P is **3.5x slower** than cube_action because the cubeful cache is
bypassed when `arProbsOut != nullptr` in `cubeful_recursive_multi`. Tree
structure is otherwise identical.

## Ranked Ideas

| #  | Idea                                                                                       | Expected speedup | Accuracy risk |
|----|--------------------------------------------------------------------------------------------|------------------|---------------|
| **A** | **Extend cubeful cache to also store probs** so post_move 4P benefits from caching.        | 3-4x on post_move 4P (matches cube_action timing) | None — same recursion, just cached. |
| **B** | Skip per-candidate `cl2cf` when cubeless gap > margin (Janowski monotonicity).             | 20-40% on interior cubeful picks | Tiny — pick a conservative margin (~0.15). |
| **C** | Batch the cubeful pick's `evaluate_probs` via `base_gps->batch_evaluate_candidates_probs`. | 10-20% on cubeful pick calls | None — same NN forward, batched. |
| **D** | Hoist `cube_efficiency`/pip-count out of cubeful pick (use shared cube_x per node).        | 2-5% on interior picks | Marginal — within-node cube_x variance tiny. |
| **E** | Bigger cubeful cache (8k → 32k entries) to reduce collision evictions.                     | 5-10% if collisions dominate | None — memory only. |
| **F** | Tighter filter chain for `best_move_index_cubeful_multi` (4P/3P intermediate).             | Affects only checker-play BMI calls | Could lose winner. |
| **G** | Drop the `arProbsOut` accumulation overhead at sub-leaf depths (compute probs only at top).| Tiny                | Need careful verification. |

Plan: implement A first (the biggest single lever). Then B, C, D, E in order.

## Results journal

### Idea A — KEPT (cubeful cache stores probs alongside equities)

Added `bool has_probs` + `float probs[NUM_OUTPUTS]` to `CubefulCacheEntry`.
Cache lookups with `arProbsOut != nullptr` only hit entries that have probs
stored; cube-decision lookups (no probs requested) hit any entry.

| Case               | Baseline | Idea A   | Speedup | Cache hits |
|--------------------|----------|----------|---------|------------|
| post_move_money    | 0.710s   | 0.459s   | **1.55x** | 0 → 1905 |
| post_move_match    | 0.717s   | 0.514s   | **1.40x** | 0 → 1375 |
| cube_action_money  | 0.199s   | 0.180s   | 1.10x    | 1574 → 1533 |
| cube_action_match  | 0.204s   | 0.182s   | 1.12x    | 1453 → 1428 |

**Numerical output: bit-identical** — all probability and equity deltas are
exactly 0.000000. The cache only memoizes work that the recursion already did
the same way.

### Idea C — DROPPED (batched probs eval in cubeful pick)

Replaced the per-candidate `evaluate_probs` loop in the cubeful interior pick
with `base_gps->batch_evaluate_candidates_equity_probs`. Bit-identical
numerics but no speedup (and a small ~10% regression on cube_action — within
noise but consistent across iterations). Reverted.

| Case               | Idea A   | Idea C   | Speedup |
|--------------------|----------|----------|---------|
| post_move_money    | 0.459s   | 0.462s   | 0.99x   |
| post_move_match    | 0.514s   | 0.521s   | 0.99x   |
| cube_action_money  | 0.180s   | 0.199s   | 0.90x   |
| cube_action_match  | 0.182s   | 0.198s   | 0.92x   |

Hypothesis: with 16 threads handling the 21-roll parallel_for, each thread does
a small batched call. The batched API's per-call setup (thread-local vector
allocation, base hidden layer save) eats whatever delta-eval savings exist for
small candidate sets. The cubeful pick is not the bottleneck I thought it was.

### Idea E — DROPPED (bigger cubeful cache: 8k → 32k)

| Case               | Idea A   | Idea E   | Speedup |
|--------------------|----------|----------|---------|
| post_move_money    | 0.459s   | 0.459s   | 1.00x   |
| post_move_match    | 0.514s   | 0.512s   | 1.00x   |
| cube_action_money  | 0.180s   | 0.201s   | 0.90x   |
| cube_action_match  | 0.182s   | 0.195s   | 0.93x   |

No benefit — cache hits ticked up tiny amount (~0.4%); the 8k cache was not
overflowing. Reverted.

### Idea F — DROPPED (cubeful pick only at plies >= 3)

Hypothesis: ~93% of internal nodes at 4P are at plies=2 (one above the leaf).
The cubeful vs cubeless choice at plies=2 has minimal accuracy impact since
only 1 half-move remains. Restricting cubeful pick to plies >= 3 would skip
most of the cubeful work.

| Case               | Idea A   | Idea F   | Speedup |
|--------------------|----------|----------|---------|
| post_move_money    | 0.459s   | 0.457s   | 1.00x   |
| post_move_match    | 0.514s   | 0.526s   | 0.98x   |
| cube_action_money  | 0.180s   | 0.197s   | 0.91x   |
| cube_action_match  | 0.182s   | 0.198s   | 0.92x   |

Accuracy hit on post_move_match: gw delta **+0.97pp**, exceeds the 0.005
probability budget. Speed unchanged. Reverted on both counts.

### Idea H — KEPT (analyzer was passing n_threads=1 to cubeful_probs_nply)

**Discovery during profiling**: I noticed direct calls to
`bgbot_cpp.cubeful_probs_nply` ran in 0.05s, while `analyzer.post_move_analytics`
took 0.51s for the same numerical work. Tracing the difference, the analyzer
was calling `cubeful_probs_nply(...)` **without `n_threads=`**, so the pybind
default of 1 took effect — the cubeful_recursive_multi's `allow_parallel`
gate (`n_threads > 1 && n_plies > 2`) was false, and all 21 dice rolls ran
serially on the calling thread instead of being dispatched across 16 workers.

Fix: pass `n_threads=getattr(inner, '_parallel_threads', 1)` in
`analyzer.py::post_move_analytics`, mirroring how every other code path
threads the configured parallelism through.

| Case               | Baseline | Idea A   | Idea H   | Total speedup |
|--------------------|----------|----------|----------|---------------|
| post_move_money    | 0.710s   | 0.459s   | **0.053s** | **13.47x**  |
| post_move_match    | 0.717s   | 0.514s   | **0.053s** | **13.63x**  |
| cube_action_money  | 0.199s   | 0.180s   | 0.148s   | 1.34x         |
| cube_action_match  | 0.204s   | 0.182s   | 0.166s   | 1.23x         |

**Bit-identical numerics vs baseline** — purely a parallelism fix. The
small cube_action improvements are noise/variance — cube_action already used
its caller-provided n_threads correctly.

Post-move 4P now takes the same time as cube_action 4P — exactly what we'd
expect since both traverse the same shaped tree. The original 1.5s wall time
the user observed was the combination of the cache-bypass (~3.5x slower than
cube_action, fixed by Idea A) and the missed n_threads parameter
(~10x slower than parallel, fixed by Idea H).

`_cubeful_equity` in analyzer.py had the same omission for the matched-up
`cubeful_equity_nply` calls — fixed in the same commit. Those feed
`checker_play_analytics`'s 3-ply+ cubeful equity, so checker analysis at 3-ply
and above also benefits.

## Final summary

| Case               | Baseline (s) | Final (s) | Speedup |
|--------------------|--------------|-----------|---------|
| post_move_money 4P | 0.710        | **0.051** | **13.84x** |
| post_move_match 4P | 0.717        | **0.051** | **13.94x** |
| cube_action_money 4P | 0.199      | **0.146** | 1.36x   |
| cube_action_match 4P | 0.204      | **0.149** | 1.37x   |

All numerics bit-identical to baseline. Code changes:
- `cpp/include/bgbot/cube.h`, `cpp/src/cube.cpp`: added per-entry probs slot
  + `has_probs` flag to the cubeful cache; cache lookup honors the slot when
  arProbsOut is requested. Exposed `reset_cubeful_counters` /
  `get_cubeful_counters` for Python profiling.
- `cpp/pybind/bindings.cpp`: pybind exposure of the two counter helpers.
- `python/bgsage/analyzer.py`: pass `n_threads=self._parallel_threads` to
  every `cubeful_probs_nply` / `cubeful_equity_nply` call so the configured
  parallelism actually reaches the cubeful recursion.

Ideas that didn't pan out (kept for the record):
- **Idea C**: batched probs eval in cubeful pick — wash, slight regression
  on cube_action; reverted.
- **Idea E**: bigger cubeful cache (8k → 32k) — wash; reverted.
- **Idea F**: cubeful pick only at plies >= 3 — accuracy hit (gw +0.97pp on
  match exceeds 0.005 budget) and no speedup; reverted.
