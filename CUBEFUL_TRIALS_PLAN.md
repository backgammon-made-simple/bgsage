# Plan: Cube/Match-Aware Decisions Inside Rollout Trials

## Goal

During a rollout trial, the simulated mover should pick checker moves by **cubeful
equity** that incorporates current cube state and (for match play) score, not by
**cubeless equity**. Cube decisions during trials are already cubeful in 1-ply
Janowski and N-ply paths; the remaining gap is checker move selection.

This document is a design plan, not an implementation order. Open questions for
the user are collected in §11.

## TL;DR

The cleanest, least invasive design is:

1. Add a new **cube-aware variant** of `best_move_index` (overload) on `Strategy`:
   `best_move_index_cubeful(candidates, pre_move_board, ci, cube_x)`.
2. Plumb `CubefulBranch` into the four call sites in
   [run_trial_unified](cpp/src/rollout.cpp) and the two cache prefills, so trials
   ask for cube-aware decisions when active.
3. Implement the new method on `NNStrategy` (Janowski/MWC over candidates),
   `MultiPlyStrategy` (cube-aware leaf collapse via the existing `cubeful_recursive_multi`
   tree), and `RolloutStrategy` (inner cube-aware rollout). Most of the math is
   already there; this is mostly a routing change.
4. **Multi-branch trials** (cube_decision: ND+DT branches) get the trickiest
   treatment: each branch may pick a different move on a given roll, which means
   branches' boards diverge. Two options on the table — see §6, this is the main
   design choice the user needs to make.
5. Move0/Move1 caches need branch-aware variants (or get bypassed in the
   cubeful-decision path). VR luck math extends per-branch (already partially
   there).

Expected performance impact: see §10 — best case ~10–20%, worst case ~50% slower.

---

## 1. Scope of the Change

### In scope

- **Single-branch cubeful position trials** (`cubeful_rollout_position`, used by
  `BgBotAnalyzer.checker_play` rollouts) — pick checker moves by cubeful equity
  against the branch's current cube state.
- **Two-branch cube-decision trials** (`cubeful_cube_decision`) — pick checker
  moves per branch by per-branch cubeful equity. Major design question lives here.
- **Move0Cache and Move1Cache** — currently shared across branches and computed
  cubelessly. Need branch-aware behavior or a clean bypass.
- **VR cubeful luck** — already per-branch; extend to handle per-branch chosen
  moves correctly.
- **Truncation evaluation** — already uses cubeful Janowski / `cubeful_equity_nply_multi`
  per branch. No change needed beyond consistency checks.
- **Inner strategies used at trial time** (`checker_strat_`, `checker_late_strat_`,
  `cube_inner_rollout_`, `cube_late_inner_rollout_`, `truncation_strat_`) — all
  need to support the new cube-aware path so 2-ply / 3-ply / truncated-rollout
  trial strategies become cube-aware.

### Out of scope (for this change)

- The N-ply *cubeful* recursion (`cubeful_recursive_multi`) still picks moves
  by cubeless 1-ply equity at every node. This is documented in
  [MULTI-PLY.md §6](MULTI-PLY.md). Making *that* cube-aware is a separate plan —
  it would re-architect the cubeful tree to evaluate each candidate under each
  cube state and is dramatically more expensive. The current plan changes
  *rollout trials only*.
- The top-level move ranking in `BgBotAnalyzer.checker_play` is already cubeful
  (post-hoc Janowski / `cubeful_rollout_position` already produces cubeful
  equities per candidate). No top-level API changes.

## 2. Current State (anchors from `ROLLOUT.md`)

- `run_trial_unified` calls `best_move_index` at four sites (cube-state-agnostic):
  - [`rollout.cpp:595`](cpp/src/rollout.cpp:595) — Move0 prefill, `base_` path
  - [`rollout.cpp:597`](cpp/src/rollout.cpp:597) — Move0 prefill, `current_strat` path
  - [`rollout.cpp:677`](cpp/src/rollout.cpp:677) — Move1 prefill (always `base_` per
    [`ROLLOUT.md` §10](ROLLOUT.md))
  - [`rollout.cpp:680`](cpp/src/rollout.cpp:680) — Move1 prefill, fallback
  - Inside the per-trial loop, `current_strat.best_move_index(candidates, board)`
    around [`rollout.cpp:1117`](cpp/src/rollout.cpp:1117) (selection based on
    `checker_strat_` / `checker_late_strat_` / `base_`)
- `CubefulBranch` lives in
  [`rollout.h:334`](cpp/include/bgbot/rollout.h:334) and holds `cube`,
  `basis_cube`, `vr_luck`, `finished`, `final_equity`. **No board** — all branches
  share the trial's single `Board board`.
- All cube math we need already exists in [`cube.h`](cpp/include/bgbot/cube.h):
  `cl2cf_money`, `cl2cf_match`, `cl2cf` (dispatcher), `cube_efficiency`, plus
  `cube_decision_1ply` / `cube_decision_nply_multi` / `cubeful_equity_nply_multi`.

## 3. New Strategy Interface

Add a parallel cube-aware overload of `best_move_index` on `Strategy`:

```cpp
// strategy.h additions
class Strategy {
public:
    // Existing cubeless overloads ...

    // Cube-aware: pick the candidate with highest cubeful equity for the given
    // cube state. Returned index is into `candidates`.
    //   - cube_x: cube efficiency for the pre-move board (cached by caller to
    //     avoid recomputing per branch).
    //   - Default implementation: 1-ply cubeless eval each candidate, apply
    //     cl2cf(ci, cube_x), return argmax. Overridden by MultiPly/Rollout for
    //     deeper search.
    virtual int best_move_index_cubeful(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        const CubeInfo& ci,
        float cube_x) const;

    // Batched variant: choose best candidate independently for each cube state.
    // Returns one index per cube. Default impl: loop. Overridden where batching
    // can share NN work (key for MultiPlyStrategy — the 1-ply candidate scores
    // are shared, only Janowski differs per state).
    virtual void best_move_index_cubeful_multi(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        const CubeInfo* cubes,
        int n_cubes,
        float cube_x,
        int* out_indices) const;
};
```

**Why a new method instead of changing the existing signature?**

- Strategies are widely used outside rollouts (top-level
  `BgBotAnalyzer.checker_play`, multi-ply N-ply recursion as a leaf evaluator,
  `score_benchmarks_*`, etc.). Forcing all of them to pass a `CubeInfo` would
  ripple through ~20 callers and the entire benchmark suite.
- Cube-aware ranking is meaningful only when the caller has a cube state. A
  separate method keeps the cubeless path zero-overhead and zero-risk.
- The existing cubeless `best_move_index` continues to be the canonical move
  selector everywhere else, including inside `cubeful_recursive_multi` (which
  is out of scope here).

## 4. Strategy Implementations

### 4.1 `NNStrategy::best_move_index_cubeful` (1-ply path)

```
for each candidate:
    probs = evaluate_probs(candidate, pre_move_board)
    clamp_probs_to_board(probs, candidate)
    cubeful_eq = cl2cf(probs, ci, cube_x)
return argmax
```

For multi: evaluate each candidate's probs once, loop `cl2cf` per cube. Use
`batch_evaluate_candidates_equity_probs` to get probs in a single batched NN call.

### 4.2 `MultiPlyStrategy::best_move_index_cubeful` (N-ply path)

The cleanest re-use of existing machinery is to call
`cubeful_equity_nply_multi(candidate, cubes, n, base, plies)` per candidate (it
already evaluates a single board against multiple cube states sharing the cubeful
tree). Implementation sketch:

```
results[n_cubes][n_candidates]
for c in candidates:
    cubeful_equity_nply_multi(c, cubes, n_cubes, base_, plies, results[c])
for k in n_cubes:
    out_indices[k] = argmax over c of results[k][c]
```

**Subtlety**: `cubeful_equity_nply_multi` evaluates one board's equity at N-ply.
Wrapping it in a per-candidate loop preserves the "all cubes share the same NN
tree per candidate" optimization but loses sharing **across candidates**. For
trial-level use this is fine — we typically have 1–24 candidates per roll.

If profiling shows this is too slow, the alternative is to extend
`cubeful_equity_nply_multi` to take a list of `(board, cube_state)` pairs and
share the N-ply recursion across both axes. Defer until measured.

### 4.3 `RolloutStrategy::best_move_index_cubeful` (truncated-rollout-within-rollout)

For inner rollouts used as checker strategy: per candidate, run
`cubeful_rollout_position(candidate, cube)` for each cube state. This is the
existing entry point — it produces cubeful equity per candidate per cube state.

### 4.4 `BearoffStrategy::best_move_index_cubeful`

Delegate to wrapped strategy after intercepting bearoff positions. For bearoff
candidates, look up exact cubeless probs from DB and apply `cl2cf` directly
(no NN call). Mirrors the existing bearoff-aware default `best_move_index`.

## 5. Trial Loop Integration

### 5.1 Plumbing `CubefulBranch` into the call sites

`run_trial_unified` already maintains `branches[]` and knows `cube_active`. At
each move selection point, decide:

- `cube_active == false` (cubeless trials, or all branches have dead cubes):
  call existing cubeless `best_move_index`.
- `cube_active == true`:
  - Compute `cube_x` once for the current board (re-use the existing helper).
  - Call `best_move_index_cubeful_multi` with the active branches' cube states.
  - Apply the per-branch chosen indices to per-branch boards (see §6).

### 5.2 `cube_x` re-use

`cube_efficiency` depends on pre-move board + cubeless probs + race/pip counts.
Compute it once per call site per move (the cubeless probs needed by VR mean are
already computed in Phase 3 → reuse).

### 5.3 1-ply VR base re-use

When the trial-level checker strategy is `base_` (1-ply), the VR mean computation
in Phase 3 already calls `batch_evaluate_candidates_best_prob` over the 21 rolls
and produces per-candidate probs. For cube-aware selection we additionally need to
apply Janowski to each candidate's probs. Two cheap options:

- Extend VR's batch helper to also return the per-candidate **probs vector** (not
  just the best one) so Janowski can be applied without re-evaluating.
- Or just call `batch_evaluate_candidates_equity_probs` once and do both VR mean
  and cubeful selection from the same probs.

This avoids any per-candidate NN re-evaluation in the cube-aware path at 1-ply.

## 6. The Multi-Branch Question (Cube Decision Trials)

This is the central design choice and the only place I need a user decision.

### Setup

`cubeful_cube_decision` runs trials with two branches sharing a single trial:

- ND branch: original cube state
- DT branch: doubled cube, opponent owns

Both branches today share one board because moves are cube-agnostic. If the
*move* depends on cube state, branches' boards diverge after the first turn where
moves differ.

### Option A — Per-branch boards (full fidelity)

Make `CubefulBranch` hold its own `Board board` and run move generation /
selection / flips independently for each active branch.

| Pros | Cons |
|---|---|
| Most accurate: each branch plays its own optimal line. | Trial cost roughly doubles in active-cube-active mode (two branches → two boards → two move-gen calls + per-branch flips). |
| Cleanest semantics — no fuzz about which cube state drove the choice. | Move1Cache becomes per-branch and the cross-branch sharing optimization (one cubeless mover_probs feeding both branches' Janowski) partially breaks because move1 board differs. |
| VR cubeful luck stays per-branch, exact. | Larger memory per trial (extra Board per branch + per-branch dice usage). |

### Option B — Shared board, "representative" cube state for selection

Keep one shared board per trial. Pick the move using a single chosen branch's
cube state (e.g. always the ND branch's cube, since ND is the "natural" path).
The DT branch then evaluates the trial-evolved cubeless probs through *its* cube
in VR luck and terminal conversion, but the move chosen reflects ND's preferences.

| Pros | Cons |
|---|---|
| Trivial implementation — share everything, just route ND's cube_state into `best_move_index_cubeful`. | DT-branch equity is biased: a position where the DT-side player should overplay for gammons / defend differently never gets that play. |
| Move0/Move1 caches stay shared across branches with no changes. | Cube-decision rollout's ND vs DT differential narrows — possibly negating the whole point of the change for the cube-decision use case. |
| ~0% perf overhead vs current. | Semantics fuzzy — what does "DT equity" mean when DT-branch moves were chosen for ND? |

### Option C — Two passes, no shared trial (cleanest semantics, expensive)

Run `cubeful_cube_decision` as two separate `cubeful_rollout_position` calls,
one per scenario. Each is single-branch and cleanly cube-aware. Lose the
shared-RNG correlation that gives ND vs DT cleaner SE today.

| Pros | Cons |
|---|---|
| Reuses single-branch machinery entirely; cube-decision rollout becomes a thin wrapper. | Trial cost doubles. |
| Single-branch path is the simplest place to add cube awareness — focus complexity there. | ND/DT SE correlation is lost; reported `nd - dt` gap has higher variance for the same trial budget. |
| No multi-branch cache or VR complications. | Behavior change visible to users (SE of ND vs DT differential will look different). |

### Recommendation

**Option A (per-branch boards)** is the right long-term answer because it
preserves the design intent of cube-decision rollouts (shared dice, divergent
strategy) while being mathematically clean. The cost increase (~2× in
two-branch trials) is bounded and acceptable.

Option B is a useful first milestone if you want to see the impact on
`cubeful_rollout_position` (single branch, no multi-branch complications) before
committing to A. Option C is a "Plan B" if A's complexity blows up.

## 7. Cache Strategy

### 7.1 Move0Cache

- **Cubeless trials**: unchanged.
- **Single-branch cubeful trials**: extend cache entries to be keyed by the
  branch's cube state. In practice each `cubeful_rollout_position` call has
  exactly one cube state, so the cache is "cube-state-stamped" at construction
  and `chosen[roll]` is computed for that specific cube. No structural change
  beyond adding a `CubeInfo cube_for_prefill` field.
- **Two-branch cubeful trials (Option A)**: cache must be keyed by `(roll, branch_idx)`.
  Easiest representation: two separate Move0Cache instances, one per branch.
  Prefill cost roughly doubles.

### 7.2 Move1Cache

The Move1Cache stores rich VR data, including `mover_probs` (used for cube
decisions, cube-state-independent) and `roll_best_probs` / `chosen` (which would
diverge per branch under Option A).

Two sub-options:

- **B.1** Keep `mover_probs` shared (it's cube-state-agnostic) and store
  per-branch `chosen[roll]` and `actual_probs[roll]`. Saves ~half the NN cost
  of full duplication.
- **B.2** Per-branch Move1Cache, simpler but more NN work.

Recommend B.1 (shared mover_probs, per-branch move arrays).

### 7.3 SharedPosCache

The cross-thread N-ply position cache keys on `(board_hash, ply)` today. The
cube-aware N-ply selection uses `cubeful_equity_nply_multi`, which internally
keys its own cubeful cache on `(board, cci, plies, fTop)` — i.e., already
cube-state-aware. So `SharedPosCache` doesn't need to change for this plan;
cube-aware N-ply work uses the cubeful recursion's existing cache.

The main hit is that the cubeful tree is *deeper* per candidate (we now evaluate
each candidate at N-ply against each cube state) versus the cubeless path
(N-ply once per candidate). Cache hit rate within a single
`best_move_index_cubeful_multi` call is high; across calls it depends on
position repetition.

## 8. VR Math Extensions

Per [`ROLLOUT.md` §4](ROLLOUT.md), cubeful VR already tracks per-branch luck:

```
mean_cf[b]   = sum_rolls w_r * cl2cf(roll_best_probs[r], branch[b].cube, cube_x)
actual_cf[b] = cl2cf(actual_probs[chosen_for_branch_b], branch[b].cube, cube_x)
luck_cf[b]   = actual_cf[b] - mean_cf[b]
branch[b].vr_luck += sp_sign * luck_cf[b]
```

Under Option A, `actual_probs[chosen_for_branch_b]` differs per branch because
the chosen move differs. Concretely: the VR mean stays computed from
`roll_best_probs` (the per-roll *cubeless* best for VR purposes — best for E[luck]=0
to hold), but the **actual** path's probs are taken from the branch-specific
chosen move.

This is mathematically clean: as long as `mean_cf[b]` is the expectation of
`actual_cf[b]` over rolls (which it is, since `actual_cf[b]` per roll = cubeful
eval of branch b's chosen move, averaged with weight w_r), luck has zero mean
and bias-cancellation holds.

**One wrinkle**: the existing VR mean is computed as "cubeless best for each
roll, then `cl2cf` per branch". For the cubeful-aware variant, the *mean* should
arguably be "cubeful best for each roll per branch", not "cubeless best, then
cl2cf". Both are valid VR-mean choices; the cubeful one is more aligned with the
actual being computed but adds a 21×n_branches Janowski computation per move.
Cheap, so do it that way. Update §4 of ROLLOUT.md to reflect.

## 9. Inner Strategy Construction

Trial-level strategies (`checker_strat_`, etc.) are built once at `RolloutStrategy`
construction from `TrialEvalConfig`. For the cube-aware path:

- 1-ply (`base_`): the `NNStrategy::best_move_index_cubeful` default impl
  handles it. No build-time changes.
- N-ply (`MultiPlyStrategy`): override `best_move_index_cubeful` on
  `MultiPlyStrategy`. The internal filter (`{2, 0.03}`) used today is per the
  cubeless 1-ply pre-filter. For the cube-aware path, the filter should rank
  candidates by cubeful equity before the N-ply rescore. Tighter than the cubeless
  filter is fine — Janowski is monotonic enough in probs that filtering by
  cubeless-then-cubeful-rerank should match closely.
- Truncated rollout (inner `RolloutStrategy`): override `best_move_index_cubeful`
  via the `cubeful_rollout_position` path. Single-threaded inner rollouts
  already work; just per-candidate per-cube.

No `RolloutStrategy` constructor changes needed — the new method is
auto-available wherever the strategy is used.

## 10. Performance Analysis

### 10.1 Per-move overhead (single-branch cubeful trials)

| Component | Today | After change |
|---|---|---|
| 1-ply VR mean (21-roll batch) | 21 batched NN calls | Same + 21× cl2cf (~µs) |
| Actual move selection (`base_`) | 1 batched NN argmax | Same + 1 cl2cf (~ns) |
| Actual move selection (N-ply) | 1× N-ply over filter survivors | Same + per-survivor `cubeful_equity_nply_multi` call with n_cubes=1 — roughly same cost as cubeless N-ply since the tree is the same |
| Cube decision (already cubeful) | unchanged | unchanged |

Net: **~5–10% slower** for single-branch trials. The cubeful tree depth is the
same as the cubeless tree depth (one cube state, one tree).

### 10.2 Per-move overhead (two-branch cubeful trials, Option A)

| Component | Today | After change (Option A) |
|---|---|---|
| Move generation per trial | 1× | 2× (per-branch boards diverge eventually) |
| 1-ply VR mean | 21 batched NN calls | 2× same (per-branch board) |
| N-ply move selection | shared cubeless tree | 2× shared cubeful trees (one per branch board) |
| Cache (Move0/Move1) | 1 cache, 21 entries | 2 caches, 21 entries each |

Net: **~30–50% slower** in two-branch mode. Trials run roughly 2× the work for
~1.5× wall-time because dice generation, branch state, and parallelism overhead
amortize.

### 10.3 Per-move overhead (Option B, shared board)

Net: **~5–10% slower** because the shared-board path is identical to
single-branch in cost. Cubeful selection adds the Janowski layer at each call.

### 10.4 Cache-hit rate

`SharedPosCache` and the cubeful recursion's own internal cache should be
unaffected — both already key on cube-relevant state where it matters. The
biggest risk is that N-ply trees evaluated under different cube states have
different leaf Janowski values but identical NN intermediate values — both
caches recognize this (the cubeful recursion stores the NN intermediate
separately from the cubeful collapse).

### 10.5 Benchmark expectations

Estimated wall-clock impact on standard configs:

| Config | Today | Expected (single-branch) | Expected (two-branch, Option A) |
|---|---|---|---|
| 1T (XG Roller, 72 trials) | 1× | 1.05–1.10× | 1.3–1.5× |
| 2T (360 trials, 2-ply) | 1× | 1.05–1.10× | 1.3–1.5× |
| 3T (360 trials, 3-ply) | 1× | 1.10–1.20× | 1.4–1.6× |
| Full R (1296 trials, play-to-end) | 1× | 1.15–1.25× | 1.5–1.7× |

These are rough; real numbers will need benchmarking. Variance in trial paths
means cache fragmentation could push the high end up.

## 11. Decisions Made

1. **Branch divergence in cube-decision trials**: **Option A — per-branch boards.**
   Each `CubefulBranch` gains its own `Board` field; trial loop runs move
   generation, selection, flips independently per active branch.

2. **Default behavior**: **Flag-gated, default off; flip default after validation.**
   Add `RolloutConfig.cubeful_trial_moves` (bool, default `false`). Existing
   benchmarks and cached analytics keep their current values until the flag flips.

3. **VR mean construction**: **Cubeful-best per roll per branch.**
   At each VR-active half-move, the per-branch mean is computed by selecting the
   *cubeful*-best candidate for each of the 21 rolls under that branch's cube
   state, then weighting. Mean and actual live in the same value space; luck
   cancellation holds.

4. **Validation bar before flipping the default**: **Manual XG comparison + curated
   XG/GNUbg cube-decision set.** User will hand-verify against XG on chosen
   positions; structured comparison against a curated set of cube decisions
   provides regression confidence. (Benchmark PR sweep is optional but
   recommended as a safety net; not required for flip.)

## 12. Remaining Decisions (less load-bearing)

These have clear recommendations; the user can confirm or override.

5. **N-ply hybrid mode filter**: Cubeless filter + cubeful leaf rescore (filter
   for pruning, Janowski-monotonicity protects winners) vs cubeful filter too.
   *Recommendation: cubeless filter, cubeful leaf.*

6. **Beaver in trial selection**: `cl2cf` with `beaver=true` during candidate
   ranking so the player anticipates beaver responses (consistent with 1-ply cube
   decisions during trials, which already respect beaver).
   *Recommendation: yes, consistent.*

7. **Truncation under Option A**: Branches reach different truncation boards →
   the existing batched `cubeful_equity_nply_multi(n_cubes=n_branches)` call
   becomes per-branch (n_cubes=1). Small perf loss at truncation.
   *Recommendation: accept.*

8. **Top-level rollout-based move ranking** (`RolloutStrategy::best_move_index`,
   used by callers that treat the rollout as a `Strategy`): should this gain a
   cubeful variant too? Today, top-level `checker_play` already ranks candidates
   by full cubeful rollout (via `cubeful_rollout_position` per candidate). The
   question is whether the `Strategy` interface itself exposes the cubeful
   ranker for callers that go through the abstract interface.
   *Recommendation: yes, falls out naturally; one extra binding line.*

9. **Match-only or money-too**: Money play also has cube ownership effects
   (aggressive doubling-zone play vs cube-owned defensive play). `cl2cf` is the
   unified dispatcher; gating to match-only would be artificial.
   *Recommendation: both, no separate gating.*

10. **Test coverage**:
    - Unit tests on `NNStrategy::best_move_index_cubeful` picking the right
      candidate in canonical setups (e.g. gammon-favored move when match score
      makes it correct).
    - Integration tests on `cubeful_rollout_position` showing probability shifts
      in known-relevant positions (DMP, Crawford, the analyzed 1-4-to-5 position).
    - Regression tests pinning post-change values for ~5 positions so future
      refactors don't silently shift them.
    *Recommendation: all three.*

## 12. Estimated Implementation Effort

Assuming Option A and flag-gated default-off:

| Step | Days |
|---|---|
| New Strategy interface methods + default impls | 1 |
| `NNStrategy::best_move_index_cubeful` + batched variant | 0.5 |
| `MultiPlyStrategy::best_move_index_cubeful` | 1 |
| `RolloutStrategy::best_move_index_cubeful` (truncated inner) | 0.5 |
| `BearoffStrategy::best_move_index_cubeful` | 0.25 |
| `CubefulBranch` board field + trial-loop plumbing (Option A) | 1.5 |
| Move0Cache / Move1Cache per-branch variants | 1 |
| VR luck per-branch chosen-move fix | 0.5 |
| Python binding additions (RolloutConfig flag, etc.) | 0.25 |
| Tests (unit + integration + regression) | 1.5 |
| Documentation updates (ROLLOUT.md §5–10, this file as ADR) | 0.5 |
| Benchmark sweep + perf tuning | 2 |
| Buffer for cache-hit-rate surprises | 1 |
| **Total** | **~11 days** |

A leaner Option B (shared-board) variant cuts the multi-branch plumbing and
saves ~3 days. Option C is similar to A in effort but cleaner.
