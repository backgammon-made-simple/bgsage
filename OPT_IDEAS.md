# Optimization Ideas — Flag-On Speedup

Target: shrink wall-clock for `cubeful_trial_moves=True` on the analyzed
position across {3T, R(1296,2P)} × {checker_play, cube_action,
post_move_analytics}, match play. Tolerance: ≤ 0.01 cubeful equity change
and ≤ 0.005 cubeless prob change. Bonus if flag-OFF also speeds up. Test
on 16 cores.

## Root-cause picture

Flag-on slowdown ranks (from code reading):

1. **Caches bypassed at moves 0 and 1** ([rollout.cpp:1245-1277](cpp/src/rollout.cpp))
   — `use_cubeful_select` skips Move0Cache and Move1Cache `chosen[]` because
   they store cubeless-best moves. Each trial re-runs full cubeful BMI at moves 0
   and 1 (the most expensive moves: deep N-ply, many candidates). With 1296 trials
   and caches, the cubeless path computes ~21 (move 0) + ~21×21 (move 1) BMI calls
   total. Without caches, it's 1296 × 2 = 2592 calls — roughly 5× more BMI work
   at the two costliest moves.
2. **MultiPlyStrategy::best_move_index_cubeful_multi** ([multipy.cpp:838](cpp/src/multipy.cpp))
   does the cubeless filter chain, then loops per-survivor `cubeful_equity_nply_multi`.
   Adds ~10–25% per call vs cubeless N-ply (more leaf Janowski work; tree itself
   identical because internal move selection stays cubeless).
3. **No `best_candidate_idx` reuse** ([rollout.cpp:1317-1322](cpp/src/rollout.cpp))
   — flag-off 1-ply path reuses VR's stored best-candidate index. Flag-on always
   re-picks via cubeful BMI.
4. **Strategy::best_move_index_cubeful_multi default** ([strategy.cpp:85](cpp/src/strategy.cpp))
   currently loops `evaluate_probs` per candidate instead of using
   `batch_evaluate_candidates_equity_probs`. I disabled the batch path during
   debugging due to a suspected thread issue; never re-verified.

Flag-OFF cost is dominated by Phase 1 cube decisions per move (`cube_decision_
nply_multi` at decision_ply) and Phase 3 VR mean (21 × 1-ply NN evals per move).

## 10 Ideas (ranked by expected impact × ease)

| # | Idea | Expected wins | Risk |
|---|---|---|---|
| **1** | **Cube-state-stamped Move0Cache / Move1Cache for flag-on**. Cube state is fixed across all trials in a rollout (it's a parameter to `cubeful_evaluate_board` / `cubeful_cube_decision`), so prefill the caches with **cubeful**-best moves under that cube state. Single-branch trials: cache stamped with the one cube. Two-branch trials: cache stamped with `branches[0]`'s cube (matches shared-board MVP). | 2–3× speedup on checker_play and cube_action; big win on post_move_analytics. The biggest single lever. | Cache must be invalidated/rebuilt per (cube state, decision_ply, base_strategy) tuple. Mostly mechanical. |
| **2** | **Reuse cubeless `best_candidate_idx` when the cubeless gap dominates**. At 1-ply, if the cubeless equity gap between rank-1 and rank-2 exceeds a margin (e.g. > 0.10), Janowski-monotonicity guarantees the cubeful winner is the same for any cube state. Skip cubeful evaluation in that case. | 20–40% speedup at the BMI call rate where this hits. Especially on simple positions where the move choice is obvious. | Needs careful margin proof — margin must dominate `max_cl2cf_slope × gap_threshold`. Use a conservative margin (0.15-0.20) to be safe. |
| **3** | **Skip cubeful selection at ultra-late moves regardless of flag**. The trial loop already drops to base (1-ply cubeless) at `ultra_late_threshold`. Make flag-on respect this — at deep moves, picks-via-cubeful adds noise more than it adds signal. For R with `ultra_late_threshold=9999` this needs a new cubeful-specific threshold (e.g. drop after move ~10). | 50% speedup for R full-game trials. Mostly free. | Need to verify accuracy holds for late-game cube-aware play. |
| **4** | **Tighter cubeful filter chain**. Currently runs cubeless filter, then ALL 5 survivors get full cubeful N-ply. Add an intermediate cubeful 1-ply step that drops to 2–3 candidates before the expensive N-ply. | 30–40% speedup on cubeful BMI calls. | Tighter filter risks dropping the cubeful winner. Use a generous threshold (0.04 cubeful) to mitigate. |
| **5** | **Restore `batch_evaluate_candidates_equity_probs` in Strategy default**. Investigate why I disabled it; if it's just a misread, restore. Even if there's a real thread issue, can probably wrap with a per-thread strategy clone or local lock. | 10–20% on the 1-ply default path. Helps when `current_strat` is base (post-ultra-late or decision_ply=1 configs). | Need to root-cause the original segfault to avoid regression. |
| **6** | **Specialize `n_cubes == 1`** in MultiPlyStrategy::best_move_index_cubeful_multi. For post_move_analytics (single branch), the multi-cube bookkeeping is overhead. Single-cube path skips the cf_equities[i][c] 2D vector, calls `cubeful_equity_nply` directly. | 5–10% on single-branch BMI calls. | Negligible. Just a path simplification. |
| **7** | **Cache cl2cf per (probs, cube_state)** inside the trial. Phase 3's per-branch cubeful VR mean computes cl2cf over 21 rolls × n_branches. If `move1_entry->roll_best_probs` is shared across trials, the cl2cf result for each is too — cache once per cube_state. | 5% on cube_active trials with move1 cache. | Mostly transparent. |
| **8** | **Coalesce Phase 1 cube decisions** when the cube isn't going to matter. At cube=2 already, redouble to 4 is the only future cube turn. After both branches have D/T'd, Phase 1 is mostly redundant work. Add a "cube settled" fast path. | 5–10% on cube_action and post_move_analytics where cube turns rapidly. | Need careful accounting of when "settled" is safe. |
| **9** | **Cubeful selection at lower ply for late moves**. Like `late_ply` for cubeless, but `cubeful_late_ply`. E.g. 3-ply cubeful early, 1-ply cubeful late. Combined with Idea 3 (drop entirely past ultra-late) this gives a 2-stage falloff. | Synergistic with Idea 3; small alone. | Marginal accuracy loss in late game. |
| **10** | **Parallelize cubeful BMI across survivors** in MultiPlyStrategy when `parallel_evaluate_` is true and we're at the top level. Currently serial — 5 survivors × cubeful tree calls run sequentially. | 2–3× on a 5-survivor BMI call IF trial parallelism isn't already saturating. Probably zero benefit at 16 threads × 1296 trials. | Oversubscription risk; trial-level parallelism dominates. Skip unless explicit need. |

## Execution order

Do tier-A ideas first (1, 2, 4), measure after each. Then tier-B (3, 5, 6, 8).
Skip tier-C (9, 10) unless impact gap warrants. Measure with
`scripts/opt_bench.py --label <idea_name> --save data/opt_bench.json` after
every change.
