# Comparing the Open Sage Bot Engine to XG's Bot Engine

XG (eXtreme Gammon) is a standard reference for backgammon analysis, and its bot engine is considered a very strong one.

XG does not include an API to run its bot engine programmatically, so we could not run head-to-head games between Open Sage and XG at the scale required to identify small differences between them.

However, we did settle on an approach to test Open Sage against XG at scale using XG's Batch Analysis function.

Our goal was to compare Open Sage evaluations against XG evaluations at a comparable level. We compared:
* Sage 3T vs XG Roller ++. Both are truncated rollouts that incorporate variance reduction, truncate after 7 turns, and use 3-ply (or better) for decisions along each simulation path.
* Sage 2T vs XG Roller +. Like 3T/++ except that they make 2-ply decisions internally.
* Sage 1T vs XG Roller. Truncated rollouts with 72 (Sage 1T)/42 (XG Roller) paths, use variance reduction, truncate after 5 turns, and use 1-ply evaluations internally.
* Sage 3P vs XG 3-ply. Both are algorithms that look forward three plies (turns) and average the results over those possible futures. At the end of each path both do a 1-ply calculation - that is, the raw neural network output.
* Sage 4P vs XG 4-ply. Four-ply lookahead.

We looked at three approaches:

* Rollout PR: we simulated money games and match play over many games, rolled out the closest decisions, and scored bot decisions against these rolled out results, and ended up with a Performance Rating (PR) against the rollout truth. We store these benchmark decision results. Then we run each decision by a candidate bot and ask it to give its decision, and score its result against the benchmark equities.
* Disputed Positions: we simulated money games and match play over many games and found the subset of positions where Sage 3T and XG Roller ++ differed on their decision. We rolled those out to see which bot got closer to the truth.
* Real-Match PR Agreement: instead of measuring strength against a rollout truth, we ask a practical question — if you analyze a real match in XG and again in Sage, do the two engines report the same Performance Rating? We re-analyzed hundreds of real tournament matches that had already been analyzed in XG, and compared the per-player PRs the two engines produced.

## Rollout PR Analysis

This is similar in approach to the analysis done on XG (and a number of other bots) in 2012: https://www.extremegammon.com/studies.aspx.

### Money Games

#### Rollout PR Algorithm

We simulated 500 money games of Sage 3P vs Sage 3P. We ran through all the decisions, and did a second pass, re-evaluating any decisions at Sage 3T where the best decision was within 0.05 equity of the next best decision. We then did a third pass, rolling out any decisions which Sage 3T evaluated as within 0.02 equity of the next best decision. We saved out all those results and counted them as the "true" decision results, against which we can benchmark any bot's decisions.

For rollouts we used Open Sage rollouts with 3P decisions for checker play and cube actions. We ran batches of 1,296 paths until the 95% accuracy range on the equity was less than 0.005, or it did 20,736 (=16 times 1,296) paths.

For a given bot (and evaluation level), we had the bot evaluate its decision for each one of those benchmark decisions, and scored it against the benchmark truth. We calculated a Performance Rating (PR) as the average error (as measured against the benchmark equities) multiplied by 500. We also broke out the results into checker play and cube action PRs.

For XG results, we manually ran XG's Batch Analyze on the 500 individual game files, then automatically parsed the XG decisions from the .xg files it generates (one per game). The Batch Analyze settings were 3-ply decisions, moving to the listed eval level for disputes.

#### Rollout PR Results

There were 17,535 decisions across 16,889 positions. Of the 16,889 positions, 7,652 were settled at 3-ply; the other 9,237 were re-evaluated at 3T, of which 3,260 settled there and 5,977 were rolled out. Some rollouts were very quick, while the slowest took well over an hour to roll out on a machine with 16 cores.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.22 | 0.19 | 0.36 | 0.02 | 0.25 | 0.17 | 0.31 | 0.29 |
| XG Roller ++ | 0.32 | 0.31 | 0.38 | 0.04 | 0.40 | 0.24 | 0.41 | 0.44 |
| Sage 2T | 0.26 | 0.23 | 0.44 | 0.02 | 0.32 | 0.21 | 0.39 | 0.30 |
| XG Roller + | 0.41 | 0.41 | 0.39 | 0.05 | 0.59 | 0.31 | 0.47 | 0.54 |
| Sage 1T | 0.50 | 0.52 | 0.40 | 0.04 | 0.58 | 0.43 | 0.60 | 0.74 |
| XG Roller | 0.53 | 0.54 | 0.48 | 0.05 | 0.63 | 0.44 | 0.71 | 0.66 |
| Sage 4P | 0.42 | 0.41 | 0.50 | 0.07 | 0.54 | 0.38 | 0.45 | 0.56 |
| XG 4-ply | 0.46 | 0.45 | 0.52 | 0.06 | 0.58 | 0.40 | 0.57 | 0.58 |
| Sage 3P | 0.60 | 0.61 | 0.57 | 0.14 | 0.79 | 0.52 | 0.64 | 0.78 |
| XG 3-ply | 0.57 | 0.57 | 0.58 | 0.05 | 0.71 | 0.48 | 0.73 | 0.71 |
| Sage 2P | 1.66 | 1.43 | 2.88 | 0.40 | 1.81 | 1.85 | 1.86 | 1.78 |
| Sage 1P | 2.63 | 2.52 | 3.20 | 0.78 | 2.66 | 2.80 | 3.11 | 2.99 |

Sage evaluations are stronger than their equivalent XG evaluations in every case except 3-ply, where XG is slightly stronger, but the two are very close.

#### Running the Pipeline

The rollout-PR data set is built entirely by `scripts/benchmark_money.py`, run
from the `bgsage/` repo root — it resolves its Python path and the compiled
`bgbot_cpp.pyd` from inside `bgsage/`, so the only prerequisite is a local Open
Sage build (no external services). The build is three adaptive-precision passes,
each an independently resumable stage of `benchmark_money.py build`, so you can
run them one at a time, all locally. The `--n-games 100` below is just an example
— scale it up for a larger set.

**1. Simulate the games and capture 3P (pass 1).**

```bash
python scripts/benchmark_money.py build --stages pass1 --n-games 100 --workers 6
```

Plays `--n-games` Sage-3P-vs-Sage-3P money games (Jacoby + beavers on) across
`--workers` parallel self-play processes, capturing 3-ply checker and cube
analytics for every real decision. Writes one `build/stage1/seed_<N>.json` per
game; with `--write-txt` (on by default) it also writes an XG-import
`xg/seed_<N>.txt` transcript per game — those are the files you later batch-
analyze in XG to score XG against the same positions.

**2. Re-evaluate close decisions at 3T (pass 2).**

```bash
python scripts/benchmark_money.py build --stages pass2 --n-threads 16
```

Re-evaluates in-process every decision whose 3-ply best-vs-second-best gap is
under 0.05, using Sage 3T (the Roller++-style truncated rollout). `--n-threads`
is the thread count per evaluation. Appends to `build/stage2_3t.jsonl` and is
resumable (a re-run skips positions already done).

**3. Roll out the closest decisions (pass 3).**

```bash
python scripts/benchmark_money.py build --stages pass3 --n-threads 16
```

Rolls out every decision still within 0.02 equity after the 3T pass: 1,296-path
batches with 3-ply checker and cube decisions and variance reduction, repeated
until the 95% equity band is under ~0.005 or 16 batches (20,736 paths) are
reached. Appends to `build/stage3_rollout.jsonl`. **This is by far the longest
stage** — the hardest back-game positions take well over an hour each and run one
after another locally — but it is fully resumable, so you can stop and restart at
will.

After pass 3 the assembled benchmark is written to
`data/money_benchmark/benchmark.json`. (Running `build` with no `--stages` runs
all three passes in order.)

**4. Score a bot against it.**

```bash
python scripts/benchmark_money.py score --level 3ply --n-threads 16    # Sage 3P
python scripts/benchmark_money.py score --level truncated3             # Sage 3T
```

`--level` takes `1ply`–`4ply`, `truncated1`/`2`/`3` (= 1T/2T/3T) or `rollout`;
`--n-threads` scores positions concurrently. Decisions whose stored reference is
too coarse for how close they are (e.g. a not-yet-rolled-out position) are skipped
and reported, so a partially built data set still scores cleanly.

To score **XG**, batch-analyze the pass-1 `xg/*.txt` transcripts (with **Save
Games after analyze** checked) so each gets a matching `seed_<N>.xg`, then:

```bash
python scripts/benchmark_pr_xg.py
```

which reads XG's #1 decision per position and scores it against the same saved
reference equities, printing the same PR breakdown.

### Match Play

We repeated the Rollout PR experiment in match play, where the score on the board changes the value of every decision. A 5-point match is a good test case: the match score materially affects checker and cube decisions through these relatively short matches, so it exercises the engines' match-equity handling, not just their raw position evaluation.

#### Rollout PR Algorithm

We simulated 130 5-point matches of Sage 3P vs Sage 3P (both sides played by Sage at 3-ply). The match state — each player's away-count and the Crawford flag — is threaded through every evaluation, so all decisions, and the rolled-out "truth", are computed in match-equity (MWC) space against the correct score; cube decisions use the Kazaross-XG2 match equity table. Otherwise the method is identical to the money-game build: a first pass capturing 3-ply analytics for every decision, a second pass re-evaluating at Sage 3T any decision within 0.05 equity of its next-best alternative, and a third pass rolling out (1,296-path batches, 3-ply checker and cube decisions, variance reduction, repeated until the 95% equity band is under 0.005 or 20,736 paths) any decision still within 0.02 equity. The strongest tier reached for each decision is its benchmark truth.

For XG, we manually batch-analyzed the 130 match transcripts (3-ply decisions, upgrading to the listed eval level for disputes) — one `.xg` per match, each containing every game of the match — and scored XG's chosen decision against the same saved reference equities.

#### Rollout PR Results

There were 18,292 decisions across 17,892 positions. Of the 17,892 positions, 7,522 were settled at 3-ply; the other 10,370 were re-evaluated at 3T, of which 3,460 settled there and 6,910 were rolled out.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.19 | 0.16 | 0.40 | 0.01 | 0.17 | 0.18 | 0.25 | 0.26 |
| XG Roller ++ | 0.35 | 0.35 | 0.41 | 0.08 | 0.44 | 0.32 | 0.33 | 0.45 |
| Sage 2T | 0.26 | 0.24 | 0.44 | 0.02 | 0.36 | 0.18 | 0.34 | 0.33 |
| XG Roller + | 0.44 | 0.44 | 0.41 | 0.09 | 0.55 | 0.44 | 0.40 | 0.52 |
| Sage 1T | 0.49 | 0.49 | 0.52 | 0.05 | 0.57 | 0.44 | 0.57 | 0.63 |
| XG Roller | 0.51 | 0.51 | 0.54 | 0.09 | 0.64 | 0.50 | 0.51 | 0.62 |
| Sage 4P | 0.42 | 0.42 | 0.48 | 0.05 | 0.52 | 0.38 | 0.45 | 0.55 |
| XG 4-ply | 0.46 | 0.45 | 0.56 | 0.09 | 0.60 | 0.44 | 0.42 | 0.56 |
| Sage 3P | 0.57 | 0.56 | 0.67 | 0.05 | 0.74 | 0.56 | 0.58 | 0.68 |
| XG 3-ply | 0.54 | 0.53 | 0.61 | 0.09 | 0.69 | 0.53 | 0.52 | 0.65 |
| Sage 2P | 1.30 | 1.26 | 1.60 | 0.39 | 1.59 | 1.23 | 1.50 | 1.41 |
| Sage 1P | 2.30 | 2.29 | 2.31 | 0.69 | 2.48 | 2.43 | 2.47 | 2.56 |

As in money play, Sage's evaluations are stronger than the equivalent XG evaluation at every matched level except 3-ply, where the two are within noise (XG 0.54, Sage 0.57). The truncated-rollout levels that most users rely on — 3T and 2T — show Sage's clearest edge.

#### Running the Pipeline

The match data set is built by `scripts/benchmark_match.py`, the match-play twin of `benchmark_money.py`; the match length and number of matches are arguments, and the match state is threaded through every decision. The three passes are the same independently-resumable stages, run locally from a fresh `bgsage` checkout:

```bash
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass1 --workers 6   # simulate + capture 3P; one XG-import .txt per match
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass2 --n-threads 16  # re-evaluate close decisions at 3T
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass3 --n-threads 16  # roll out the closest decisions
```

The assembled benchmark is written to `data/match_benchmark/5pt/benchmark.json` (shipped as `benchmark.json.gz`). Score a bot, or XG, exactly as in the money case:

```bash
python scripts/benchmark_match.py score --match-length 5 --level truncated3   # Sage 3T
python scripts/benchmark_pr_xg_match.py --match-length 5                       # XG, from batch-analyzed .xg files
```

As with the money build, pass 3 is by far the longest stage and fully resumable; the hardest back-game and long-race positions take well over an hour each.

## Disputed Position Analysis

Another way to compare Open Sage against XG is to look at a range of realistic positions that come up in real games, find cases where Open Sage and XG disagree on a decision, then roll out those cases. We assume that the rollout results are "truth" and can then judge, for each position, whether Open Sage or XG was correct on their decision (or if neither was).

### Money Games

The first set of positions was generated for money games, with Jacoby and beavers on.

#### Comparison Algorithm

The algorithm we used to generate these positions was:
* Generate 200 simulated money games where Sage 3P plays itself. 3P is a modestly strong evaluation level that can be trusted to lead to a realistic distribution of backgammon positions, in many different game plans, through those simulated games.
* For each game, write out a .txt file that transcribes the game, and can be imported by XG. This is a standard file format used, for example, by Backgammon Galaxy when it exports games.
* _Manual step_: once the 200 .txt files have been generated, use XG's Batch Analysis function to analyze them. We set up a custom Analyze Level that did 4-ply XG decisions, but upgraded to XG Roller ++ for any cases where the Sage 3P decision didn't match the XG 4-ply decision. That generates 200 .xg files, one per game, that sit alongside the .txt files.
* For each of the .xg files, parse out the XG bot analytics, and identify positions where XG thinks Sage 3P made a decision error. At that point, re-evaluate the Sage decision using Sage 3T. If that matches the XG decision, then skip it - this is just a case where 3P was too weak, and we're trying to compare 3T vs Roller ++.
* For the decisions where Sage 3T and XG Roller ++ are still different, and the equity difference (as measured by XG's analysis of the decision Sage 3T made) is more than 0.02, roll out the decision in Sage. We did 5,184 paths and 3P decisions for the Sage rollout.

#### Money Game Results

There were 7,404 decisions in those 200 money games. Of those, there were 130 (1.8%) where Sage 3T and XG Roller ++ differed. The vast majority of those disputes were small differences: only 31 positions (24%) were larger than an 0.01 difference.

There were 11 examples where the equity difference was greater than 0.02. We rolled out those 11 examples to see which bot performed better on these disputed cases.

As compared against the Sage rollout, Sage 3T was correct on 6/11, XG ++ was correct on 2/11, and there were 3/11 cases where neither bot found the best decision. Averaged across those 11 positions, Sage 3T's average error was 0.005 and XG ++'s was 0.015. There may be some bias associated with using the Sage rollout as the "truth" for this analysis, but hand-checked XG rollouts show broad agreement with Sage rollouts.

On balance Sage 3T was noticeably better than XG Roller ++ in these disputed money game positions. However, in almost all decisions across the simulate games, the two agree closely, and on an overall basis the two evaluations are very close.

### Matches

To test the relative accuracy of the two bots in match play, we simulated 5-point matches. A 5-point match is a nice test case because the match score often materially affects decisions through these relatively short matches.

#### Comparison Algorithm

The algorithm we used to generate these match positions was:
* Generate 70 simulated 5-point matches where Sage 3P plays itself. 3P is a modestly strong evaluation level that can be trusted to lead to a realistic distribution of backgammon positions, in many different game plans, through those simulated games.
* For each match, write out a .txt file that transcribes the match, and can be imported by XG. This is a standard file format used, for example, by Backgammon Galaxy when it exports matches.
* _Manual step_: once the 70 .txt files have been generated, use XG's Batch Analysis function to analyze them. We set up a custom Analyze Level that did 4-ply XG decisions, but upgraded to XG Roller ++ for any cases where the Sage 3P decision didn't match the XG 4-ply decision. That generates 70 .xg files, one per match, that sit alongside the .txt files.
* For each of the .xg files, parse out the XG bot analytics, and identify positions where XG thinks Sage 3P made a decision error. At that point, re-evaluate the Sage decision using Sage 3T. If that matches the XG decision, then skip it - this is just a case where 3P was too weak, and we're trying to compare 3T vs Roller ++.
* For the decisions where Sage 3T and XG Roller ++ are still different, and the equity difference (as measured by XG's analysis of the decision Sage 3T made) is more than 0.02, roll out the decision in Sage. We did 5,184 paths and 3P decisions for the rollout.

#### Match Play Results

There were 10,160 decisions in those 70 5-point matches. Of those, there were 192 (1.9%) where Sage 3T and XG Roller ++ differed. The vast majority of those 192 disputes were small differences: only 19% (37 positions) were larger than an 0.01 difference.

There were 16 examples where the equity difference was greater than 0.02. We rolled out those 16 examples to see which bot performed better on these disputed cases.

As compared against the Sage rollout, Sage 3T was correct on 9/16, XG ++ was correct on 4/16, and there were 3/16 cases where neither bot found the best decision. Averaged across those 16 positions, Sage 3T's average error was 0.004 and XG ++'s was 0.013. There may be some bias associated with using the Sage rollout as the "truth" for this analysis, but hand-checked XG rollouts show broad agreement with Sage rollouts.

Sage looks noticeably stronger than XG in these disputes, getting more right and having a smaller average error. Still, these are a small fraction of all positions where the two bots dispute, and overall they are very close to equal.

### Running the Pipeline

Both experiments follow the same three-stage flow: simulate games with Sage 3P,
hand the transcripts to XG for Batch Analysis (manual), then run an aggregator
that finds the disputed positions and (optionally) rolls them out. All scripts
live in `bgsage/scripts/` and resolve their Python paths and the compiled
`bgbot_cpp.pyd` from inside the `bgsage/` repo.

#### Prerequisites

* Open Sage built locally.
* eXtreme Gammon installed on a Windows machine for the manual analysis step.

#### XG Batch Analysis settings (used for both experiments)

Set up a custom Analyze Level in XG with:

* **Move decisions:** XG 4-ply with an upgrade to XG Roller ++ whenever the
  played move differs from XG's 4-ply pick.
* **Cube decisions:** same — XG 4-ply with an upgrade to XG Roller ++ on
  disagreement.
* **Save Games after analyze:** checked. Without this XG prints summary
  statistics but doesn't write per-game `.xg` files, and the aggregator
  has nothing to read.

Point Batch Analysis at the folder of `.txt` transcripts; XG writes one
`.xg` file next to each `.txt`. Make sure to select the custom Analyze
level _after_ choosing the set of .txt files for analysis.

#### Money Games

**1. Simulate the games.**

```bash
python bgsage/scripts/run_sage_vs_sage_games.py 1 200 --level 3P --workers 6
```

Arguments: `initial_seed n_games`. Game `i` uses RNG seed `initial_seed + i`.
With `--workers 6`, six worker processes each pre-load their own 3P analyzer
at `parallel_threads=1` so the CPU isn't oversubscribed. One `.txt` per
game is written to `logs/sage_vs_sage/seed_<N>.txt` under the host project
root (the legacy default for this script).

**2. Batch-analyze in XG** (see settings above). Output: one `.xg` per
`.txt` in the same folder.

**3. Aggregate, find disputes, roll them out.**

```bash
python bgsage/scripts/aggregate_xg_pr.py --threshold 0.02 --rollout-threads 24 --n-trials 5184
```

The aggregator:

* Parses every `.xg` with `bgsage.xg_compare.parse_xg_game` and computes
  per-game PR using XG's own equities.
* Re-evaluates each Sage-3P-vs-XG disagreement at Sage 3T (the
  `--re-eval-level` argument, default `truncated3`). If 3T agrees with
  XG the disagreement vanishes; if it still disagrees the position is
  recorded as a `Dispute` whose `xg_measured_error` is XG's own
  measurement of Sage 3T's pick.
* For every `Dispute` whose `xg_measured_error` exceeds `--threshold`,
  runs a 5,184-trial Sage rollout at 3-ply throughout. Below-threshold
  disputes are skipped and assumed Sage-wrong (they contribute
  `+xg_measured_error` to the net Sage error).

Output files (alongside the `.xg` files):

| File | Purpose |
|---|---|
| `sage_3T_cache.jsonl` | 3T re-eval cache, appended incrementally; resumable across runs. |
| `rollout_disputes.jsonl` | One record per dispute (rolled out or skipped); also resumable. |

Re-running the aggregator skips already-completed rollouts and already-cached
3T evaluations, so iterating on thresholds or limits is cheap.

#### Matches

**1. Simulate the matches.**

```bash
python bgsage/scripts/run_sage_vs_sage_match.py 5 70 --level 3P --workers 6
```

Arguments: `match_length n_matches`. Match `i` uses RNG seed
`initial_seed + i` (default `--initial-seed 1`). The simulator threads
`away1`/`away2`/`is_crawford` through every analyzer call, suppresses cube
offers in the Crawford game, and caps each game's point award at the
remaining match length so excess points aren't recorded. One `.txt` per
match is written to `bgsage/logs/sage_vs_sage_match/match_seed_<N>.txt`
(inside the bgsage repo).

**2. Batch-analyze in XG** with the same custom Analyze Level as above.
XG writes one `.xg` per match — each `.xg` internally contains a
`HEADER_MATCH` record, then one `HEADER_GAME ... FOOTER_GAME` block per
game in the match.

**3. Aggregate, find disputes, roll them out.**

```bash
python bgsage/scripts/aggregate_xg_match_pr.py --threshold 0.02 --rollout-threads 24 --n-trials 5184
```

This is the match-aware analog of `aggregate_xg_pr.py`. Differences from
the money-game version:

* Uses `bgsage.xg_compare.parse_xg_match` to extract the per-game start
  scores and Crawford flag from each `.xg`.
* Computes per-game `away1`/`away2`/`is_crawford` and threads them
  through dispute detection, the 3T re-eval, and the rollouts so every
  evaluation is done in MWC space against the right match score.
* Cache key includes the game number, so match state changing between
  games of the same `.xg` doesn't cause cache collisions.

Output files have the same names and locations (alongside the `.xg`
files in `bgsage/logs/sage_vs_sage_match/`).

#### Useful flags

Both aggregators accept the same set of optional flags.

| Flag | What it does |
|---|---|
| `--skip-rollouts` | Stop after PR aggregation and dispute detection. Fast feedback loop for "how many disputes did Sage 3T have with XG?" |
| `--dry-run` | Run dispute detection and report how many rollouts would be triggered at the current `--threshold`, but don't actually roll out or write to `rollout_disputes.jsonl`. |
| `--top-disputes N` | After PR aggregation, print the top N disputes by `xg_measured_error` with their move notations / cube actions. Pairs well with `--skip-rollouts` for a quick look at the biggest disagreements. |
| `--threshold T` | Below-T disputes are skipped (assumed Sage-wrong). Default `0.005`. Higher thresholds trade rollout compute for more pessimistic skip-assumptions. The match aggregator always prints a threshold breakdown table so you can see the tradeoff before committing. |
| `--re-eval-level LEVEL` | `truncated2` or `truncated3` (default). The level Sage uses to re-evaluate a position before declaring a Dispute. |
| `--rollout-threads N` | Threads per rollout (default 0 = auto-detect cores). |
| `--n-trials N` | Paths per rollout (default 1,296). Pass `--n-trials 5184` for the 5,184-path runs. Accepted by both aggregators. |
| `--limit N` | Roll out only the first N pending disputes. Useful for sampling. |

#### Replicating the exact published numbers

The published runs used `--threshold 0.02 --re-eval-level truncated3
--n-trials 5184` — 5,184 trials (4×1,296), no truncation, 3-ply throughout,
`ultra_late_threshold=9999`. (`--n-trials` defaults to 1,296, so it must be passed
explicitly to reproduce the 5,184-path rollouts.) Wall-clock time was several hours per experiment
on a single workstation; the longest part is the per-position rollout phase
(~10 min each), and that scales linearly with the number of disputes above
threshold.

## Match PR Agreement on Real Matches

The two analyses above measure *strength* — how close each engine's decisions are to a rolled-out truth. A third, equally practical question matters to anyone who uses an engine to study their own play: **if you analyze a real match in XG, note your Performance Rating, then analyze the same match in Sage, how close are the two PRs?** A player who has spent years building intuition for what a given PR means in XG should get essentially the same number from Sage.

To test this directly, we took a large set of real tournament matches that had already been analyzed in XG, re-analyzed every one from scratch in Sage, and compared the Performance Rating each engine assigned to each player.

### Evaluation Settings

Each match was re-analyzed in Sage at a **3-ply base**, with an **expert 3T pass** (a 360-path truncated rollout) applied to the decisions where the player's actual move disagreed with the 3-ply best. This mirrors how a strong XG analysis works — a base ply for the clear decisions, escalating to a truncated rollout for the close ones — and the two levels are matched in strength: **Sage 3P ≈ XG 3-ply** and **Sage 3T ≈ XG Roller ++** (the same level pairs compared in the studies above). Each engine then computes a PR per player from its own evaluations and its own decision counting — exactly what a user sees in each app.

### The Matches

The match files come from three 2026 tournaments — **UBC Texas**, **UBC Istanbul**, and **UBC Japan** — all 7-point matches, analyzed in XG and generously provided by **Máté Fehér**.

| Event | Matches |
|---|---:|
| UBC Texas 2026 | 100 |
| UBC Istanbul 2026 | 146 |
| UBC Japan 2026 | 44 |
| **Total** | **290** |

That is 290 matches and 580 individual player ratings. (One further match was set aside as a corrupted transcription.)

### Results

For each player in each match we have two Performance Ratings — XG's and Sage's — and their difference. Pooling all 580 player ratings:

| Per-player PR | XG | Sage | **Difference (Sage − XG)** |
|---|---:|---:|---:|
| Average | 4.36 | 4.36 | **+0.002** |
| Standard deviation | 2.08 | 2.10 | **0.37** |
| 95% range | 1.52 – 9.36 | 1.44 – 9.67 | **−0.76 – +0.74** |

The two engines agree almost exactly. The **average difference is +0.002 PR** — statistically indistinguishable from zero (95% confidence interval ±0.03; *p* = 0.90). The standard deviation of the difference (**0.37**) is small next to the spread in PR itself (**2.08**), so the disagreement on any single rating is minor relative to how much PR naturally varies from player to player and match to match. The two engines' per-player ratings correlate at **r = 0.98**.

In practical terms: a player who analyzes a match in Sage will, in the large majority of cases, see essentially the same Performance Rating that XG would give. As a measure of how well a match was played, the two engines are interchangeable.

## Conclusion

Open Sage and XG are close at every matched evaluation level. In the Rollout PR study — now run for both money play and 5-point match play — Sage's evaluations score at least as well as the equivalent XG evaluation at every level except 3-ply, where the two are within noise. Sage's edge is clearest at the truncated-rollout levels (3T and 2T) and holds in both money and match play, and the Disputed Positions study — which rolls out only the positions where the two engines actually disagree — points the same way. The differences are small, and some of the apparent edge may be bias from using Sage rollouts instead of XG rollouts as the truth.

And on real matches, the two engines assign nearly identical Performance Ratings: across 290 tournament matches, the average difference between a player's Sage PR and XG PR is statistically indistinguishable from zero. Whether the test is strength against a rolled-out truth or simple agreement on how a real game was played, Open Sage and XG land in the same place.
