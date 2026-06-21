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

We looked at two approaches:

* Rollout PR: we simulated money games and match play over many games, rolled out the closest decisions, and scored bot decisions against these rolled out results, and ended up with a Performance Rating (PR) against the rollout truth. We store these benchmark decision results. Then we run each decision by a candidate bot and ask it to give its decision, and score its result against the benchmark equities.
* Disputed Positions: we simulated money games and match play over many games and found the subset of positions where Sage 3T and XG Roller ++ differed on their decision. We rolled those out to see which bot got closer to the truth.

## Rollout PR Analysis

This is similar in approach to the analysis done on XG (and a number of other bots) in 2012: https://www.extremegammon.com/studies.aspx.

### Money Games

#### Rollout PR Algorithm

We simulated 500 money games of Sage 3P vs Sage 3P. We ran through all the decisions, and did a second pass, re-evaluating any decisions at Sage 3T where the best decision was within 0.05 equity of the next best decision. We then did a third pass, rolling out any decisions which Sage 3T evaluated as within 0.02 equity of the next best decision. We saved out all those results and counted them as the "true" decision results, against which we can benchmark any bot's decisions.

For rollouts we used Open Sage rollouts with 3P decisions for checker play and cube actions. We ran batches of 1,296 paths until the 95% accuracy range on the equity was less than 0.005, or it did 20,736 (=16 times 1,296) paths.

For a given bot (and evaluation level), we had the bot evaluate its decision for each one of those benchmark decisions, and scored it against the benchmark truth. We calculated a Performance Rating (PR) as the average error (as measured against the benchmark equities) multiplied by 500. We also broke out the results into checker play and cube action PRs.

For XG results, we manually ran XG's Batch Analyze on the 500 individual game files, then automatically parsed the XG decisions from the .xg files it generates (one per game). The Batch Analyze settings were 3-ply decisions, moving to the listed eval level for disputes.

#### Rollout PR Results

There were 17,535 decisions total. 9,237 were re-evaluated at 3T. 5,944 positions were rolled out. Some rollouts were very quick, while the slowest took well over an hour to roll out on a machine with 16 cores.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.21 | 0.18 | 0.36 | 0.02 | 0.24 | 0.17 | 0.31 | 0.28 |
| XG Roller ++ | 0.32 | 0.31 | 0.38 | 0.04 | 0.40 | 0.24 | 0.41 | 0.44 |
| Sage 2T | 0.36 | 0.35 | 0.44 | 0.39 | 0.38 | 0.25 | 0.49 | 0.39 |
| XG Roller + | 0.41 | 0.41 | 0.39 | 0.05 | 0.59 | 0.31 | 0.47 | 0.54 |
| Sage 1T | 0.50 | 0.52 | 0.40 | 0.04 | 0.57 | 0.43 | 0.59 | 0.73 |
| XG Roller | 0.53 | 0.54 | 0.48 | 0.05 | 0.63 | 0.44 | 0.71 | 0.66 |
| Sage 4P | 0.41 | 0.39 | 0.50 | 0.07 | 0.51 | 0.37 | 0.45 | 0.53 |
| XG 4-ply | 0.46 | 0.45 | 0.52 | 0.06 | 0.58 | 0.40 | 0.57 | 0.58 |
| Sage 3P | 0.58 | 0.58 | 0.57 | 0.14 | 0.72 | 0.52 | 0.63 | 0.74 |
| XG 3-ply | 0.57 | 0.57 | 0.58 | 0.05 | 0.71 | 0.48 | 0.73 | 0.71 |
| Sage 2P | 1.64 | 1.39 | 2.88 | 0.40 | 1.77 | 1.83 | 1.86 | 1.71 |
| Sage 1P | 2.59 | 2.48 | 3.20 | 0.78 | 2.60 | 2.79 | 3.10 | 2.89 |

Sage evaluations are stronger than their equivalent XG evaluations in every case except 3-ply, where XG is slightly stronger, but the two are very close.

### Running the Pipeline

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

## Conclusion

Open Sage 3T evaluations and XG Roller ++ evaluations are roughly comparable. There is some evidence that Sage 3T is slightly stronger than XG Roller ++, especially in match play, but the differences are small, and the relative strength might just be a bias due to using Sage rollouts instead of XG rollouts.
