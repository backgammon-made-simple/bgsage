# Comparing the Open Sage Bot Engine to XG's Bot Engine

XG (eXtreme Gammon) is a standard reference for backgammon analysis, and its bot engine is considered a very strong one.

XG does not include an API to run its bot engine programmatically, so we could not run head-to-head games between Open Sage and XG at the scale required to identify small differences between them.

However, we did settle on an approach to test Open Sage against XG at scale using XG's Batch Analysis function.

Our goal was to compare the relative strengths of Open Sage's 3T evaluation and XG's Roller ++ evaluation, which are designed to give similar levels of accuracy. Both are truncated rollouts with 360 paths and incorporating variance reduction.

## Money Games

To compare Open Sage against XG we want to look at a range of realistic positions that come up in real games, find cases where Open Sage and XG disagree on a decision, then roll out those cases. We assume that the rollout results are "truth" and can then judge, for each position, whether Open Sage or XG was correct on their decision (or if neither was).

The first set of positions was generated for money games, with Jacoby and beavers on.

### Comparison Algorithm

The algorithm we used to generate these positions was:
* Generate 200 simulated money games where Sage 3P plays itself. 3P is a modestly strong evaluation level that can be trusted to lead to a realistic distribution of backgammon positions, in many different game plans, through those simulated games.
* For each game, write out a .txt file that transcribes the game, and can be imported by XG. This is a standard file format used, for example, by Backgammon Galaxy when it exports games.
* _Manual step_: once the 200 .txt files have been generated, use XG's Batch Analysis function to analyze them. We set up a custom Analyze Level that did 4-ply XG decisions, but upgraded to XG Roller ++ for any cases where the Sage 3P decision didn't match the XG 4-ply decision. That generates 200 .xg files, one per game, that sit alongside the .txt files.
* For each of the .xg files, parse out the XG bot analytics, and identify positions where XG thinks Sage 3P made a decision error. At that point, re-evaluate the Sage decision using Sage 3T. If that matches the XG decision, then skip it - this is just a case where 3P was too weak, and we're trying to compare 3T vs Roller ++.
* For the decisions where Sage 3T and XG Roller ++ are still different, and the equity difference (as measured by XG's analysis of the decision Sage 3T made) is more than 0.02, roll out the decision in Sage. We did 1,296 paths and 3P decisions for the rollout.

### Money Game Results

There were 7,348 decisions in those 200 money games. Of those, there were 138 (1.9%) where Sage 3T and XG Roller ++ differed. The vast majority of those 138 disputes were small differences: only 29% (40 positions) were larger than an 0.01 difference.

There were 16 examples where the equity difference was greater than 0.02. We rolled out those 16 examples to see which bot performed better on these disputed cases.

As compared against the Sage rollout, Sage 3T was correct on 7/16, XG ++ was correct on 5/16, and there were 4/16 cases where neither bot found the best decision. Averaged across those 16 positions, Sage 3T's average error was 0.0113 and XG ++'s was 0.0157. There may be some bias associated with using the Sage rollout as the "truth" for this analysis, but it was possible to automate the Sage rollouts, and it was not possible to automate XG rollouts.

On balance it seems like Sage 3T was roughly equal to XG Roller ++ in these disputed money game positions. Perhaps Sage 3T is a little stronger, but the difference is quite small.

## Matches

To test the relative accuracy of the two bots in match play, we simulated 5-point matches. A 5-point match is a nice test case because the match score often materially affects decisions through these relatively short matches.

### Comparison Algorithm

The algorithm we used to generate these match positions was:
* Generate 70 simulated 5-point matches where Sage 3P plays itself. 3P is a modestly strong evaluation level that can be trusted to lead to a realistic distribution of backgammon positions, in many different game plans, through those simulated games.
* For each match, write out a .txt file that transcribes the match, and can be imported by XG. This is a standard file format used, for example, by Backgammon Galaxy when it exports matches.
* _Manual step_: once the 70 .txt files have been generated, use XG's Batch Analysis function to analyze them. We set up a custom Analyze Level that did 4-ply XG decisions, but upgraded to XG Roller ++ for any cases where the Sage 3P decision didn't match the XG 4-ply decision. That generates 70 .xg files, one per match, that sit alongside the .txt files.
* For each of the .xg files, parse out the XG bot analytics, and identify positions where XG thinks Sage 3P made a decision error. At that point, re-evaluate the Sage decision using Sage 3T. If that matches the XG decision, then skip it - this is just a case where 3P was too weak, and we're trying to compare 3T vs Roller ++.
* For the decisions where Sage 3T and XG Roller ++ are still different, and the equity difference (as measured by XG's analysis of the decision Sage 3T made) is more than 0.02, roll out the decision in Sage. We did 1,296 paths and 3P decisions for the rollout.

### Match Play Results

There were 9,033 decisions in those 70 5-point matches. Of those, there were 257 (2.8%) where Sage 3T and XG Roller ++ differed. The vast majority of those 257 disputes were small differences: only 28% (72 positions) were larger than an 0.01 difference.

There were 38 examples where the equity difference was greater than 0.02. We rolled out those 38 examples to see which bot performed better on these disputed cases.

As compared against the Sage rollout, Sage 3T was correct on 19/38, XG ++ was correct on 11/38, and there were 8/38 cases where neither bot found the best decision. Averaged across those 38 positions, Sage 3T's average error was 0.0105 and XG ++'s was 0.0198. There may be some bias associated with using the Sage rollout as the "truth" for this analysis, but it was possible to automate the Sage rollouts, and it was not possible to automate XG rollouts.

Sage looks noticeably stronger than XG in these disputes, getting more right and having a smaller average error. Still, these are a small fraction of all positions where the two bots dispute, and overall they are very close to equal.

## Running the Pipeline

Both experiments follow the same three-stage flow: simulate games with Sage 3P,
hand the transcripts to XG for Batch Analysis (manual), then run an aggregator
that finds the disputed positions and (optionally) rolls them out. All scripts
live in `bgsage/scripts/` and resolve their Python paths and the compiled
`bgbot_cpp.pyd` from inside the `bgsage/` repo.

### Prerequisites

* Open Sage built locally.
* eXtreme Gammon installed on a Windows machine for the manual analysis step.

### XG Batch Analysis settings (used for both experiments)

Set up a custom Analyze Level in XG with:

* **Move decisions:** XG 4-ply with an upgrade to XG Roller ++ whenever the
  played move differs from XG's 4-ply pick.
* **Cube decisions:** same — XG 4-ply with an upgrade to XG Roller ++ on
  disagreement.
* **Save Games after analyze:** checked. Without this XG prints summary
  statistics but doesn't write per-game `.xg` files, and the aggregator
  has nothing to read.

Point Batch Analysis at the folder of `.txt` transcripts; XG writes one
`.xg` file next to each `.txt`.

### Money Games

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
python bgsage/scripts/aggregate_xg_pr.py --threshold 0.02 --rollout-threads 24
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
  runs a 1,296-trial Sage rollout at 3-ply throughout. Below-threshold
  disputes are skipped and assumed Sage-wrong (they contribute
  `+xg_measured_error` to the net Sage error).

Output files (alongside the `.xg` files):

| File | Purpose |
|---|---|
| `sage_3T_cache.jsonl` | 3T re-eval cache, appended incrementally; resumable across runs. |
| `rollout_disputes.jsonl` | One record per dispute (rolled out or skipped); also resumable. |

Re-running the aggregator skips already-completed rollouts and already-cached
3T evaluations, so iterating on thresholds or limits is cheap.

### Matches

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
python bgsage/scripts/aggregate_xg_match_pr.py --threshold 0.02 --rollout-threads 24
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

### Useful flags

Both aggregators accept the same set of optional flags.

| Flag | What it does |
|---|---|
| `--skip-rollouts` | Stop after PR aggregation and dispute detection. Fast feedback loop for "how many disputes did Sage 3T have with XG?" |
| `--dry-run` | Run dispute detection and report how many rollouts would be triggered at the current `--threshold`, but don't actually roll out or write to `rollout_disputes.jsonl`. |
| `--top-disputes N` | After PR aggregation, print the top N disputes by `xg_measured_error` with their move notations / cube actions. Pairs well with `--skip-rollouts` for a quick look at the biggest disagreements. |
| `--threshold T` | Below-T disputes are skipped (assumed Sage-wrong). Default `0.005`. Higher thresholds trade rollout compute for more pessimistic skip-assumptions. The match aggregator always prints a threshold breakdown table so you can see the tradeoff before committing. |
| `--re-eval-level LEVEL` | `truncated2` or `truncated3` (default). The level Sage uses to re-evaluate a position before declaring a Dispute. |
| `--rollout-threads N` | Threads per rollout (default 0 = auto-detect cores). |
| `--limit N` | Roll out only the first N pending disputes. Useful for sampling. |

### Replicating the exact published numbers

The published runs used `--threshold 0.02 --re-eval-level truncated3` with the
default rollout configuration (1,296 trials, no truncation, 3-ply throughout,
`ultra_late_threshold=9999`). Wall-clock time was several hours per experiment
on a single workstation; the longest part is the per-position rollout phase
(~10 min each), and that scales linearly with the number of disputes above
threshold.

## Conclusion

Open Sage 3T evaluations and XG Roller ++ evaluations are roughly comparable. There is some evidence that Sage 3T is slightly stronger than XG Roller ++, especially in match play, but the differences are small, and the relative strength might just be a bias due to using Sage rollouts instead of XG rollouts.
