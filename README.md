# The Open Sage Backgammon Bot Engine Library

Open Sage is a "backgammon bot engine" - that is, a library that you can ask questions like "what's the best
move in this situation, with all the analytics to back it up", or "is this a double, and if so, is it a take or a pass 
(including analytics)?".

## What it Contains

It is a joint Python/C++ library that includes:
* A neural network-based backgammon bot.
* Neural network training framework using both self play and supervised learning, including training code, benchmark scoring, with customizable inputs. Uses your NVIDIA GPU (via CUDA) if you have one.
* The post-training weights for several different versions of the bot engine.
* Multi-ply and rollout calculations that efficiently parallelize on the CPU.
* Test framework.
* VIBING.md: information on how to use Claude Code-style tools to interact with and change it, and how to submit changes back to us (the maintainers). Most of this code was written by Anthropic's Claude Code and OpenAI's Codex.

## What Evaluation Levels Does It Support?

The library supports "multi-ply" lookahead calculations. 1-ply is the raw neural network evaluation (we follow the XG/eXtreme Gammon
numbering convention; GNUbg calls this "0-ply"). Adding a ply makes the calculation roughly 20x slower. It efficiently parallelizes
these multi-ply calculations on your CPUs.

It supports truncated rollout calculations, where it simulates the game several turns into the future and then stops the simulation, using bot evaluations at the leaves.

It also supports full rollout calculations, which are simulations playing out the game over and over to completion. 

Truncated and full rollouts both include variance reduction and efficiently parallelize on CPUs.

## What are Its Interfaces?

It offers both Python and C++ interfaces for:
* Checker play analytics: given a list of checker positions, the two dice, and cube information, it returns you a list of information about the top possible moves, sorted in descending order of equity; for each it gives you equity and cubeless post-move probabilities. You can specify the evaluation level.
* Post-move position analytics: given a list of checker positions and the cube information, it returns you cubeful equity, cubeless equity, and the cubeless probabilities - for a post-move position (right before the opponent's turn).
* Cube action analytics: given a list of checker positions and the cube information, it returns you cubeful equity information about the three states (ND, D/T, and D/P), cubeless equity, and the cubeless probabilities - for a pre-roll position.
* Game plan classification: given a list of checkers, it returns the optimal game plans of the player and the opponent.
* Game utilities (flip a board, etc).

## How Does it Compare to XG?

eXtreme Gammon (XG) is a commercial backgammon analysis application that uses its own proprietary bot engine. That XG bot engine is not directly exposed via API for non-commercial use - just the Windows desktop application. XG is an important standard for backgammon bot analysis strength and speed.

The most direct way to compare the Open Sage bot engine against XG's would be to play many thousands, or tens of thousands, of head to head games to pick the signal out of the dice noise. Unfortunately, since XG has no API, this must be done by hand, which would take too long.

Another slightly more indirect way is to have XG analyze Open Sage's play. The approach: run many separate money games where the Open Sage bot plays itself; for each game, write out the list of plays to a file that XG can import; when those files are written, use XG's Batch Analyze function to analyze them (at World Class level, which corresponds roughly to XG Roller + evaluation strength) and write out per-game XG files next to the text files; then pull the XG analytics out of those files to see how XG scores Open Sage's decisions. CLAUDE.md has details on which scripts to use for this.

The result: Open Sage, using 3-ply evaluation strength, scored a PR of 0.39 across 200 money games. That is very close to identical play. In addition, when examining individual positions where Open Sage and XG are different, it is unclear whether Open Sage or XG is actually correct - there are examples of both - which suggests that Open Sage and XG are performing comparably on this aggregate basis.
