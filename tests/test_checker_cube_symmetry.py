# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Tests for the symmetry between checker_play and cube_action analytics.

For any post-move position M produced by a checker play, the cubeful equity
reported by `analyzer.checker_play(...).moves[i]` must equal the negative of
`analyzer.cube_action(flip(M), ...).optimal_equity` at the same eval level —
because they're computing the same thing from inverted perspectives. The
multi-ply path enforces this via `cubeful_equity_nply`; the rollout path
enforces it via the C++ `cubeful_rollout_position`, which delegates to
`cubeful_cube_decision` on the flipped position.

Catches regressions where one path forgets to apply the opponent's cube
decision (e.g. returning -ND instead of -optimal for moves that leave a drop
position).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)

import pytest

from bgsage import BgBotAnalyzer, flip_board


# Position the user reported: money game, centered cube, dice 1-1. At 3T the
# opponent's optimal action for this position (after most plays) is D/P, so a
# correct checker_play must report cubeful equity = -1.0 for the move 3/1(2).
# The board is from the player-on-roll's perspective.
USER_REPORTED_BOARD = [
    0, 0, 0, 2, -2, 3, 3, 2, 2, 2, -1, 0, 0,
    0, 0, 0, -1, 0, -2, -2, -2, -2, -1, -2, 1, 0,
]
USER_REPORTED_DICE = (1, 1)

# 3/1(2) post-move board (two checkers moved from the 3 to the 1 point).
MOVE_3_1_2_POST_BOARD = [
    0, 2, 0, 0, -2, 3, 3, 2, 2, 2, -1, 0, 0,
    0, 0, 0, -1, 0, -2, -2, -2, -2, -1, -2, 1, 0,
]


def _opp_after(post_move_board):
    """Flip a post-move board to the opponent's pre-roll perspective."""
    return flip_board(post_move_board)


def _check_symmetry(analyzer, board, die1, die2, *, tol):
    """Best move's cubeful equity must equal -opp_cube_action.optimal_equity.

    Returns the (checker_cubeful, opp_optimal, diff) triple for diagnostics.
    """
    cp = analyzer.checker_play(
        board, die1, die2,
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    assert cp.moves, "Expected at least one legal move"
    best = cp.moves[0]

    ca = analyzer.cube_action(
        _opp_after(best.board),
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    diff = abs(best.equity - (-ca.optimal_equity))
    assert diff < tol, (
        f"checker_play cubeful equity ({best.equity:+.4f}) does not match "
        f"-cube_action.optimal_equity ({-ca.optimal_equity:+.4f}); diff={diff:.4f}. "
        f"Best move post-board: {best.board}; opp action: {ca.optimal_action}, "
        f"ND={ca.equity_nd:+.4f} DT={ca.equity_dt:+.4f} DP={ca.equity_dp:+.4f}"
    )
    return best.equity, ca.optimal_equity, diff


# ---------------------------------------------------------------------------
# Multi-ply paths (already worked pre-fix; included for regression coverage).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["2ply", "3ply"])
def test_multi_ply_best_move_symmetry(level):
    """Multi-ply checker_play's best-move cubeful must match opp cube_action."""
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True)
    _check_symmetry(
        analyzer, USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        # Multi-ply uses the same cubeful_equity_nply for both paths, so this
        # is an exact equality up to floating-point noise.
        tol=1e-3,
    )


# ---------------------------------------------------------------------------
# Rollout paths (the regression the bug fix targets).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["truncated2", "truncated3"])
def test_rollout_best_move_symmetry(level):
    """Truncated rollout checker_play's best-move cubeful must match opp cube_action.

    Both paths run the same `cubeful_cube_decision` machinery (cubeful_rollout_position
    is a thin wrapper) with the same seeded dice, so the values should match
    exactly (no Monte Carlo noise across the two calls).
    """
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True)
    _check_symmetry(
        analyzer, USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        tol=1e-3,
    )


def test_3T_specific_move_3_1_2_is_drop():
    """User's specific complaint: at 3T, the move 3/1(2) leaves opp with a clear
    D/P. The cubeful equity for 3/1(2) must therefore be exactly -1.0 (equal to
    -DP_money = -1.0). Pre-fix this returned ~-0.79 (the ND value)."""
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True)
    cp = analyzer.checker_play(
        USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    match = next((m for m in cp.moves if m.board == MOVE_3_1_2_POST_BOARD), None)
    assert match is not None, "Expected the 3/1(2) move in the candidate list"
    # Tolerance is loose: cube_decision_1ply's beaver / Janowski math is
    # exact, and DP is exactly +1.0 in money, so -optimal_equity is exactly
    # -1.0. Allow a tiny epsilon for fp arithmetic.
    assert abs(match.equity - (-1.0)) < 1e-3, (
        f"3/1(2) cubeful equity at 3T is {match.equity:+.4f}, expected -1.0000. "
        f"Pre-fix value was ~-0.7880 (= -ND, ignoring opp's D/P). "
        f"Post-move board: {match.board}"
    )


def test_3T_3_1_2_matches_cube_action_on_flipped_board():
    """Direct symmetry check for the specific move 3/1(2) at 3T.

    Independent of which move ends up "best", the cubeful equity for 3/1(2)
    in the checker_play result must equal -cube_action.optimal_equity on its
    flipped post-move board.
    """
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True)
    cp = analyzer.checker_play(
        USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    match = next((m for m in cp.moves if m.board == MOVE_3_1_2_POST_BOARD), None)
    assert match is not None, "Expected the 3/1(2) move in the candidate list"

    ca = analyzer.cube_action(
        _opp_after(MOVE_3_1_2_POST_BOARD),
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    diff = abs(match.equity - (-ca.optimal_equity))
    assert diff < 1e-3, (
        f"3/1(2) checker_play cubeful = {match.equity:+.4f}, "
        f"-opp cube_action optimal = {-ca.optimal_equity:+.4f}, diff={diff:.4f}"
    )
