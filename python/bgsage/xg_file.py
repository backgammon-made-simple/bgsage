# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Low-level read/write access to eXtreme Gammon (.xg / .xgp) files.

``xg_compare.py`` is a read-only parser oriented at extracting *turns* for PR
scoring. This module works one level lower — the raw archive container and
fixed-size records — and adds the pieces needed to drive XG's **Batch Rollout**
feature programmatically:

- read AND re-write the ZLibArchive container (so a modified ``temp.xg`` can be
  saved back into a valid .xg file that XG will open),
- set/clear the batch-rollout marks (``TimeDelayMove`` / ``TimeDelayCube``,
  the v26 "marked for later RO" fields) on move and cube records,
- parse the ``temp.xgr`` rollout stream (``TRolloutContext``, one 2184-byte
  record per stored rollout) that XG writes after a Batch Rollout completes.

Format authority: the official spec published at extremegammon.com/xgformat.aspx
(``XG_format.pas`` + ``ZLibArchive.pas``). All offsets below are derived from
those Pascal declarations under Delphi record-alignment rules and have been
verified against real batch-analyzed .xg files (see tests/test_xg_file.py).

Archive layout of a .xg file:

    [TRichGameHeader: 8232 bytes]['RGMH' magic, thumbnail sizes ...]
    [thumbnail JPG]
    [file data blobs]            <- zlib-compressed entries (temp.xg, temp.xgi, ...)
    [registry]                   <- zlib-compressed array of 532-byte entries
    [archive trailer: 36 bytes]  <- CRC32 over blobs+registry, counts, sizes

``temp.xg`` is a flat sequence of 2560-byte ``TSaveRec`` records; ``temp.xgi``
is a fast-access copy of the first and last record; ``temp.xgr`` holds the
rollout contexts referenced by ``RolloutindexM`` / ``RolloutindexD``.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from .board import flip_board

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RICH_HEADER_SIZE = 8232
MAGIC_RGMH = 0x484D4752  # 'RGMH'
MAGIC_DMLI = 0x494C4D44  # 'DMLI'

TSAVEREC_SIZE = 2560
ROLLOUT_CONTEXT_SIZE = 2184
ARCHIVE_TRAILER_SIZE = 36
REGISTRY_ENTRY_SIZE = 532

TS_HEADER_MATCH = 0
TS_HEADER_GAME = 1
TS_CUBE = 2
TS_MOVE = 3
TS_FOOTER_GAME = 4
TS_FOOTER_MATCH = 5

# XG PLAYERLEVEL table (XG_format.pas): used by per-move EvalLevel.Level,
# EngineStructDoubleAction.Level/LevelRequest and AnalyzeM/AnalyzeC.
PLAYER_LEVEL_LABELS = {
    -1: "none",
    0: "1ply", 1: "2ply", 2: "3ply", 3: "4ply", 4: "5ply", 5: "6ply", 6: "7ply",
    12: "3ply_red",
    100: "rollout",
    998: "book", 999: "book",
    1000: "xgroller", 1001: "xgroller+", 1002: "xgroller++",
}


def player_level_label(level: int) -> str:
    return PLAYER_LEVEL_LABELS.get(level, f"level{level}")


# ---------------------------------------------------------------------------
# Archive container
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    """One file inside the ZLibArchive. ``blob`` holds the on-disk (compressed)
    bytes so an unmodified entry round-trips byte-for-byte."""
    name: str
    osize: int
    crc: int          # CRC32 of the *uncompressed* content, signed int32
    blob: bytes       # compressed bytes as stored in the archive
    compressed: bool  # Status field: fsCompressed(0) / fsStored(1)

    def data(self) -> bytes:
        return zlib.decompress(self.blob) if self.compressed else self.blob


def _crc32_signed(data: bytes) -> int:
    return struct.unpack('<i', struct.pack('<I', zlib.crc32(data) & 0xFFFFFFFF))[0]


class XgArchive:
    """A .xg/.xgp file: verbatim prefix (RichGameHeader + thumbnail) + entries."""

    def __init__(self, prefix: bytes, entries: list[ArchiveEntry]):
        self.prefix = prefix
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> "XgArchive":
        raw = Path(path).read_bytes()
        if len(raw) < RICH_HEADER_SIZE + ARCHIVE_TRAILER_SIZE:
            raise ValueError(f"{path}: too small to be a valid XG file")
        magic = struct.unpack_from('<I', raw, 0)[0]
        if magic != MAGIC_RGMH:
            raise ValueError(f"{path}: bad magic 0x{magic:08X}, expected RGMH")

        trailer_off = len(raw) - ARCHIVE_TRAILER_SIZE
        crc, file_count, _version, reg_size, data_size, comp_reg_i = \
            struct.unpack_from('<iiiiii', raw, trailer_off)
        reg_start = trailer_off - reg_size
        data_start = reg_start - data_size
        if data_start < RICH_HEADER_SIZE:
            raise ValueError(f"{path}: invalid archive layout")

        calc = _crc32_signed(raw[data_start:trailer_off])
        if calc != crc:
            raise ValueError(f"{path}: archive CRC mismatch ({crc} != {calc})")

        reg_raw = raw[reg_start:trailer_off]
        index = zlib.decompress(reg_raw) if (comp_reg_i & 0xFF) else reg_raw

        entries = []
        for i in range(file_count):
            rec = index[i * REGISTRY_ENTRY_SIZE:(i + 1) * REGISTRY_ENTRY_SIZE]
            name = rec[1:1 + rec[0]].decode('ascii', errors='replace')
            osize, csize, start, fcrc = struct.unpack_from('<iiii', rec, 512)
            compressed = (rec[528] == 0)
            blob = raw[data_start + start:data_start + start + csize]
            entries.append(ArchiveEntry(name, osize, fcrc, blob, compressed))
        return cls(raw[:data_start], entries)

    def get(self, name: str) -> bytes | None:
        for e in self.entries:
            if e.name == name:
                return e.data()
        return None

    def set(self, name: str, data: bytes) -> None:
        """Replace (or append) an entry's content; recompresses and re-CRCs."""
        entry = ArchiveEntry(name, len(data), _crc32_signed(data),
                             zlib.compress(data), True)
        for i, e in enumerate(self.entries):
            if e.name == name:
                self.entries[i] = entry
                return
        self.entries.append(entry)

    def save(self, path: str | Path) -> None:
        blobs = bytearray()
        registry = bytearray()
        for e in self.entries:
            start = len(blobs)
            blobs += e.blob
            rec = bytearray(REGISTRY_ENTRY_SIZE)
            name_b = e.name.encode('ascii')
            rec[0] = len(name_b)
            rec[1:1 + len(name_b)] = name_b
            # path shortstring at 256 stays empty
            struct.pack_into('<iiii', rec, 512, e.osize, len(e.blob), start, e.crc)
            rec[528] = 0 if e.compressed else 1
            rec[529] = 2  # fcDefault compression level
            registry += rec

        reg_comp = zlib.compress(bytes(registry))
        stream = bytes(blobs) + reg_comp
        trailer = struct.pack('<iiiii', _crc32_signed(stream), len(self.entries),
                              1, len(reg_comp), len(blobs))
        trailer += bytes([1]) + bytes(ARCHIVE_TRAILER_SIZE - len(trailer) - 1)
        Path(path).write_bytes(self.prefix + stream + trailer)


# ---------------------------------------------------------------------------
# TSaveRec record access (offsets relative to the 2560-byte record start)
# ---------------------------------------------------------------------------

# tsHeaderMatch
_HDR_MATCH_LENGTH = 92
_HDR_JACOBY = 101
_HDR_BEAVER = 102
_HDR_VERSION = 552
_HDR_MAGIC = 556
_HDR_PLAYER1 = 880        # TShortUnicodeString (v24+)
_HDR_PLAYER2 = 1138
_HDR_TOT_TIMEDELAY = 1944  # 4 x int32: move, cube, move done, cube done

# tsHeaderGame
_GAME_SCORE1 = 12
_GAME_SCORE2 = 16
_GAME_CRAWFORD = 20
_GAME_NUMBER = 48

# tsMove
_MOVE_POSITION_I = 9       # 26 x int8, player-1 frame (flip for actif = -1)
_MOVE_ACTIF = 64
_MOVE_DICE = 100           # 2 x int32
_MOVE_CUBE_A = 108
_MOVE_DATAMOVES = 124      # EngineStructBestMove, 2184 bytes
_MOVE_ERR_MOVE = 2312
_MOVE_ROLLOUT_INDEX = 2344  # 32 x int32, -1 = no rollout
_MOVE_ANALYZE_M = 2472
_MOVE_FLAGGED = 2520
_MOVE_TIMEDELAY = 2532      # Dword bit list: move i (0-based) -> bit i
_MOVE_TIMEDELAY_DONE = 2536

# EngineStructBestMove (offsets within the sub-record at _MOVE_DATAMOVES)
_BM_NMOVES = 64
_BM_POS_PLAYED = 68        # 32 x 26 int8, mover's frame
_BM_MOVES = 900            # 32 x 8 int8
_BM_EVAL_LEVEL = 1156      # 32 x TEvalLevel {Level: int16, isDouble: u8, fill}
_BM_EVAL = 1284            # 32 x 7 float32

# tsCube
_CUBE_ACTIF = 12
_CUBE_DOUBLE = 16
_CUBE_TAKE = 20
_CUBE_B = 32
_CUBE_POSITION = 36        # 26 x int8, player-1 frame
_CUBE_DOUBLE_ACTION = 64   # EngineStructDoubleAction, 132 bytes
_CUBE_ERR_CUBE = 200
_CUBE_ROLLOUT_INDEX = 224
_CUBE_ANALYZE_C = 232
_CUBE_ANALYZE_CR = 256
_CUBE_FLAGGED_DOUBLE = 288
_CUBE_TIMEDELAY = 297      # Boolean
_CUBE_TIMEDELAY_DONE = 298

# EngineStructDoubleAction (offsets within the sub-record)
_DA_LEVEL = 28
_DA_FLAG_DOUBLE = 56       # int16: 0 = no double, 1 = double
_DA_IS_BEAVER = 58
_DA_EVAL = 60              # 7 x float32, No-Double line
_DA_EQU_ND = 88
_DA_EQU_DT = 92
_DA_EQU_DP = 96
_DA_LEVEL_REQUEST = 100
_DA_EVAL_DT = 104          # 7 x float32, Double/Take line


def iter_records(data: bytes | bytearray):
    """Yield ``(offset, record_type)`` for each TSaveRec in a temp.xg stream."""
    for off in range(0, len(data) - TSAVEREC_SIZE + 1, TSAVEREC_SIZE):
        yield off, data[off + 8]


def read_position(data: bytes | bytearray, offset: int) -> list[int]:
    return list(struct.unpack_from('<26b', data, offset))


def norm_bars(board) -> list[int]:
    """XG signs the bar cells by owner; bgsage stores both bars as counts."""
    b = list(board)
    b[0] = abs(b[0])
    b[25] = abs(b[25])
    return b


def to_mover_board(raw_board: list[int], actif: int) -> tuple[int, ...]:
    """Raw record board (player-1 frame) -> mover's perspective, bgsage bars."""
    b = raw_board if actif == 1 else flip_board(raw_board)
    return tuple(norm_bars(b))


def xg_eval_to_probs(ev) -> list[float]:
    """XG 7-float eval [bg_loss, g_loss, loss, win, g_win, bg_win, equity]
    -> bgsage probs [win, g_win, bg_win, g_loss, bg_loss]."""
    return [float(ev[3]), float(ev[4]), float(ev[5]), float(ev[1]), float(ev[0])]


# --- header ----------------------------------------------------------------


def parse_header(data, off=0) -> dict:
    version, magic = struct.unpack_from('<ii', data, off + _HDR_VERSION)
    if magic != MAGIC_DMLI:
        raise ValueError("tsHeaderMatch record missing DMLI magic")
    tot = struct.unpack_from('<4i', data, off + _HDR_TOT_TIMEDELAY)
    return {
        "version": version,
        "match_length": struct.unpack_from('<i', data, off + _HDR_MATCH_LENGTH)[0],
        "jacoby": bool(data[off + _HDR_JACOBY]),
        "beaver": bool(data[off + _HDR_BEAVER]),
        "tot_timedelay_move": tot[0],
        "tot_timedelay_cube": tot[1],
        "tot_timedelay_move_done": tot[2],
        "tot_timedelay_cube_done": tot[3],
    }


def set_header_timedelay_totals(data: bytearray, off: int, n_move: int, n_cube: int,
                                n_move_done: int = 0, n_cube_done: int = 0) -> None:
    struct.pack_into('<4i', data, off + _HDR_TOT_TIMEDELAY,
                     n_move, n_cube, n_move_done, n_cube_done)


def parse_game_header(data, off) -> dict:
    return {
        "score1": struct.unpack_from('<i', data, off + _GAME_SCORE1)[0],
        "score2": struct.unpack_from('<i', data, off + _GAME_SCORE2)[0],
        "crawford": bool(data[off + _GAME_CRAWFORD]),
        "game_number": struct.unpack_from('<i', data, off + _GAME_NUMBER)[0],
    }


# --- tsMove ----------------------------------------------------------------


def parse_move_record(data, off) -> dict:
    """Decode the fields of a tsMove record needed for matching + harvesting."""
    actif = struct.unpack_from('<i', data, off + _MOVE_ACTIF)[0]
    dm = off + _MOVE_DATAMOVES
    nmoves = struct.unpack_from('<i', data, dm + _BM_NMOVES)[0]
    nmoves = max(0, min(nmoves, 32))
    moves = []
    for i in range(nmoves):
        level, = struct.unpack_from('<h', data, dm + _BM_EVAL_LEVEL + 4 * i)
        ev = struct.unpack_from('<7f', data, dm + _BM_EVAL + 28 * i)
        moves.append({
            "board": norm_bars(read_position(data, dm + _BM_POS_PLAYED + 26 * i)),
            "level": level,
            "eval": list(ev),
        })
    return {
        "actif": actif,
        "position_raw": read_position(data, off + _MOVE_POSITION_I),
        "mover_board": to_mover_board(read_position(data, off + _MOVE_POSITION_I), actif),
        "dice": list(struct.unpack_from('<2i', data, off + _MOVE_DICE)),
        "cube_a": struct.unpack_from('<i', data, off + _MOVE_CUBE_A)[0],
        "n_moves": nmoves,
        "moves": moves,
        "rollout_indices": list(struct.unpack_from('<32i', data, off + _MOVE_ROLLOUT_INDEX)),
        "analyze_m": struct.unpack_from('<i', data, off + _MOVE_ANALYZE_M)[0],
        "timedelay": struct.unpack_from('<I', data, off + _MOVE_TIMEDELAY)[0],
        "timedelay_done": struct.unpack_from('<I', data, off + _MOVE_TIMEDELAY_DONE)[0],
    }


def set_move_timedelay(data: bytearray, off: int, move_bits: int,
                       done_bits: int = 0) -> None:
    """Mark stored move ``i`` (0-based) for batch rollout by setting bit ``i``."""
    struct.pack_into('<II', data, off + _MOVE_TIMEDELAY, move_bits, done_bits)


# --- tsCube ----------------------------------------------------------------


def parse_cube_record(data, off) -> dict:
    actif = struct.unpack_from('<i', data, off + _CUBE_ACTIF)[0]
    da = off + _CUBE_DOUBLE_ACTION
    eval_nd = struct.unpack_from('<7f', data, da + _DA_EVAL)
    eval_dt = struct.unpack_from('<7f', data, da + _DA_EVAL_DT)
    equ_nd, equ_dt, equ_dp = struct.unpack_from('<3f', data, da + _DA_EQU_ND)
    return {
        "actif": actif,
        "position_raw": read_position(data, off + _CUBE_POSITION),
        "mover_board": to_mover_board(read_position(data, off + _CUBE_POSITION), actif),
        "double": struct.unpack_from('<i', data, off + _CUBE_DOUBLE)[0],
        "take": struct.unpack_from('<i', data, off + _CUBE_TAKE)[0],
        "cube_b": struct.unpack_from('<i', data, off + _CUBE_B)[0],
        "level": struct.unpack_from('<i', data, da + _DA_LEVEL)[0],
        "level_request": struct.unpack_from('<h', data, da + _DA_LEVEL_REQUEST)[0],
        "flag_double": struct.unpack_from('<h', data, da + _DA_FLAG_DOUBLE)[0],
        "is_beaver": struct.unpack_from('<h', data, da + _DA_IS_BEAVER)[0],
        "eval_nd": list(eval_nd),
        "eval_dt": list(eval_dt),
        "equity_nd": float(equ_nd),
        "equity_dt": float(equ_dt),
        "equity_dp": float(equ_dp),
        "rollout_index": struct.unpack_from('<i', data, off + _CUBE_ROLLOUT_INDEX)[0],
        "analyze_c": struct.unpack_from('<i', data, off + _CUBE_ANALYZE_C)[0],
        "analyze_cr": struct.unpack_from('<i', data, off + _CUBE_ANALYZE_CR)[0],
        "timedelay": bool(data[off + _CUBE_TIMEDELAY]),
        "timedelay_done": bool(data[off + _CUBE_TIMEDELAY_DONE]),
    }


def set_cube_timedelay(data: bytearray, off: int, marked: bool = True,
                       done: bool = False) -> None:
    data[off + _CUBE_TIMEDELAY] = 1 if marked else 0
    data[off + _CUBE_TIMEDELAY_DONE] = 1 if done else 0


# --- temp.xgi --------------------------------------------------------------


def rebuild_xgi(tempxg: bytes | bytearray) -> bytes:
    """temp.xgi is a fast-access copy of the first and last TSaveRec."""
    n = len(tempxg) // TSAVEREC_SIZE
    if n < 1:
        raise ValueError("empty temp.xg stream")
    return bytes(tempxg[:TSAVEREC_SIZE]) + bytes(tempxg[(n - 1) * TSAVEREC_SIZE:
                                                        n * TSAVEREC_SIZE])


# ---------------------------------------------------------------------------
# temp.xgr rollout contexts (TRolloutContext, 2184 bytes)
# ---------------------------------------------------------------------------


def count_rollout_contexts(xgr: bytes | None) -> int:
    return 0 if not xgr else len(xgr) // ROLLOUT_CONTEXT_SIZE


def parse_rollout_context(xgr: bytes, index: int) -> dict:
    """Decode one TRolloutContext. ``index`` is the RolloutindexM/D value."""
    off = index * ROLLOUT_CONTEXT_SIZE
    if index < 0 or off + ROLLOUT_CONTEXT_SIZE > len(xgr):
        raise IndexError(f"rollout context {index} out of range "
                         f"({count_rollout_contexts(xgr)} contexts)")
    u = struct.unpack_from
    return {
        # settings (echo of the rollout preset used)
        "truncated": bool(xgr[off + 0]),
        "error_limited": bool(xgr[off + 1]),
        "truncate": u('<i', xgr, off + 4)[0],
        "min_roll": u('<i', xgr, off + 8)[0],
        "error_limit": u('<d', xgr, off + 16)[0],
        "max_roll": u('<i', xgr, off + 24)[0],
        "level1": u('<i', xgr, off + 28)[0],
        "level2": u('<i', xgr, off + 32)[0],
        "level_cut": u('<i', xgr, off + 36)[0],
        "variance_reduction": bool(xgr[off + 40]),
        "cubeless": bool(xgr[off + 41]),
        "level1_cube": u('<i', xgr, off + 44)[0],
        "level2_cube": u('<i', xgr, off + 48)[0],
        "truncate_bearoff": u('<i', xgr, off + 56)[0],
        "seed": u('<i', xgr, off + 64)[0],
        "roll_both": bool(xgr[off + 68]),
        "search_interval": u('<f', xgr, off + 72)[0],
        "first_roll": bool(xgr[off + 80]),
        "do_double": bool(xgr[off + 81]),  # roll both lines in multiple rollouts
        "extended": bool(xgr[off + 82]),
        "level_trunc": u('<i', xgr, off + 2136)[0],
        "rotation": u('<i', xgr, off + 2164)[0],
        # results
        "rolled": u('<i', xgr, off + 84)[0],       # ND-line games rolled
        "rolled2": u('<i', xgr, off + 2140)[0],    # D/T-line games rolled
        "ci": u('<f', xgr, off + 2020)[0],         # 95% CI, ND line
        "ci2": u('<f', xgr, off + 2024)[0],        # 95% CI, D/T line
        "result_nd": list(u('<7f', xgr, off + 2028)),
        "result_dt": list(u('<7f', xgr, off + 2056)),
        "duration": u('<f', xgr, off + 2132)[0],   # seconds
        "user_interrupted": bool(xgr[off + 2168]),
        "ver_maj": u('<H', xgr, off + 2170)[0],
        "ver_min": u('<H', xgr, off + 2172)[0],
    }
