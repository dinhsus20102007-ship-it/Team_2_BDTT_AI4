"""Agent 2048: bitboard + bảng tra + expectimax, iterative deepening theo ngân sách compute."""

from __future__ import annotations

import time

from game2048.observation import Observation
from game2048.types import Move

# ------------------------------------------------------------ heuristic (theo từng hàng)
LOST_PENALTY = 200000.0
EMPTY_WEIGHT = 270.0
MERGE_WEIGHT = 700.0
MONO_WEIGHT = 47.0
SUM_WEIGHT = 11.0
MONO_POWER = 4.0
SUM_POWER = 3.5

# ------------------------------------------------------------ search
MAX_DEPTH = 12
PROB_CUT = 0.0001          # cắt nhánh có xác suất tích lũy quá nhỏ
# chia ngân sách còn lại của cả ván cho hệ số này để ra ngân sách 1 nước
DIV_SAFE = 1500.0          # >= 7 ô trống: đi nhanh
DIV_MID = 600.0            # 4..6 ô trống
DIV_DANGER = 250.0         # <= 3 ô trống: nghĩ sâu
MIN_BUDGET_MS = 3.0


def _pack(cells):
    return cells[0] | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)


def _slide_left(line):
    tiles = [x for x in line if x]
    out = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(min(tiles[i] + 1, 15))
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return out


def _build_tables():
    left = [0] * 65536
    right = [0] * 65536
    heur = [0.0] * 65536
    for row in range(65536):
        line = [(row >> (4 * i)) & 0xF for i in range(4)]

        left[row] = _pack(_slide_left(line))
        right[row] = _pack(_slide_left(line[::-1])[::-1])

        total = 0.0
        empty = 0
        merges = 0
        prev = 0
        counter = 0
        for rank in line:
            total += rank ** SUM_POWER
            if rank == 0:
                empty += 1
            else:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank
        if counter > 0:
            merges += 1 + counter

        mono_l = mono_r = 0.0
        for i in range(1, 4):
            if line[i - 1] > line[i]:
                mono_l += line[i - 1] ** MONO_POWER - line[i] ** MONO_POWER
            else:
                mono_r += line[i] ** MONO_POWER - line[i - 1] ** MONO_POWER

        heur[row] = (LOST_PENALTY
                     + EMPTY_WEIGHT * empty
                     + MERGE_WEIGHT * merges
                     - MONO_WEIGHT * min(mono_l, mono_r)
                     - SUM_WEIGHT * total)
    return left, right, heur


_LEFT, _RIGHT, _HEUR = _build_tables()     # chạy 1 lần khi import
_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def _transpose(x: int) -> int:
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _shift(b: int, table) -> int:
    return (table[b & 0xFFFF]
            | (table[(b >> 16) & 0xFFFF] << 16)
            | (table[(b >> 32) & 0xFFFF] << 32)
            | (table[(b >> 48) & 0xFFFF] << 48))


def _apply(b: int, i: int) -> int:
    """i: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT. Trả về bảng mới (bằng b nếu không đổi)."""
    if i == 2:
        return _shift(b, _LEFT)
    if i == 3:
        return _shift(b, _RIGHT)
    t = _transpose(b)
    t = _shift(t, _LEFT if i == 0 else _RIGHT)
    return _transpose(t)


def _eval(b: int) -> float:
    t = _transpose(b)
    return (_HEUR[b & 0xFFFF] + _HEUR[(b >> 16) & 0xFFFF]
            + _HEUR[(b >> 32) & 0xFFFF] + _HEUR[b >> 48]
            + _HEUR[t & 0xFFFF] + _HEUR[(t >> 16) & 0xFFFF]
            + _HEUR[(t >> 32) & 0xFFFF] + _HEUR[t >> 48])


def _to_bits(board) -> int:
    b = 0
    for i, v in enumerate(board.flatten().tolist()):
        if v:
            b |= min(int(v).bit_length() - 1, 15) << (4 * i)
    return b


class _Timeout(Exception):
    pass


class Agent:
    name = "bitboard-expectimax"

    def __init__(self):
        self.cache = {}
        self.deadline = float("inf")

    def reset(self, seed: int | None = None) -> None:
        self.cache.clear()
        self.deadline = float("inf")

    # ---- nút người chơi
    def _max(self, b: int, depth: int, cprob: float) -> float:
        if depth == 0:
            return _eval(b)
        if time.perf_counter() > self.deadline:
            raise _Timeout
        key = (b, depth)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        best = 0.0                                  # hết nước đi = thua = 0
        for i in range(4):
            nb = _apply(b, i)
            if nb != b:
                v = self._chance(nb, depth - 1, cprob)
                if v > best:
                    best = v
        self.cache[key] = best
        return best

    # ---- nút sinh ô mới
    def _chance(self, b: int, depth: int, cprob: float) -> float:
        if cprob < PROB_CUT:
            return _eval(b)
        empties = [s for s in range(0, 64, 4) if not (b >> s) & 0xF]
        n = len(empties)
        c2 = cprob * 0.9 / n
        c4 = cprob * 0.1 / n
        total = 0.0
        for s in empties:
            total += 0.9 * self._max(b | (1 << s), depth, c2)
            total += 0.1 * self._max(b | (2 << s), depth, c4)
        return total / n

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        legal = obs.legal_moves
        if not legal:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")
        if len(legal) == 1:
            return legal[0]

        b = _to_bits(obs.board)
        empty = sum(1 for s in range(0, 64, 4) if not (b >> s) & 0xF)

        total = obs.time_remaining_ms if obs.time_remaining_ms is not None else time_limit_ms
        div = DIV_SAFE if empty >= 7 else (DIV_MID if empty >= 4 else DIV_DANGER)
        budget = max(total / div, MIN_BUDGET_MS)

        cands = [(i, _ORDER[i]) for i in range(4) if _ORDER[i] in legal]
        best_move = legal[0]

        t0 = time.perf_counter()
        self.cache.clear()
        self.deadline = float("inf")                # depth 1 luôn chạy xong

        for depth in range(1, MAX_DEPTH + 1):
            cur_move, cur_val = None, -1.0
            try:
                for i, mv in cands:
                    nb = _apply(b, i)
                    if nb == b:
                        continue
                    v = self._chance(nb, depth - 1, 1.0)
                    if v > cur_val:
                        cur_val, cur_move = v, mv
            except _Timeout:
                break                               # bỏ kết quả dở dang của tầng này
            if cur_move is not None:
                best_move = cur_move
            elapsed = (time.perf_counter() - t0) * 1000.0
            if elapsed > budget * 0.3:              # tầng sau tốn gấp vài lần, không đủ thì dừng
                break
            self.deadline = t0 + budget / 1000.0

        self.deadline = float("inf")
        return best_move