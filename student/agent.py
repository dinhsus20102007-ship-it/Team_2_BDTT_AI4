"""Agent 2048: expectimax + heuristic tra bảng + iterative deepening theo thời gian."""

from __future__ import annotations

import time
import random

import numpy as np

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move

# ---------------------------------------------------------------- tham số
LOST_PENALTY = 200000.0   # cộng vào mỗi hàng/cột, thua = 0 nên luôn thấp hơn mọi trạng thái sống
EMPTY_WEIGHT = 270.0
MERGE_WEIGHT = 700.0
MONO_WEIGHT = 47.0
SUM_WEIGHT = 11.0
MONO_POWER = 4.0
SUM_POWER = 3.5

MOVE_BUDGET_MS = 80.0     # thời gian tối đa cho 1 nước, chỉnh theo giới hạn của đề
MAX_DEPTH = 8
CHANCE_SAMPLE = 4         # số ô trống được lấy mẫu khi có nhiều hơn 4 ô trống


def _build_row_table() -> np.ndarray:
    """Điểm heuristic cho mọi hàng có thể có (4 ô, mỗi ô là rank 0..15)."""
    idx = np.arange(65536, dtype=np.int64)
    r = np.stack([(idx >> (4 * i)) & 15 for i in range(4)], axis=1).astype(np.float64)

    empty = (r == 0).sum(axis=1)
    # số cặp ô kề nhau bằng nhau (có thể gộp)
    merges = ((r[:, :-1] == r[:, 1:]) & (r[:, :-1] > 0)).sum(axis=1)

    # đơn điệu: hàng nên tăng hoặc giảm đều, phạt phần ngược chiều nhỏ hơn
    p = r ** MONO_POWER
    diff = p[:, :-1] - p[:, 1:]
    left = np.where(diff > 0, diff, 0.0).sum(axis=1)
    right = np.where(diff < 0, -diff, 0.0).sum(axis=1)
    mono = np.minimum(left, right)

    # phạt tổng lũy thừa rank: ép bot gộp ô lớn thay vì để rải rác
    total = (r ** SUM_POWER).sum(axis=1)

    return (LOST_PENALTY
            + EMPTY_WEIGHT * empty
            + MERGE_WEIGHT * merges
            - MONO_WEIGHT * mono
            - SUM_WEIGHT * total)


ROW_TABLE = _build_row_table()          # chạy 1 lần khi import
_SHIFTS = np.array([0, 4, 8, 12], dtype=np.int64)


class _Timeout(Exception):
    pass


class Agent:
    name = "expectimax-table"

    def __init__(self):
        self.cache = {}
        self.deadline = float("inf")

    def reset(self, seed: int | None = None) -> None:
        self.cache.clear()
        self.deadline = float("inf")
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def evaluate(board: np.ndarray, merge_score: int = 0) -> float:
        # ô -> rank (2->1, 4->2, ...), chặn ở 15 để vừa bảng 65536 hàng
        e = np.minimum(np.log2(np.maximum(board, 1)).astype(np.int64), 15)
        rows = (e << _SHIFTS).sum(axis=1)
        cols = (e.T << _SHIFTS).sum(axis=1)
        return float(ROW_TABLE[rows].sum() + ROW_TABLE[cols].sum())

    def expectiminimax(self, board: np.ndarray, depth: int, is_max_turn: bool) -> float:
        if time.perf_counter() > self.deadline:
            raise _Timeout

        key = (board.tobytes(), depth, is_max_turn)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if depth == 0:
            result = self.evaluate(board)
            self.cache[key] = result
            return result

        if is_max_turn:
            best = float("-inf")
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nb, ms, changed = move_board(board, move)
                if changed:
                    v = ms + self.expectiminimax(nb, depth - 1, False)
                    if v > best:
                        best = v
            result = 0.0 if best == float("-inf") else best   # hết nước đi = thua
            self.cache[key] = result
            return result

        # lượt sinh ô mới
        rows, cols = np.where(board == 0)
        empty = len(rows)
        if empty == 0:
            result = self.evaluate(board)
            self.cache[key] = result
            return result

        cells = list(zip(rows.tolist(), cols.tolist()))
        if empty > 4:
            cells = random.sample(cells, min(CHANCE_SAMPLE, empty))
        prob = 1.0 / len(cells)

        expected = 0.0
        for r, c in cells:
            try:
                board[r, c] = 2
                v2 = self.expectiminimax(board, depth, True)
                board[r, c] = 4
                v4 = self.expectiminimax(board, depth, True)
            finally:
                board[r, c] = 0      # luôn trả board về nguyên trạng, kể cả khi timeout
            expected += prob * (0.9 * v2 + 0.1 * v4)

        self.cache[key] = expected
        return expected

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")
        if len(obs.legal_moves) == 1:
            return obs.legal_moves[0]
        if obs.time_remaining_ms is not None and obs.time_remaining_ms < 500:
            return obs.legal_moves[0]

        # ngân sách cho nước này
        budget = min(MOVE_BUDGET_MS, 0.6 * time_limit_ms)
        if obs.time_remaining_ms is not None:
            budget = min(budget, obs.time_remaining_ms / 150.0)
        budget = max(budget, 5.0)

        t0 = time.perf_counter()
        self.cache.clear()
        self.deadline = float("inf")          # depth 1 luôn chạy xong
        best_move = obs.legal_moves[0]

        for depth in range(1, MAX_DEPTH + 1):
            cur_move, cur_val = None, float("-inf")
            try:
                for move in obs.legal_moves:
                    nb, ms, changed = move_board(obs.board, move)
                    if not changed:
                        continue
                    v = ms + self.expectiminimax(nb, depth, False)
                    if v > cur_val:
                        cur_val, cur_move = v, move
            except _Timeout:
                break                          # bỏ kết quả dở dang của tầng này
            if cur_move is not None:
                best_move = cur_move

            elapsed = (time.perf_counter() - t0) * 1000.0
            if elapsed > budget * 0.35:        # tầng sau tốn gấp vài lần, không đủ thì dừng
                break
            self.deadline = t0 + budget / 1000.0

        self.deadline = float("inf")
        return best_move