"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import numpy as np
import random

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move

class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "starter-one-ply"
    
    def __init__(self):
        self.cache = {} # dict, khóa là cache_key, gtri là điểm số trạng thái đó

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""
        self.cache.clear() # xóa bộ nhớ

    @staticmethod
    def evaluate(board: np.ndarray, merge_score: int, _cache={}) -> float:
        """Heuristic nâng cấp: ô trống + ô lớn ở góc + đơn điệu + mượt + cặp ghép được.
        Bản tối ưu: tính điểm theo từng hàng/cột và cache lại (kết quả giống hệt bản NumPy)."""
        hang = board.tolist()                                # int Python thuần, nhanh hơn thao tác trên mảng NumPy nhỏ
        duong = hang + [list(c) for c in zip(*hang)]         # 4 hàng + 4 cột = 8 đường

        tong = 0.0
        for duong_i in duong:
            khoa = tuple(duong_i)
            diem = _cache.get(khoa)
            if diem is None:
                # log2 nhanh cho số nguyên 2^k: 2->1, 4->2, ... ; ô trống = 0
                bang = [v.bit_length() - 1 if v else 0 for v in khoa]
                tang = giam = 0
                phat_muot = 0
                so_cap = 0
                for i in range(len(khoa) - 1):
                    d = bang[i + 1] - bang[i]
                    if d > 0:
                        tang += d
                    else:
                        giam -= d
                    if khoa[i] and khoa[i + 1]:              # cả hai ô đều có số
                        phat_muot += abs(d)
                        if khoa[i] == khoa[i + 1]:
                            so_cap += 1
                # mỗi ô trống nằm trong 1 hàng + 1 cột nên 125 x 2 = 250 như bản gốc
                diem = (125.0 * khoa.count(0)
                        + 100.0 * so_cap
                        - 40.0 * min(tang, giam)
                        - 15.0 * phat_muot)
                _cache[khoa] = diem
            tong += diem

        # thưởng góc: không tách theo hàng được nên tính riêng
        lon_nhat = max(map(max, hang))
        if lon_nhat and lon_nhat in (hang[0][0], hang[0][-1], hang[-1][0], hang[-1][-1]):
            tong += 60.0 * (lon_nhat.bit_length() - 1)

        return float(tong + merge_score)
    
    def expectiminimax(self, board: np.ndarray, depth: int, is_max_turn: bool):
        board_hash = board.tobytes() # ép bảng hiện tại thành 1 chuỗi byte, 2 trạng thái bảng giống nhau chung 1 chuỗi
        cache_key = (board_hash, depth, is_max_turn) # ép các biến tại trạng thái hiện tại vào 1 tuple khóa
        if cache_key in self.cache:
            return self.cache[cache_key]
        if depth == 0: # điều kiện dừng
            result = self.evaluate(board, 0)
            self.cache[cache_key] = result
            return result
        if is_max_turn: # Lượt lấy max
            max_score = float('-inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_board, merge_score, _changed = move_board(board, move)
                if _changed:
                    value = merge_score + self.expectiminimax(next_board, depth-1,False)
                    if value > max_score:
                        max_score = value
            if max_score == float('-inf'):
                result = self.evaluate(board, 0)
            else:
                result = max_score
            self.cache[cache_key] = result
            return result
        else: # Lượt sinh ô mới
            empty_rows, empty_cols = np.where(board==0)
            empty = len(empty_rows)
            if empty == 0:
                result = self.evaluate(board, 0)
                self.cache[cache_key] = result
                return result
            toa_do_trong = list(zip(empty_rows,empty_cols))
    # Cắt tỉa nhánh, nếu nhiều hơn 4 ô trống, lấy bừa 1 ô để mô phỏng, nhiều hơn 4 ô thì min(empty,2)
            SAMPLE = 1 if empty > 4 else min(empty, 2) 
            if empty > SAMPLE:
                toa_do_bat_ky = random.sample(toa_do_trong,SAMPLE)
                ty_le = 1/SAMPLE # xác suất rơi vào ô này
            else: 
                toa_do_bat_ky = toa_do_trong
                ty_le = 1/empty
            expected_value = 0
            for r,c in toa_do_bat_ky:
                board[r,c] = 2
                expected_value += (ty_le*0.9)*self.expectiminimax(board,depth,True) # xác suất là ô 2
                board[r,c] = 4
                expected_value += (ty_le*0.1)*self.expectiminimax(board,depth,True) # xác suất là ô 4
                board[r,c] = 0
            self.cache[cache_key] = expected_value
            return expected_value
 
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``.

        Khi chấm theo tổng thời gian, ``time_limit_ms`` và
        ``obs.time_remaining_ms`` là ngân sách còn lại của cả ván — không phải
        lời hứa rằng mỗi nước được phép dùng hết số đó.
        """
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")
        if obs.time_remaining_ms is not None and obs.time_remaining_ms < 500:#tgian có hạn và chỉ còn 500ms
            return obs.legal_moves[0] # đi bừa 1 bước hợp lệ

        best_move = obs.legal_moves[0]
        best_value = float("-inf")
        empty_count = obs.board.flatten().tolist().count(0) #mảng 2D ép thành 1D, chuyển về list, đếm số 0
        if empty_count <= 1:
            search_depth = 4
        elif empty_count <= 3:
            search_depth = 3
        else:
            search_depth = 2
        for move in obs.legal_moves:
            next_board, merge_score, _changed = move_board(obs.board, move)
            if _changed: 
                value = merge_score + self.expectiminimax(next_board, search_depth, False)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
