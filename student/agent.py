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
        def evaluate(board: np.ndarray, merge_score: int) -> float:
        """Heuristic nâng cấp: ô trống + ô lớn ở góc + đơn điệu + mượt + cặp ghép được."""
        # Đổi giá trị ô sang log2: 2,4,8,16... -> 1,2,3,4... (ô trống = 0)
        # Làm vậy để chênh lệch 2 ô cạnh nhau không bị phình to theo cấp số nhân
        bang = np.log2(np.maximum(board, 1)).astype(np.float64)
 
        # chênh lệch giữa 2 ô liền kề (theo hàng và theo cột)
        chenh_hang = np.diff(bang, axis=1)
        chenh_cot = np.diff(bang, axis=0)
 
        # cặp ô liền kề mà cả hai đều có số (không tính ô trống)
        co_so_hang = (board[:, :-1] > 0) & (board[:, 1:] > 0)
        co_so_cot = (board[:-1, :] > 0) & (board[1:, :] > 0)
 
        # 1. Số ô trống: càng nhiều càng tốt (giữ lại ý tưởng của bản gốc)
        so_o_trong = int(np.count_nonzero(board == 0))
 
        # 2. Đơn điệu: mỗi hàng/cột nên chỉ tăng dần HOẶC giảm dần
        #    tang = tổng các chỗ bị "tụt xuống", giam = tổng các chỗ bị "nhảy lên"
        #    lấy min vì chỉ cần theo 1 hướng là đủ tốt
        tang_hang = np.maximum(chenh_hang, 0).sum(axis=1)
        giam_hang = np.maximum(-chenh_hang, 0).sum(axis=1)
        tang_cot = np.maximum(chenh_cot, 0).sum(axis=0)
        giam_cot = np.maximum(-chenh_cot, 0).sum(axis=0)
        phat_don_dieu = (np.minimum(tang_hang, giam_hang).sum()
                         + np.minimum(tang_cot, giam_cot).sum())
 
        # 3. Độ mượt: 2 ô cạnh nhau càng gần giá trị nhau càng dễ ghép
        phat_khong_muot = ((np.abs(chenh_hang) * co_so_hang).sum()
                           + (np.abs(chenh_cot) * co_so_cot).sum())
 
        # 4. Số cặp bằng nhau đang đứng cạnh nhau (sắp ghép được)
        so_cap_ghep = int(((board[:, :-1] == board[:, 1:]) & co_so_hang).sum()
                          + ((board[:-1, :] == board[1:, :]) & co_so_cot).sum())
 
        # 5. Thưởng nếu ô lớn nhất nằm ở 1 trong 4 góc
        o_lon_nhat = board.max()
        bon_goc = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
        thuong_goc = float(bang.max()) if max(bon_goc) == o_lon_nhat else 0.0
 
        return float(250.0 * so_o_trong
                     + 100.0 * so_cap_ghep
                     + 60.0 * thuong_goc
                     - 40.0 * phat_don_dieu
                     - 15.0 * phat_khong_muot
                     + merge_score)
        
    
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
