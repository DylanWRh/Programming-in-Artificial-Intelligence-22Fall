import random
import numpy as np
from chessboard import *
from judge import *
import time
from enum import Enum

# 1 for black and -1 for white
ai_player = -1
try:
    ai_player = 2 * int(input('Choose the Color You Want to Play:\n1. Black\n2. White\n')) - 3
except:
    print('Wrong Input')
    quit()
assert ai_player in [-1, 1]


class game_status(Enum):
        # BLACK for AI and WHITE for opponent
        CHESS_WHITE_WIN = 1             
        CHESS_BLACK_WIN = 2             
        CHESS_FLEX4_WHITE = 3               
        CHESS_FLEX4_BLACK = 4           
        CHESS_BLOCK4_WHITE = 5              
        CHESS_BLOCK4_BLACK = 6          
        CHESS_FLEX3_WHITE = 7               
        CHESS_FLEX3_BLACK = 8           
        CHESS_BLOCK3_WHITE = 9              
        CHESS_BLOCK3_BLACK = 10          
        CHESS_FLEX2_WHITE = 11               
        CHESS_FLEX2_BLACK = 12           
        CHESS_BLOCK2_WHITE = 13              
        CHESS_BLOCK2_BLACK = 14          
        CHESS_FLEX1_WHITE = 15               
        CHESS_FLEX1_BLACK = 16           
        CHESS_WEIGHT_END = 17


class AI:
    def __init__(self, color, time_limit=10, depth=2):
        # Basic params
        self.color = color  # 1 for black and -1 for white
        self.time_limit = time_limit
        self.depth = depth
        self.round = 0
        self.start_time = 0

        self.x_grid = 15
        self.y_grid = 15

        self.ai_next_move_x = 0
        self.ai_next_move_y = 0

        #evaluation
        self.status = game_status
        self.eval_state = self.init_status_identifier()
        self.weights = np.array([0, -4000, 4000, -2000, 2000, -1000, 1000,
                                -1000, 1000, -600, 400, -600, 400, -150, 100,
                                -150, 100])

    def in_mp(self, x, y):
        return (x >= 0) and (x < self.x_grid) and (y >= 0) and (y < self.y_grid)

    def get_legal_steps(self, mp):
        legal_steps = []
        

        # old solution
        # st = time.time()
        # for i in range(15):
        #     for j in range(15):
        #         # banned move
        #         if (self.round == 1) and (self.color == 1) and (6 <= i <= 8) and (6 <= j <= 8):
        #             continue

        #         # there should be enough chess in 5x5 area
                
        #         if mp[i, j] == 0:
        #             flag = True
        #             near_cnt = 0
        #             for di in range(-2, 3):
        #                 for dj in range(-2, 3):
        #                     if self.in_mp(i + di, j + dj) and mp[i + di, j + dj] != 0:
        #                         near_cnt += 1
        #                     if near_cnt >= min(self.round, 2):
        #                         flag = False
        #                         legal_steps.append((i, j))
        #                         break
        #                 if not flag:
        #                     break
        # et = time.time()
        # print(f'old solution: {et-st}s')

        temp_board = (mp != 0).astype(int)
        x_slots, y_slots = np.where((temp_board == 0))
        for i, j in zip(x_slots, y_slots):
            # banned move
            if (self.round == 1) and (self.color == 1) and (6 <= i <= 8) and (6 <= j <= 8):
                continue

            left, up = max(0, i- 2), max(0, j-2)
            right, down = min(self.x_grid, i+3), min(self.y_grid, j+3)

            near_cnt = np.sum(temp_board[left:right, up:down])
            if near_cnt >= min(self.round, 2):
                legal_steps.append((i, j))

        return legal_steps

    def minimax(self, mp, cur_color, alpha, beta, depth):
        # search to enough depth, or game ends
        if depth >= self.depth or judge_tot(mp) or np.sum(mp == 0) == 0:
            return self.evaluation(mp, cur_color)

        # get all legal nodes
        legal_steps = self.get_legal_steps(mp)

        # minimax search
        if cur_color == self.color:
            for legal_step in legal_steps:
                new_mp = mp.copy()
                x, y = legal_step
                new_mp[x, y] = cur_color
                score = self.minimax(new_mp, -cur_color, alpha, beta, depth + 1)

                if score > alpha:
                    if not depth:
                        self.ai_next_move_x, self.ai_next_move_y = x, y
                    alpha = score
                if alpha >= beta:
                    return alpha
            return alpha
        else:
            for legal_step in legal_steps:
                new_mp = mp.copy()
                x, y = legal_step
                new_mp[x, y] = cur_color
                score = self.minimax(new_mp, -cur_color, alpha, beta, depth + 1)

                if score < beta:
                    beta = score
                if alpha >= beta:
                    return beta
            return beta

    def find_sol(self, mp):
        self.start_time = time.process_time()
        self.round += 1
        self.minimax(mp, self.color, -10000000, 10000000, 0)
        end_time = time.process_time()
        print(f'Execution time of round {self.round}: {end_time - self.start_time}, put at ({self.ai_next_move_x}, {14-self.ai_next_move_y})')
        return self.ai_next_move_x, self.ai_next_move_y

    #note: need to consider when computing scores!
    # 1 for black and 2 for white
    def init_status_identifier(self):

        state = np.zeros((3, 3, 3, 3, 3, 3))
        # black wins
        state[1, 1, 1, 1, 1, 1] = self.status.CHESS_BLACK_WIN.value
        state[1, 1, 1, 1, 1, 0] = self.status.CHESS_BLACK_WIN.value
        state[0, 1, 1, 1, 1, 1] = self.status.CHESS_BLACK_WIN.value
        state[1, 1, 1, 1, 1, 2] = self.status.CHESS_BLACK_WIN.value
        state[2, 1, 1, 1, 1, 1] = self.status.CHESS_BLACK_WIN.value

        # white wins
        state[2, 2, 2, 2, 2, 2] = self.status.CHESS_WHITE_WIN.value
        state[2, 2, 2, 2, 2, 0] = self.status.CHESS_WHITE_WIN.value
        state[0, 2, 2, 2, 2, 2] = self.status.CHESS_WHITE_WIN.value
        state[2, 2, 2, 2, 2, 1] = self.status.CHESS_WHITE_WIN.value
        state[1, 2, 2, 2, 2, 2] = self.status.CHESS_WHITE_WIN.value

        # black has 4 FLEX
        state[0, 1, 1, 1, 1, 0] = self.status.CHESS_FLEX4_BLACK.value

        # white has 4 FLEX
        state[0, 2, 2, 2, 2, 0] = self.status.CHESS_FLEX4_WHITE.value

        # black has 3 FLEX
        state[0, 1, 1, 1, 0, 0] = self.status.CHESS_FLEX3_BLACK.value
        state[0, 1, 1, 0, 1, 0] = self.status.CHESS_FLEX3_BLACK.value
        state[0, 1, 0, 1, 1, 0] = self.status.CHESS_FLEX3_BLACK.value
        state[0, 0, 1, 1, 1, 0] = self.status.CHESS_FLEX3_BLACK.value

        # white has 3 FLEX
        state[0, 2, 2, 2, 0, 0] = self.status.CHESS_FLEX3_WHITE.value
        state[0, 2, 2, 0, 2, 0] = self.status.CHESS_FLEX3_WHITE.value
        state[0, 2, 0, 2, 2, 0] = self.status.CHESS_FLEX3_WHITE.value
        state[0, 0, 2, 2, 2, 0] = self.status.CHESS_FLEX3_WHITE.value

        # black has 2 FLEX
        state[0, 1, 1, 0, 0, 0] = self.status.CHESS_FLEX2_BLACK.value
        state[0, 1, 0, 1, 0, 0] = self.status.CHESS_FLEX2_BLACK.value
        state[0, 1, 0, 0, 1, 0] = self.status.CHESS_FLEX2_BLACK.value
        state[0, 0, 1, 1, 0, 0] = self.status.CHESS_FLEX2_BLACK.value
        state[0, 0, 1, 0, 1, 0] = self.status.CHESS_FLEX2_BLACK.value
        state[0, 0, 0, 1, 1, 0] = self.status.CHESS_FLEX2_BLACK.value

        # white has 2 FLEX
        state[0, 2, 2, 0, 0, 0] = self.status.CHESS_FLEX2_WHITE.value
        state[0, 2, 0, 2, 0, 0] = self.status.CHESS_FLEX2_WHITE.value
        state[0, 2, 0, 0, 2, 0] = self.status.CHESS_FLEX2_WHITE.value
        state[0, 0, 2, 2, 0, 0] = self.status.CHESS_FLEX2_WHITE.value
        state[0, 0, 2, 0, 2, 0] = self.status.CHESS_FLEX2_WHITE.value
        state[0, 0, 0, 2, 2, 0] = self.status.CHESS_FLEX2_WHITE.value

        for i in range(4):
            state[0, i==0, i==1, i==2, i==3, 0] = self.status.CHESS_FLEX1_BLACK.value
        for i in range(4):
            state[0, int(i==0)*2, int(i==1)*2, int(i==2)*2, int(i==3)*2, 0] = self.status.CHESS_FLEX1_WHITE.value

        # define BLOCKING
        
        pos = np.zeros(7).astype(int)
        while pos[0] <= 2:
            while pos[1] <= 2:
                while pos[2] <= 2:
                    while pos[3] <= 2:
                        while pos[4] <= 2:
                            while pos[5] <= 2:

                                leftb, rightb, leftw, rightw = 0, 0, 0, 0
                                if pos[0] == 1: leftb += 1
                                elif pos[0] == 2: leftw += 1
                                for i in range(1, 5):
                                    if pos[i] == 1:
                                        leftb += 1
                                        rightb += 1
                                    elif pos[i] == 2:
                                        leftw += 1
                                        rightw += 1
                                if pos[5] == 1: rightb += 1
                                elif pos[5] == 2: rightw += 1

                                def get_block(x, a, b):
                                        if ((leftb == x and leftw == 0) or (rightb == x and rightw == 0)) and not state[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]:
                                            state[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]] = a
                                        if ((leftb == 0 and leftw == x) or (rightb == 0 and rightw == x)) and not state[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]:
                                            state[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]] = b

                                get_block(4, self.status.CHESS_BLOCK4_BLACK.value, self.status.CHESS_BLOCK4_WHITE.value)

                                get_block(3, self.status.CHESS_BLOCK3_BLACK.value, self.status.CHESS_BLOCK3_WHITE.value)

                                get_block(2, self.status.CHESS_BLOCK2_BLACK.value, self.status.CHESS_BLOCK2_WHITE.value)


                                pos[5] += 1

                            pos[4] += 1
                        pos[3] += 1
                    pos[2] += 1
                pos[1] += 1
            pos[0] += 1

        return state.astype(int)

    def evaluation(self, mp, cur_color):
        res = judge_tot(mp)
        if res == self.color:
            return 1000000
        if res + self.color == 0:
            return -1000000
        if np.sum(mp == 0) == 0:
            return 0

        status = np.zeros((4, 17))
        # compute status according to current board.
        # switch -1 (white) to 2
        if cur_color == 1:
            board = np.where(mp == -1, 2, mp)
        else:
            board = np.where(mp == 1, 2, mp)
            board = np.where(board == -1, 1, board)

        # horizon
        for i in range(15):
            for j in range(10):
                status[0][self.eval_state[board[i, j], board[i, j+1], board[i, j+2], board[i, j+3], board[i, j+4], board[i, j+5]]] += 1

        # vertical
        for i in range(10):
            for j in range(15):
                status[1][self.eval_state[board[i, j], board[i+1, j], board[i+2, j], board[i+3, j], board[i+4, j], board[i+5, j]]] += 1

        # from top left to bottom right
        for i in range(10):
            for j in range(10):
                status[2][self.eval_state[board[i, j], board[i+1, j+1], board[i+2, j+2], board[i+3, j+3], board[i+4, j+4], board[i+5, j+5]]] += 1

        # from top right to bottom left
        for i in range(14, 4, -1):
            for j in range(10):
                status[3][self.eval_state[board[i, j], board[i-1, j+1], board[i-2, j+2], board[i-3, j+3], board[i-4, j+4], board[i-5, j+5]]] += 1
        
        status_cnt = np.zeros(17)
        acc_score = 0

        for i in range(4):
            acc_score += np.dot(status[i, :], self.weights)

        for i in range(17):
            status_cnt[i] = np.sum(np.sign(status[:, i]))

        # additional scores
        if status_cnt[self.status.CHESS_BLACK_WIN.value]:
            acc_score += 100000
        elif status_cnt[self.status.CHESS_WHITE_WIN.value]:
            acc_score -= 100000
        elif status_cnt[self.status.CHESS_FLEX4_WHITE.value] > 0:
            acc_score -= 50000
        elif status_cnt[self.status.CHESS_BLOCK4_WHITE.value] > 0:
            acc_score -= 30000
        elif status_cnt[self.status.CHESS_FLEX4_WHITE.value] == 0 and status_cnt[self.status.CHESS_BLOCK4_WHITE.value] == 0:
            k = 0
            for i in range(4):
                for j in range(4):
                    if i != j:
                        k += status[i][self.status.CHESS_BLOCK4_BLACK.value] * status[j][self.status.CHESS_FLEX3_BLACK.value]

            if status_cnt[self.status.CHESS_FLEX4_BLACK.value]:
                acc_score += 20000
            elif status_cnt[self.status.CHESS_BLOCK4_BLACK.value] >= 2:
                acc_score += 20000
            elif k > 0:
                acc_score += 20000
            elif status_cnt[self.status.CHESS_FLEX3_WHITE.value] and status_cnt[self.status.CHESS_BLOCK4_BLACK.value] == 0:
                acc_score -= 20000
            elif status_cnt[self.status.CHESS_FLEX3_WHITE.value] == 0 and status_cnt[self.status.CHESS_FLEX3_BLACK.value] >= 2:
                acc_score += 10000
        return acc_score


ai = AI(ai_player)
