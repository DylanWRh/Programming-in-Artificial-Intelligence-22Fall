import tkinter as tk
import numpy as np
from judge import judge, judge_tot
from aiplay import ai


main_window = tk.Tk()
line_space, board_space = 40, 30
# Set the main window of the game
main_window.title('Gomoku-Pro')

# Paint the chessboard
board_width = board_height = 14 * line_space + 2 * board_space
board_canvas = tk.Canvas(main_window, width=board_width, height=board_height, background='lightcyan')
board_canvas.pack()

# Set crossing line params
line_start = board_space
line_end = line_space * 14 + board_space

# Paint crossing lines
for line in range(15):
    line_i = line * line_space + board_space
    board_canvas.create_line(line_i, line_start, line_i, line_end)
    board_canvas.create_line(line_start, line_i, line_end, line_i)

# Set crossing dots params
dot_r = 5
x_left = y_up = board_space + 3 * line_space
x_right = y_down = board_space + 11 * line_space
x_mid = y_mid = board_space + 7 * line_space

# Paint crossing dots
board_canvas.create_oval(x_left - dot_r, y_up - dot_r, x_left + dot_r, y_up + dot_r, fill='black')  # left up
board_canvas.create_oval(x_left - dot_r, y_down - dot_r, x_left + dot_r, y_down + dot_r, fill='black')  # left down
board_canvas.create_oval(x_right - dot_r, y_up - dot_r, x_right + dot_r, y_up + dot_r, fill='black')  # right up
board_canvas.create_oval(x_right - dot_r, y_down - dot_r, x_right + dot_r, y_down + dot_r, fill='black')  # right down
board_canvas.create_oval(x_mid - dot_r, y_mid - dot_r, x_mid + dot_r, y_mid + dot_r, fill='black')  # mid

# Paint chess dots
chess_r = line_space // 2

# Game progress information
cur_player = 1
chess_mp = np.zeros((15, 15), dtype=int)  # 0 for empty, 1 for black and -1 for white
game_end = False


def put_chess_player(event):
    global game_end
    if game_end:
        return
    for i in range(15):
        for j in range(15):
            if (event.x - board_space - i * line_space) ** 2 + (
                    event.y - board_space - j * line_space) ** 2 <= chess_r ** 2:
                if chess_mp[i, j] == 0:
                    put_chess(i, j)
                    if not game_end:
                        ai_put = ai.find_sol(chess_mp)
                        if game_end:
                            return
                        put_chess(ai_put[0], ai_put[1])
                    return


def put_chess(x, y):
    global cur_player, chess_mp, game_end
    chess_x = board_space + line_space * x
    chess_y = board_space + line_space * y
    global cur_player, chess_mp
    if cur_player == 1 and chess_mp[x, y] == 0:
        board_canvas.create_oval(chess_x - chess_r, chess_y - chess_r,
                                 chess_x + chess_r, chess_y + chess_r, fill='black')
        chess_mp[x, y] = 1
        cur_player = -cur_player
    elif cur_player == -1 and chess_mp[x, y] == 0:
        board_canvas.create_oval(chess_x - chess_r, chess_y - chess_r,
                                 chess_x + chess_r, chess_y + chess_r, fill='white')
        chess_mp[x, y] = -1
        cur_player = -cur_player
    winner = judge(chess_mp, x, y)
    if winner == 1:
        print('Black Wins')
        game_end = True
    elif winner == -1:
        print('White Wins')
        game_end = True
    elif np.sum(chess_mp == 0) == 0:
        print('Draw')
        game_end = True


# Connect canvas with functions
board_canvas.bind('<Button -1>', put_chess_player)
board_canvas.pack()

# Put the first chess
if ai.color == 1:
    put_chess(7, 7)

if __name__ == '__main__':
    tk.mainloop()
