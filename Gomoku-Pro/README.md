# Gomoku Pro

A small project for the course, to implement an AI for Gomoku-Pro.

Use min-max searching algorithm and alpha-beta pruning, with a simplified [evaluation function](https://github.com/sxysxy/GensokyoGomoku) and a searching depth of only 2.

## Rule

Basic rules are the same as naive Gomoku with 15 x 15 board.

Restrictions for BLACK:
1) For the first step, BLACK can only put at the middle of the board, i.e. (7, 7).
2) For the second step, BLACK cannot put at the 3 x 3 area in the middle, i.e. [6, 8] x [6, 8].

No restrictions for WHITE.

Only AI will follow the restrictions; players can just ignore them, and put chess anywhere you like.

## Run

Directly run `main.py`.
