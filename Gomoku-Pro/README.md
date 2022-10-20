# Gomoku Pro

A small project for the course, to implement an AI for Gomoku-Pro.

This is a group project, and another contributor is *Yiming Wang*.

Use min-max searching algorithm and alpha-beta pruning, with [evaluation function](https://github.com/sxysxy/GensokyoGomoku)(greatly simplified here) and a searching depth of only 2. Therefore, it is not strange that everyone has a great chance to beat it.

Since it is only a one-week project, we did not spend too much effort on optimizing the algorithm, so the efficiency is apparently low. 

## Rule

Basic rules are the same as naive Gomoku with 15 x 15 board.

Restrictions for BLACK:
1) For the first step, BLACK can only make a move at the middle of the board, i.e. (7, 7).
2) For the second step, BLACK cannot make a move within the 3 x 3 area in the middle, i.e. \[6, 8\] x \[6, 8\].

No restrictions for WHITE.

Only AI will follow the restrictions; human players can just ignore them, and make a move anywhere as you like.

## Run

Directly run `main.py`.
