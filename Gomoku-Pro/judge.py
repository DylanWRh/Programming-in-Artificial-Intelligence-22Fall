import numpy as np
import time

def judge(mp, x, y):
    # if mp[x, y] == 0:
    #     return 0
    dirs = [[0, 1], [1, 0], [1, 1], [-1, 1]]
    for d in dirs:
        dx, dy = d[0], d[1]
        cnt1 = cnt2 = 0
        cx, cy = x, y
        while mp[cx][cy] == mp[x][y]:
            cnt1 += 1
            nx, ny = cx + dx, cy + dy
            if (nx >= 0) and (ny >= 0) and (nx < 15) and (ny < 15) and mp[nx, ny] == mp[x, y]:
                cx, cy = nx, ny
            else:
                cx, cy = x, y
                break
        while mp[cx][cy] == mp[x][y]:
            cnt2 += 1
            nx, ny = cx - dx, cy - dy
            if (nx >= 0) and (ny >= 0) and (nx < 15) and (ny < 15) and mp[nx, ny] == mp[x, y]:
                cx, cy = nx, ny
            else:
                break
        if cnt1 + cnt2 >= 6:
            return mp[x, y]
    

        
    return 0

def judge_new(mp, x, y):
    if abs(np.sum(mp[x:x+5, y])) == 5 or abs(np.sum(mp[x, y:y+5])) == 5 or abs(np.sum(np.diag(mp[x:x+5, y:y+5]))) == 5 or abs(np.sum(np.diag(mp[max(x-4, 0):x+1, y:y+5][::-1]))) == 5:

        return mp[x, y]
    else:
        return 0


def judge_tot(mp):
    #print(mp)
    # for i in range(15):
    #     for j in range(15):
    #         if judge(mp, i, j) != 0:
    #             return judge(mp, i, j)
    chess_x, chess_y = np.where((mp !=0))
    for i, j in zip(chess_x, chess_y):
        func_value = judge(mp, i, j)
        if func_value != 0:
            return func_value

    # for i, j in zip(chess_x, chess_y):
    #     if judge_new(mp, i, j) != 0:
    #         return judge_new(mp, i, j)

    return 0
