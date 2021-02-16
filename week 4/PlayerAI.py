from BaseAI import BaseAI
import math
from Grid import Grid
import time
import random


class PlayerAI(BaseAI):
    def __init__(self, weights=None):
        super(PlayerAI, self, ).__init__()
        self.max_depth = 6
        self.max_time = 0.2
        self.weights = weights
        self.move = None

    def compute_prob(self, array):
        prob = []
        sum = 0
        for i in array:
            sum += i
        for i in array:
            prob.append(i / sum)
        return prob

    def compute_utility(self, grid):
        if self.weights is not None:
            prob = self.compute_prob(self.weights)
        else:
            w1 = 1
            w2 = 0
            w3 = 5
            w4 = 0
            w5 = 0
            w6 = 0
            array = [w1, w2, w3, w4, w5, w6]
            prob = self.compute_prob(array)

        u1 = self.grid_credit(grid)
        u2 = self.grid_sum(grid)
        u3 = grid.getMaxTile()
        u4 = self.monotonicity(grid)
        u5 = self.empty_cells(grid)
        u6 = self.smoothness(grid)
        us = [u1, u2, u3, u4, u5, u6]
        # us = compute_prob(us)
        utility = 0
        for u, w in zip(us, prob):
            utility += u * w
        return utility

    def grid_credit(self, grid):
        WEIGHT_MATRIX1 = [
            [2048, 1024, 64, 32],
            [512, 128, 16, 2],
            [256, 8, 2, 1],
            [4, 2, 1, 1]
        ]

        #SNAKE MATRIX
        WEIGHT_MATRIX2 = [
            [16, 15, 14, 13],
            [9, 10, 11, 12],
            [5, 6, 7, 8],
            [1, 2, 3, 4]
        ]
        '''
        for i in range(4):
            for j in range(4):
                WEIGHT_MATRIX2[i][j] = math.exp(WEIGHT_MATRIX2[i][j])
        '''
        WEIGHT_MATRIX = [WEIGHT_MATRIX1, WEIGHT_MATRIX2]
        # choice = random.choice((0,1))

        utility = 0
        for i, wi in enumerate(WEIGHT_MATRIX1):
            for j, wj in enumerate(wi):
                utility += wj * grid.getCellValue([i, j])
        return utility

    def grid_sum(self, grid):
        utility = 0
        for i in range(4):
            for j in range(4):
                utility += grid.getCellValue([i, j])
        return utility

    def empty_cells(self, grid):
        return 1/1+math.exp(len(grid.getAvailableCells())+10e-6)

    def monotonicity(self, grid):
        # https://doi.org/10.1145/3337722.3341838
        def rotate_grid(grid):
            new_matrix_ = [[grid.map[j][ni] for j in range(len(grid.map))] for ni in range(len(grid.map[0]) - 1, -1, -1)]
            #new_matrix_ = [[grid.getCellValue([j, i]) for j in range(4)] for i in range(3, -1, -1)]
            new_matrix = Grid(4)
            new_matrix.map = new_matrix_
            return new_matrix

        eval_grid = grid.clone()
        monotonic_score = 0
        best = -1
        for z in range(4):
            for i in range(4):
                for j in range(3):
                    if eval_grid.getCellValue([i, j]) >= eval_grid.getCellValue([i, j + 1]):
                        monotonic_score += 1

            for j in range(4):
                for i in range(3):
                    if grid.getCellValue([i, j]) >= grid.getCellValue([i + 1, j]):
                        monotonic_score += 1
            eval_grid = rotate_grid(grid)
            if monotonic_score > best:
                best = monotonic_score
        return monotonic_score

    def smoothness(self, grid):
        smooth = 0
        for i in range(4):
            for j in range(4 - 1):
                smooth += abs(grid.map[i][j] - grid.map[i][j + 1])

        for i in range(4):
            for j in range(4 - 1):
                smooth += abs(grid.map[j][i] - grid.map[j + 1][i])

        return smooth

    def getMove(self, grid):
        self.move = []
        start = time.time()
        alpha = -math.inf
        beta = math.inf
        while time.time() - start < self.max_time:
            #for depth in range(4, 10, 2):
            #self.max_depth = depth
            self.maximize(grid, alpha, beta, 0, start)
        return self.move

    def maximize(self, grid, alpha, beta, depth, start):

        moves = grid.getAvailableMoves()
        #print(depth)
        if len(moves) == 0 or depth > self.max_depth or time.time() - start >= self.max_time:  # or t>self.max_time:
            return self.compute_utility(grid)

        grids = {}
        for move in moves:
            grid_clone = grid.clone()
            grid_clone.move(move)
            grids[move] = self.compute_utility(grid_clone)

        for key in grids:#sorted(grids, key=grids.get, reverse=True):
            grid_copy = grid.clone()
            grid_copy.move(key)
            utility = self.minimize(grid_copy, alpha, beta, depth + 1, start)
            # print("max utility %s"%utility)
            if alpha >= beta:
                break

            if utility > alpha:
                alpha = utility
                if depth == 0:
                    self.move = key

        # print("alpha %s"%alpha)
        return alpha

    def minimize(self, grid, alpha, beta, depth, start):

        moves = grid.getAvailableCells()

        if depth > self.max_depth or time.time() - start >= self.max_time:
            return self.compute_utility(grid)

        for move in moves:
            for tile in (4,):  # (2, 4):
                grid_copy = grid.clone()
                grid_copy.insertTile(move, tile)
                utility = self.maximize(grid_copy, alpha, beta, depth + 1, start)
                # print("min utility %s"%utility)
                if beta <= alpha:
                    break

                if utility < beta:
                    beta = utility

        # print("beta %s"%beta)
        return beta
