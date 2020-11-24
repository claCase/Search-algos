from Grid import Grid
from ComputerAI import ComputerAI
from PlayerAI import PlayerAI
from Displayer import Displayer
from random import randint
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Queue, Pool, cpu_count
from itertools import product

defaultInitialTiles = 2
defaultProbability = 0.9
actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
timeLimit = 0.25
allowance = 0.05



class GameManager:
    def __init__(self, size=4):
        self.grid = Grid(size)
        self.possibleNewTiles = [2, 4]
        self.probability = defaultProbability
        self.initTiles = defaultInitialTiles
        self.computerAI = None
        self.playerAI = None
        self.displayer = None
        self.over = False

    def setComputerAI(self, computerAI):
        self.computerAI = computerAI

    def setPlayerAI(self, playerAI):
        self.playerAI = playerAI

    def setDisplayer(self, displayer):
        self.displayer = displayer

    def updateAlarm(self, currTime):
        if currTime - self.prevTime > timeLimit + allowance:
            self.over = True
        else:
            while time.clock() - self.prevTime < timeLimit + allowance:
                pass

            self.prevTime = time.clock()

    def start(self):
        for i in range(self.initTiles):
            self.insertRandonTile()

        #self.displayer.display(self.grid)

        # Player AI Goes First
        turn = PLAYER_TURN
        maxTile = 0

        self.prevTime = time.clock()

        while not self.isGameOver() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()

            move = None

            if turn == PLAYER_TURN:
                #print("Player's Turn:", end="")
                move = self.playerAI.getMove(gridCopy)
                print(actionDic[move])

                # Validate Move
                if move != None and move >= 0 and move < 4:
                    if self.grid.canMove([move]):
                        self.grid.move(move)

                        # Update maxTile
                        maxTile = self.grid.getMaxTile()
                    else:
                        #print("Invalid PlayerAI Move")
                        self.over = True
                else:
                    #print("Invalid PlayerAI Move - 1")
                    self.over = True
            else:
                #print("Computer's turn:")
                move = self.computerAI.getMove(gridCopy)

                # Validate Move
                if move and self.grid.canInsert(move):
                    self.grid.setCellValue(move, self.getNewTileValue())
                else:
                    #print("Invalid Computer AI Move")
                    self.over = True

            if not self.over:
                pass
                #self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            self.updateAlarm(time.clock())

            turn = 1 - turn
        print(maxTile)
        return maxTile

    def isGameOver(self):
        return not self.grid.canMove()

    def getNewTileValue(self):
        if randint(0, 99) < 100 * self.probability:
            return self.possibleNewTiles[0]
        else:
            return self.possibleNewTiles[1]

    def insertRandonTile(self):
        tileValue = self.getNewTileValue()
        cells = self.grid.getAvailableCells()
        cell = cells[randint(0, len(cells) - 1)]
        self.grid.setCellValue(cell, tileValue)


def setup(weights):
    gameManager = GameManager()
    playerAI = PlayerAI()#weights=weights)
    computerAI = ComputerAI()
    displayer = Displayer()

    gameManager.setDisplayer(displayer)
    gameManager.setPlayerAI(playerAI)
    gameManager.setComputerAI(computerAI)
    max_tile = gameManager.start()
    '''
    if max_tile not in maxs.keys():
        maxs[max_tile] = 1
    else:
        maxs[max_tile] += 1
    '''
    return max_tile


'''for i in range(20):
    weights = np.random.uniform(0,1,6)
    for i in range(10):
        maxs = setup(weights)

    avg_score_ = [.1*maxs[key]*key for key in maxs.keys()]
    if avg_score_ > avg_score:
        avg_score = avg_score_
        best_weights = weights
'''


def get_chunks(iterable, chunks=cpu_count()):
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks)]



def main():
    '''
    values = [np.linspace(0,1,10) for _ in range(6)]
    search_values = product(values)'''
    samples = 1
    iterations = 10
    n_weights = 6
    iterable = [np.random.uniform(0, 1, n_weights) for _ in range(samples)]
    pairs = np.tile(iterable, iterations).reshape(samples*iterations, n_weights)
    print(pairs)
    #chunked = get_chunks(pairs)

    pool = Pool(cpu_count()-2)
    results = pool.map(setup, pairs)
    pool.close()
    pool.join()
    print(results)
    results_freq = []
    for i in range(0, samples*iterations, iterations):
        freq = {}
        results_freq.append(freq)
        for j in range(iterations):
            if results[i + j] not in results_freq[i//iterations].keys():
                results_freq[i//iterations][results[i + j]] = 1
            else:
                results_freq[i//iterations][results[i + j]] += 1
    #print(results_freq)

    avg_results = []
    for i, freq in enumerate(results_freq):
        avg = 0
        for key in freq.keys():
            avg += key * freq[key]/iterations
        avg_results.append((avg, iterable[i], results_freq[i]))

    sorted_results = sorted(avg_results, key= lambda tup:tup[0], reverse=True)
    best = sorted_results[0]
    print(best)
    maxs = best[2]
    plt.bar(range(len(maxs.keys())), list(maxs.values()), align="center")
    plt.xticks(range(len(maxs.keys())), list(maxs.keys()))
    plt.show()


if __name__ == '__main__':
    main()