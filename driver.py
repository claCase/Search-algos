"""
Skeleton code for Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
"""

import queue as Q

import time

import psutil

import sys

import math

#### SKELETON CODE ####

## The Class that Represents the Puzzle


class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n * n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        self.goal = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = i // self.n

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print(line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = (
                new_config[target],
                new_config[blank_index],
            )

            return PuzzleState(
                tuple(new_config),
                self.n,
                parent=self,
                action="Left",
                cost=self.cost + 1,
            )

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = (
                new_config[target],
                new_config[blank_index],
            )

            return PuzzleState(
                tuple(new_config),
                self.n,
                parent=self,
                action="Right",
                cost=self.cost + 1,
            )

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = (
                new_config[target],
                new_config[blank_index],
            )

            return PuzzleState(
                tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1
            )

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = (
                new_config[target],
                new_config[blank_index],
            )

            return PuzzleState(
                tuple(new_config),
                self.n,
                parent=self,
                action="Down",
                cost=self.cost + 1,
            )

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children


class my_queue(Q.Queue):
    """docstring for my_queue"""

    def __init__(self):
        super(my_queue, self).__init__()

    def __contains__(self, item):
        with self.mutex:
            for q in self.queue:
                if q.config == item.config:
                    return True
            return False

### Students need to change the method to have the corresponding parameters
def writeOutput(state):
    state.display()


def bfs_search(initial_state):
    """BFS search"""
    i = 0
    visited = []
    fringe = my_queue()
    fringe.put(initial_state)
    while not fringe.empty():
        state = fringe.get()
        visited.append(state.config)
        reached = test_goal(state)
        if reached:
            print("Optimal Found")
            writeOutput(state)
            return True
        elif state is not None:
            expand = state.expand()
            for exp in expand:
                if exp.config not in visited and not exp in fringe:
                    fringe.put(exp)
        i += 1
        print(i)
    print("FINSHED")


def dfs_search(initial_state):
    """DFS search"""
    def check_in_fringe(elem, fringe):
        '''for f in fringe:
            if f.config == elem.config:
                print("True")
                return True
        else:
            return False'''
        if elem in fringe:
            print("TRUE\n\n\n")
            return True
        else:
            return False


    i = 0
    visited = []
    fringe = []
    n = initial_state.n
    fringe.append(initial_state.config)
    while fringe:
        state = fringe.pop()
        visited.append(state)
        state = PuzzleState(state, n)
        reached = test_goal(state)
        if reached:
            print("Optimal Found")
            writeOutput(state)
            return True
        elif state is not None:
            expand = state.expand()
            for exp in expand:
                if exp.config not in visited and exp.config not in fringe:
                    fringe.append(exp.config)
        i += 1
        print(i)
    print("FINSHED")


def A_star_search(initial_state):
    """A * search"""
    import heapq as hq

    pq = []

def calculate_total_cost(state):

    """calculate the total estimated cost of a state"""

    ### STUDENT CODE GOES HERE ###


def calculate_manhattan_dist(idx, value, n):

    """calculate the manhattan distance of a tile"""
    x = abs(idx % n - value % n)
    y = abs(idx // n - value // n)
    return x + y

'''
def goal_state(n: int):
    puzl = []
    for i in range(n * n):
        puzl.append(i)
    return PuzzleState(tuple(puzl), n)


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    goal = goal_state(puzzle_state.n)
    if puzzle_state.config == goal.config:
        return True
    else:
        return False
'''
##########
def goal_state(n: int):
    puzl = []
    for i in range(n * n):
        puzl.append(i)
    return tuple(puzl)


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    goal = goal_state(puzzle_state.n)
    if puzzle_state.config == goal:
        return True
    else:
        return False

# Main Function that reads in Input and Runs corresponding Algorithm


def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")


if __name__ == "__main__":

    main()
