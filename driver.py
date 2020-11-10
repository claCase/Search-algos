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

        self.children = {}

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
                self.children["Up"] = up_child

            down_child = self.move_down()

            if down_child is not None:
                self.children["Down"] = down_child

            left_child = self.move_left()

            if left_child is not None:
                self.children["Left"] = left_child

            right_child = self.move_right()

            if right_child is not None:
                self.children["Right"] = right_child

        return self.children




class Node():
    def __init__(self, value, type, parent_key=None):
        self.value = value
        self.key = self.generate_key()
        self.cost = 0
        self.children_keys = {}
        self.parent_key = parent_key
        self.depth = 0
        self.h = self.calculate_manhattan_dist()
        self.visited = False
        self.infringe = False
        self.type = type

    def generate_key(self):
        if self.value:
            return hash(self.value)

    def update_visited(self):
        if self.visited == False:
            self.visited = True

    def calc_depth(self, parent_depth):
        if self.parent_key:
            self.depth = parent_depth + 1

    def calc_cost(self):
        if self.type == "bfs":
            self.cost = self.depth
        if self.type == "A*":
            self.cost = self.depth + self.h
        return self.cost

    def add_child(self, node_key, action):
        self.children_keys[node_key] = action

    def get_children(self):
        return self.children.keys()

    def add_parent(self, parent_key):
        self.parent_key = parent_key

    def calculate_manhattan_dist(self):
        """calculate the manhattan distance of a tile"""
        h = 0
        n = int(math.sqrt(len(self.value)))
        # print(n)
        for idx, val in enumerate(self.value):
            if val != 0:
                x = abs(idx % n - val % n)
                y = abs(idx // n - val // n)
                h += x + y
        return h


class Graph():
    def __init__(self):
        self.edges = {}
        self.nodes = {}

    def add_node(self, Node):
        if Node.key not in self.nodes.keys():
            self.nodes[Node.key] = Node

    def check_node(self, node_key):
        if node_key in self.nodes.keys():
            return True
        else:
            return False

    def calc_node_depth(self, node_key):
        if node_key in self.nodes.keys():
            node = self.nodes[node_key]
            if node.parent_key:
                parent_depth = self.nodes[node.parent_key].depth
                node.calc_depth(parent_depth)

    def add_child(self, node_key, action, child_key):
        if node_key in self.nodes.keys() and child_key in self.nodes.keys():
            parent = self.nodes[child_key].parent_key
            # check if already has parent and delete child from parent
            if parent:
                if parent != node_key:
                    print("HAS ALREADY PARENT")
                    if self.nodes[parent].calc_cost() > self.nodes[node_key].calc_cost():
                        self.nodes[node_key].add_child(child_key, action)
                        self.nodes[child_key].add_parent(node_key)
                        self.calc_node_depth(child_key)
                        #self.nodes[child_key].calc_cost()
            else:
                self.nodes[node_key].add_child(child_key, action)
                self.nodes[child_key].add_parent(node_key)
                self.nodes[child_key].calc_cost()


    def get_nodes(self, ):
        for node in self.nodes.keys():
            print(self.nodes[node].key)

    def get_node(self, node_key):
        if node_key in self.nodes.keys():
            return self.nodes[node_key]
        else:
            return False

    def get_edges(self, ):
        for node in self.nodes.keys():
            print("Parent: {}  Child: {}".format(self.nodes[node].parent_key, self.nodes[node].key))

    def get_child(self, node_key):
        if node_key in self.nodes.keys():
            for child in self.nodes[node_key].get_children():
                print(self.nodes[child].key)

    def get_parent(self, node_key):
        node = self.nodes[node_key]
        if node.parent_key:
            return node.parent_key


    def get_path(self, node_key):
        actions = []
        i = 0
        path_cost = 0
        if node_key in self.nodes.keys():
            while self.nodes[node_key].parent_key:
                #print(i)
                i += 1
                node = self.nodes[node_key]
                parent = self.nodes[node.parent_key]
                action = parent.children_keys[node.key]
                actions.append(action)
                path_cost += node.calc_cost()
                node_key = parent.key
        actions = actions[::-1]
        print(actions)
        return actions, path_cost

    def get_max_depth(self):
        max_depth = 0
        for node in self.nodes:
            if node.depth > max_depth:
                max_depth = node.depth
        return max_depth

### Students need to change the method to have the corresponding parameters
def writeOutput(state):
    state.display()


def bfs_search(initial_state):
    i = 0
    fringe = Q.Queue()
    g = Graph()
    fringe.put(initial_state.config)
    n = initial_state.n
    root = Node(initial_state.config, type="bfs")
    g.add_node(root)

    while not fringe.empty():
        state = fringe.get()
        node = Node(state, type="bfs")
        node_g = g.get_node(node.key)
        if node_g:
            node_g.update_visited()
            node_g.infringe = False
        else:
            g.add_node(node)
            node_g = g.get_node(node.key)
            node_g.update_visited()
            node_g.infringe = False

        state = PuzzleState(state, n)
        reached = test_goal(state)
        if reached:
            print("Optimal Found")
            writeOutput(state)
            g.get_path(node.key)
            return True

        elif state is not None:
            expand = state.expand()
            # EXPAND STATES
            for exp in expand.keys():
                child_node = Node(expand[exp].config, type="bfs")
                ingraph = g.get_node(child_node.key)
                if ingraph:
                    node_visit = ingraph.visited
                    infringe = ingraph.infringe
                else:
                    #print("ADDING EXPANDED NODE")
                    g.add_node(child_node)
                    node_visit = False
                    infringe = False

                if not node_visit and not infringe:
                    fringe.put(expand[exp].config)
                    child_node.infringe = True
                    g.add_child(node_g.key, exp, child_node.key)

        i += 1
        #print(i)

    path, path_cost = g.get_path(node.key)
    nodes_expanded = i
    max_depth = g.get_max_depth()
    print("FINSHED")
    return path, path_cost, nodes_expanded, max_depth

def dfs_search(initial_state):
    from queue import LifoQueue
    """DFS search"""

    i = 0
    fringe = LifoQueue()
    fringe.put(initial_state.config)
    n = initial_state.n
    g = Graph()
    root = Node(initial_state.config, type="bfs")
    g.add_node(root)
    while not fringe.empty():
        state = fringe.get()
        node = Node(state, "bfs")
        #visited.append(state)
        node_g = g.get_node(node.key)
        # check if node is in fringe or visited
        if node_g:
            node_g.update_visited()
            node_g.infringe = False
        else:
            g.add_node(node)
            node_g = g.get_node(node.key)
            node_g.update_visited()
            node_g.infringe = False

        state = PuzzleState(state, n)
        reached = test_goal(state)
        if reached:
            print("Optimal Found")
            writeOutput(state)
            g.get_path(node.key)
            return True

        elif state is not None:
            expand = state.expand()
            # EXPAND STATES
            keys_reversed = [*expand.keys()][::-1]
            for exp in keys_reversed:
                #print(exp)
                child_node = Node(expand[exp].config, type="bfs")
                ingraph = g.get_node(child_node.key)
                if ingraph:
                    node_visit = ingraph.visited
                    infringe = ingraph.infringe
                else:
                    #print("ADDING EXPANDED NODE")
                    g.add_node(child_node)
                    node_visit = False
                    infringe = False

                if not node_visit and not infringe:
                    fringe.put(expand[exp].config)
                    child_node.infringe = True
                    g.add_child(node_g.key, exp, child_node.key)
            #quit()
        i += 1
        #print(i)

    path, path_cost = g.get_path(node.key)
    nodes_expanded = i
    max_depth = g.get_max_depth()
    print("FINSHED")
    return path, path_cost, nodes_expanded, max_depth


def A_star_search(initial_state):
    """A * search"""
    import heapq as hq

    i = 0
    fringe = []
    g = Graph()
    root = Node(initial_state.config)
    g.add_node(root)
    fringe.push(fringe, (root.calc_cost(), root.value))
    n = initial_state.n

    while not fringe.empty():
        state = fringe.get()
        node = Node(state, type="A*")
        node_g = g.get_node(node.key)
        if node_g:
            node_g.update_visited()
            node_g.infringe = False
        else:
            g.add_node(node)
            node_g = g.get_node(node.key)
            node_g.update_visited()
            node_g.infringe = False

        state = PuzzleState(state, n)
        reached = test_goal(state)
        if reached:
            print("Optimal Found")
            writeOutput(state)
            g.get_path(node.key)
            return True

        elif state is not None:
            expand = state.expand()
            # EXPAND STATES
            for exp in expand.keys():
                child_node = Node(expand[exp].config, type="A*")
                ingraph = g.get_node(child_node.key)
                if ingraph:
                    node_visit = ingraph.visited
                    infringe = ingraph.infringe
                else:
                    # print("ADDING EXPANDED NODE")
                    g.add_node(child_node)
                    node_visit = False
                    infringe = False

                if not node_visit and not infringe:
                    g.add_child(node_g.key, exp, child_node.key)
                    fringe.push(hq, (child_node.calc_cost(), child_node))
                    child_node.infringe = True

        i += 1
        # print(i)

    g.get_edges()
    g.get_nodes()
    g.get_path(node.key)
    print("FINSHED")


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""

    ### STUDENT CODE GOES HERE ###


def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    x = abs(idx % n - value % n)
    y = abs(idx // n - value // n)
    return x + y


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
