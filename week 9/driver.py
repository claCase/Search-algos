import os
import sys
import numpy as np
import queue
from copy import deepcopy
from itertools import product


class Sudoku:
    def __init__(self, initial_config):
        self.board = np.asarray([int(i) for i in initial_config])
        # board = board.reshape(9, 9)
        self.rows = 'ABCDEFGHI'
        self.cols = '123456789'
        self.digits = '123456789'
        self.variables = {}
        # coordinates of cells in grid
        self.grid_coord = product(self.rows, self.cols)
        self.sudoku_board = dict(zip(self.grid_coord, self.board))
        self.rows_all_diff = [p for row in self.rows for p in product(row, self.cols)]
        self.cols_all_diff = [p for col in self.cols for p in product(self.rows, col)]
        self.block_add_diff = [p
                               for i in range(0, 9, 3)
                               for j in range(0, 9, 3)
                               for row in self.rows[i:i + 3]
                               for col in self.cols[j:j + 3]
                               for p in product(row, col)
                               ]

        self.tot_constr = set(self.rows_all_diff + self.cols_all_diff + self.block_add_diff)
        self.units = dict((pos, [u for u in self.tot_constr if u in self.tot_constr]) for pos in self.grid_coord)


class CSP:
    def __init__(self):
        self.V = {}
        self.D = {}
        self.C = {}
        self.A_partial = {}
        self.rows = 'ABCDEFGHI'
        self.cols = '123456789'
        self.digits = '123456789'
        self.grid = ["%s%s" % (row, col) for row, col in product(self.rows, self.cols)]

        self.rows_all_diff = [["%s%s" % (x, y) for x, y in product(row, self.cols)]
                              for row in self.rows]

        self.cols_all_diff = [["%s%s" % (x, y) for x, y in product(self.rows, col)]
                              for col in self.cols]

        self.block_all_diff = [["%s%s" % (row, col) for row in self.rows[i:i + 3] for col in self.cols[j:j + 3]]
                               for j in range(0, 9, 3)
                               for i in range(0, 9, 3)]

        self.total_diff = self.rows_all_diff + self.cols_all_diff + self.block_all_diff

    def add_constraint(self, variable1_key, variable2_key):
        self.C[(variable1_key, variable2_key)] = lambda x, y: x != y

    def get_constraint(self, variable1_key, variable2_key):
        return self.C[(variable1_key, variable2_key)]

    def add_variable(self, variable_key):
        if variable_key not in self.V.keys():
            self.V[variable_key] = set()
            self.D[variable_key] = []

    def add_domain(self, variable, domain: str):
        self.D[variable].append(domain)

    def get_domain(self, variable):
        return self.D[variable]

    def remove_domain(self, variable, domain):
        self.D[variable].remove(domain)

    def get_neighbour(self, Xi: str):  # ,Xj: str):
        for xi, xj in self.C.keys():
            if Xi == xi and xj not in self.V[Xi]:  # and xj not in self.V[Xi]:
                self.V[Xi] = self.V[Xi].union({xj})
        return self.V[Xi]

    def initial_config(self, config):
        for value, variable in zip(config, self.grid):
            self.add_variable(variable)
            self.A_partial[variable] = value
            if value == "0":
                for d in self.digits:
                    self.add_domain(variable, d)
            else:
                self.add_domain(variable, value)

    def initialize_C(self):
        count = 0
        if self.V and self.D:
            for i in range(9):
                for j in range(1, 9):
                    for constr in self.total_diff:
                        if self.grid[count] in constr:
                            for c in constr:
                                if c != self.grid[count] \
                                        and (self.grid[count], c) not in self.C.keys() \
                                        and (c, self.grid[count]) not in self.C.keys():
                                    # print("Adding Constraint")
                                    self.add_constraint(self.grid[count], c)
                                    self.add_constraint(c, self.grid[count])
                    count += 1
                count += 1

    def ac3(self, D):
        q = queue.Queue()
        for arc in self.C.keys():
            q.put(arc)

        i = 0
        while not q.empty():
            i += 1
            Xi, Xj = q.get()
            # print(Xi, Xj)
            # print("%s before %s"%(Xi, D[Xi]))
            if self.revise(D, Xi, Xj):
                #print("%s after %s"%(Xi, D[Xi]))
                if len(D[Xi]) == 0:
                    return False
                for Xk in (self.get_neighbour(Xi) - {Xj}):
                    q.put((Xk, Xi))
        return True

    def revise(self, D: dict, Xi: str, Xj: str) -> bool:
        revised = False
        for x in D[Xi]:
            consistency = [self.C[(Xi, Xj)](x, y) for y in D[Xj]]
            if not any(consistency):
                D[Xi].remove(x)
                revised = True
        return revised


def back_tracking_search(csp):
    csp_copy = deepcopy(csp)
    csp_copy.ac3(csp_copy.D)
    return recursive_back_tracking({}, csp_copy, 0)


def recursive_back_tracking(assignment, csp, i):
    #print("Backtracking iter %i" % i)
    if test_goal(csp):
        return assignment, csp

    var = select_unassigned(csp.A_partial, csp.D, strategy="mrv")

    for value in csp.D[var]:
        A_ = deepcopy(csp.A_partial)
        D_ = deepcopy(csp.D)
        A_[var] = value
        D_[var] = [value]
        csp2 = deepcopy(csp)
        csp2.A_partial = A_
        csp2.D = D_
        if csp.ac3(D_):
            assignment[var] = value
            result = recursive_back_tracking(assignment, csp2, i + 1)
            if result:
                return result
            assignment[var] = ""
    return False


def select_unassigned(A_partial, D:dict, strategy="mrv"):
    if strategy == "mrv":
        mrv = None
        mrv_val = 10
        for key in A_partial:
            if A_partial[key] == "0":
                if len(D[key]) < mrv_val:
                    mrv = key
        return mrv
    else:
        for key in A_partial:
            if A_partial[key] == "0":
                return key


def complete(A):
    for a in A.keys():
        if A[a] == "0":
            return False
        else:
            return True


def test_goal(csp):
    for c in csp.D.keys():
        if len(csp.D[c]) != 1:
            return False
    return True




def check_sudokus():
    trues = []
    trues_bts = []
    with open("sudokus_start.txt", "r") as file:
        with open("sudokus_finish.txt", "r") as file_output:
            i = 0
            line = file.readline()
            while line:
                line_output = file_output.readline()
                print(i)
                print(line)
                i +=1
                csp = CSP()
                csp_bts = CSP()
                # print(line)
                csp.initial_config(line)
                csp_bts.initial_config(line)
                # print(csp.D)
                csp.initialize_C()
                csp_bts.initialize_C()
                # print(csp.C.keys())
                ac3 = csp.ac3(csp.D)
                all_ones = all([1 if len(csp.D[key]) == 1 else 0 for key in csp.D])
                trues.append(all_ones)
                assignment, csp_bts = back_tracking_search(csp_bts)
                bts_res = [False if int(output)!=int(*d) else True for output, d in zip(line_output, csp_bts.D.values())]
                if all(bts_res):
                    trues_bts.append(True)
                else:
                    trues_bts.append(False)

                print(trues)
                print(trues_bts)
                line = file.readline()

#check_sudokus()

sudoku_input = sys.argv[1]
csp = CSP()
csp.initial_config(sudoku_input)
csp.initialize_C()
csp_ac3 = deepcopy(csp)
ac3 = csp_ac3.ac3(csp_ac3.D)
all_ones = all([1 if len(csp_ac3.D[key]) == 1 else 0 for key in csp_ac3.D])

if os.path.exists("output.txt"):
    print("path exists")
    os.remove("output.txt")

if all_ones:
    with open("output.txt", "a") as file:
        solution = ""
        for value in csp_ac3.D.values():
            solution += "%i"%int(*value)
        solution += " AC3 \n"
        file.write(solution)

assignment, csp = back_tracking_search(csp=csp)

with open("output.txt", "a") as file:
    output = ""
    for o in csp.D.values():
        output += "%i" % int(*o)
    output += " BTS"
    file.write(output)
