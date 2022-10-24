import statistics as stats
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

"""
An evolutionary algorithm for solving the Knapsack problem.
     o1, o2, -, ox
along with their values
    v1, v2, -, vx
and their weights
    w1, w2, -, wx
The goal is to maximize the total value of the selected objects, subject to a weight constraint W
"""

class KnapsackProblem:

    def __init__(self, numObject):
        self.values = 2 ** np.random.randn(numObject)
        self.weights = 2 ** np.random.randn(numObject)
        self.capacity = 0.25 * np.sum(self.weights)


# Representing Solutions to ks Problem
# Idea: Put an object as long as the constraint is not satisfied

''' Candidate Solution Representation '''
class Individual:

        # order in which items are selected
        def __init__(self, kp):

            # creates an array [0 .. numObjects]
            self.order = np.arange( len(kp.values))
            np.random.shuffle(self.order)


def fitness(kp, ind):
    value = 0.0
    remainCapac = kp.capacity

#   for i in range(len(ks_prob.values)):
    for i in ind.order:
        if (kp.weights[i] <= remainCapac):
            value += kp.values[i]
            remainCapac -= kp.weights[i]
            print(f"    - Ind {i} + weight: {kp.weights[i]}")

    print(f"- Max Value: {value}")
    return value


if __name__ == '__main__':
    numObject = 5
    kp = KnapsackProblem(numObject)
    print(f"- values: {kp.values}")
    print(f"- weights: {kp.weights}")
    print(f"- capacity: {kp.caprepacity}")

    #passing ind only kp
    ind = Individual(kp)
    print(f"- list: {ind.order}")

    fitness(kp, ind)
