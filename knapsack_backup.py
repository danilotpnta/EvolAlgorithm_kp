import random
import statistics
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

class Parameters:
    def __init__(self, lambdaa, k, iterations):
        self.lambdaa     = lambdaa
        self.mu          = lambdaa * 2
        self.iterations  = iterations
        self.k           = k

class KnapsackProblem:

    def __init__(self, numObject):
        self.values = 2 ** np.random.randn(numObject)
        self.weights = 2 ** np.random.randn(numObject)
        self.capacity = 0.25 * np.sum(self.weights)

''' Candidate Solution Representation '''
class Individual:

    # order in which items are selected
    def __init__(self, kp):
        self.alphaa = 0.05
        # creates an array [0 .. numObjects]
        self.order = np.arange( len(kp.values))
        np.random.shuffle(self.order)


'''Computes the objective value of the given individual
for the given kanpsack problem instance'''

def fitness(kp, ind):
    value = 0.0
    remainCapac = kp.capacity

    for i in ind.order:
        if (kp.weights[i] <= remainCapac):
            value += kp.values[i]
            remainCapac -= kp.weights[i]
            print(f"    - Ind {i} + weight: {kp.weights[i]}")

    print(f"- ObjValue of Ind:  {value}")

    #returns objective value for that specific combination
    return value

def inKnapsack(kp, ind):
    kpItems = list()
    remainCapac = kp.capacity

    for i in ind.order:
        if (kp.weights[i] <= remainCapac):
            kpItems.append(i)
            remainCapac -= kp.weights[i]

    #returns items in kp for specific ind
    return kpItems


'''Solve a knapsack problem instance using an evolutionary algorithm'''
def KnapsackEA(kp):
    lambdaa     = 2
    mu          = lambdaa * 2
    iterations  = 100
    k           = 5

    # create indv1 [0 ....numObject] ... indv2 [0 ....numObject]
    # as many indv1 as lambdaa
    population: List[Individual] = initialize(kp,lambdaa)

    for i in range(iterations):

        '''Recombination p1 & p2'''
        offspring: List[Individual] = list()
        for j in range(mu):
            p1 = selection(kp, population)
            p2 = selection(kp, population)

            offspring.append( recombination(p1, p2) )
            mutate(offspring[j])

        '''Mutation ind = parent'''
        for ind in population:
            mutate(ind)

        '''Elimination'''
        population = elimination(kp, population, offspring)

        # calculate the objective value of each ind from population
        population_fitness  = [fitness(kp, ind) for ind in population]
        print(f"- Mean fitness value population: {statistics.mean(population_fitness)}")
        print(f"- Best fitness value population: {max(population_fitness)}")


'''create a list of individuals'''
def initialize(kp, lambdaa) -> list[Individual]:
    ListOfInd = [ Individual(kp)  for _ in range(lambdaa)]
    # print(ListOfInd)
    return ListOfInd


'''slect rnd two indices of ind & swap them. ind is a list []'''
def mutate(ind):

    # swaps if rand [0,1) < 0.05
    if np.random.rand() < ind.alphaa:
        i1 = random.randint(0, len(ind.order)-1)
        i2 = random.randint(0, len(ind.order)-1)
        ind.order[i1],ind.order[i2] = ind.order[i2], ind.order[i1]

#def recombination(p1: Individual, p2: Individual) --> Individual:
def recombination(kp, p1: Individual, p2: Individual) -> Individual:
    set1 = np.array(inKnapsack(kp, p1))
    set2 = np.array(inKnapsack(kp, p2))

    # Copy inters to offsrping with 100% prob
    #offspring = np.intersect1d(set1, set2)
    offspring = set1[ np.in1d(set1, set2) ] #to conserve the order

    # Copy symdiff to offsrping with 50% prob
    sym_diff = np.setxor1d(set1,set2)]

    for ind in sym_diff:
        if np.random.rand() < 0.5:
            offspring = np.append(offspring, ind)


    all_elem = np.arange( len(kp.values))

    #remianing elements that are not in offspring
    # - here indv that were left from 50% from p1 and p2
    # - here as well elements not present in p1 and p2
    remain = np.setdiff1d(all_elem, offspring)
    np.random.shuffle(offspring)
    np.random.shuffle(remain)

    order = np.concatenate((offspring, remain))

    '''
    alpha = 0 --> 0
    alpha = 1 --> p2
    alpha = 0.5 --> p2 .0.5

    p1*alphaa + (p2*alphaa - p1*alphaa)x

    beta [-0.5 and 3.5 ]
    '''

    beta = 2 * np.random.random() - 0.5
    alpha = p1.alpha + beta * (p2.alpha - p1.alpha)

    return Individual(len(kp.values), alpha, offspring)

''' k-tourament selection '''
def selection(kp, population):
    # select k ind from pupulation
    k = 5
    # select random ind from population
    rnd_ind = random.choices(range(np.size(population, 0)), k = k)
    fit_inds = np.array([fitness(kp, population[i]) for i in rnd_ind])
    best_fit_indv = np.argmax(fit_inds)
    return population[rnd_ind[best_fit_indv]]

''' lmanda + mu selection '''
def elimination(kp, population, offspring):
    combined = np.concatenate(population, offspring)
    #best_fited = map(lambda x: fitness(kp,x) combined)
    best_fited = np.array([fitness(kp, ind) for ind in best_fited])
    # best individuals top lowest fittest bottwom
    best_fited.sort(reverse = True)
    return best_fited[:len(kp.values)]

def print_kp():
    print()
    print(f"THE KNAPSACK PROBLEM: ")
    print(f"- values: {kp.values}")
    print(f"- weights: {kp.weights}")
    print(f"- capacity: {kp.capacity}")
    print()


if __name__ == '__main__':
    # Lambda | k tourament | itr |
    p = Parameters(2, 5, 100)
    numObjects_InKnapsack = 10
    kp = KnapsackProblem(numObjects_InKnapsack)
    print_kp()
    #kpEA = KnapsackEA(kp)


    #checking population & calc fitness of ind
    population = initialize(kp,4)
    i = 0
    for ind in population:
        print(f"- Order ind{[i]}: {ind.order}")
        fitness(kp, ind)
        print(inKnapsack(kp,ind))
        i += 1

    # checking if mutation works
    # for NumMuta in range(40):
    #     mutate(population[1])
    #     print(population[1].order)
