import random
import statistics
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import sqrt
import time


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
        self.capacity = 0.2 * np.sum(self.weights)

''' Candidate Solution Representation '''
class Individual:

    def __init__(self, kp, alpha = None, order = None ):
        # creates an array [0 .. numObjects]
        self.order = np.arange( len(kp.values))
        np.random.shuffle(self.order)
        #self.alphaa = 0.05
        self.alphaa = max(0.01, 0.1 + 0.02 * np.random.randn())
        if alpha is not None:
            self.alpha = alpha
        if order is not None:
            self.order = order

'''Computes the objective value of the given individual
for the given kanpsack problem instance'''
def fitness(kp, ind):
    value = 0.0
    remainCapac = kp.capacity

    # print(f'My Ind: {ind.order} ')
    for i in ind.order:
        if (kp.weights[i] <= remainCapac):
            value += kp.values[i]
            remainCapac -= kp.weights[i]
            # print(f"    - Ind[{i}] has weight: {kp.weights[i]}")

    # print(f"- ObjValue of Ind:  {value}")

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
def KnapsackEA(kp, p):

    # create indv1 [0 ....numObject] ... indv2 [0 ....numObject]
    # as many indv1 as lambdaa
    population: List[Individual] = initialize(kp, p.lambdaa)

    print("{: >3} {: >15} {: >15} {: >15} {: >15} {: >30}".format(*("i", "Mean Fitness", "Best Fitness", "Carring", "IndFitness","Ind")))
    for i in range(p.iterations):

        '''Recombination p1 & p2'''
        offspring: List[Individual] = list()
        for j in range(p.mu):
            p1 = selection(kp, population, p.k)
            p2 = selection(kp, population, p.k)

            offspring.append( recombination(kp, p1, p2) )
            mutate(offspring[j])
            # here LSO 2) Idea

        '''Mutation of the parents. Here ind = parent'''
        for ind in population:
            mutate(ind)
            # print(f'ind.order before lso:{ind.order}')
            lso(kp, ind) # here LSO 3) idea
            # print(f'ind.order after  lso:{ind.order}')


        '''Elimination'''
        population = elimination(kp, population, offspring, p.lambdaa)

        # calculate the objective value of each ind from population
        population_fitness  = [fitness(kp, ind) for ind in population]
        indexOfind_maxFitness = population_fitness.index(max(population_fitness))
        maxfitInd = population[indexOfind_maxFitness]

        remainCapac = kp.capacity
        value = 0.0
        # print(f'My fitness Ind: {maxfitInd.order} ')
        for ii in maxfitInd.order:
            if (kp.weights[ii] <= remainCapac):
                value += kp.values[ii]
                remainCapac -= kp.weights[ii]

        # print(f"- Carring Capacity:  {kp.capacity - remainCapac}")
        carryngCapacity = kp.capacity - remainCapac
        valuePrint = value

        remainCapac = kp.capacity
        value = 0.0

        # print(f"- Mean fitness value population: {statistics.mean(population_fitness)}")
        # print(f"- Best fitness value population: {max(population_fitness)}")

        print("{: >3} {: >15.5f} {: >15.5f} {: >15.5f} {: >15.5f}".format(*(i, np.mean(population_fitness), max(population_fitness), carryngCapacity, valuePrint ) ))

        # print("{: >3} {: >15.5f} {: >15.5f} {: >15.5f} {: >15.5f} {: >70}".format(*(i, np.mean(population_fitness), max(population_fitness), carryngCapacity, valuePrint,str(maxfitInd.order) ) ))

'''create a list of individuals'''
def initialize(kp, lambdaa) -> list[Individual]:
    ListOfInd = [ Individual(kp)  for _ in range(lambdaa)]
    # print(ListOfInd)
    return ListOfInd

'''select rnd two indices of ind & swap them. ind is a list []'''
def mutate(ind):

    # swaps if rand [0,1) < 0.05
    if np.random.rand() < ind.alphaa:
        i1 = random.randint(0, len(ind.order)-1)
        i2 = random.randint(0, len(ind.order)-1)
        ind.order[i1],ind.order[i2] = ind.order[i2], ind.order[i1]

''' LSO modifies the individual. It improves the order of packing for
the KP problem '''
def lso(kp, ind):
    # print(f'ind.order before lso:{ind.order}')
    bestFitness = fitness(kp, ind)
    # print(f'BestFit before: {bestFitness}')
    copyInd = copy.deepcopy(ind)
    bestIndFound = copy.deepcopy(ind)

    for i in range (1,len(copyInd.order)):
        copyInd.order[0] = ind.order[i]
        copyInd.order[1: i+1] = ind.order[0:i]
        copyInd.order[i+1 :] = ind.order[i+1 :]

        fitValue = fitness(kp, copyInd)
        if fitValue > bestFitness:
            bestFitness = fitValue
            bestIndFound.order[:] = copyInd.order[:]
            # print(f'ind.order after lso: {bestIndFound.order}')

    ind.order[:] = bestIndFound.order[:]
    # print(f'BestFit after:  {fitness(kp,ind)}')


'''selects two parents and produce an offsrpign'''
def recombination(kp, p1: Individual, p2: Individual) -> Individual:
    set1 = np.array(inKnapsack(kp, p1))
    set2 = np.array(inKnapsack(kp, p2))

    # Copy intersection between parents to offsrping with 100% prob
    #offspring = np.intersect1d(set1, set2)
    offspring = set1[ np.in1d(set1, set2) ] #to conserve the order

    # For remaining elements (from both parents) that were not interseption copy
    # to offspring with 50% probability

    # Copy symetric difference or A - B to offspring with 50% prob
    sym_diff = np.setxor1d(set1,set2)

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
    alpha = p1.alphaa + beta * (p2.alphaa - p1.alphaa)

    return Individual( kp, alpha, order)

''' k-tourament selection '''
def selection(kp, population, k):
    # select k ind from pupulation
    k = k
    # select random ind from population
    rnd_ind = random.choices(range(np.size(population, 0)), k = k)
    fit_inds = np.array([fitness(kp, population[i]) for i in rnd_ind])
    best_fit_indv = np.argmax(fit_inds)
    return population[rnd_ind[best_fit_indv]]

''' lmanda + mu selection '''
def elimination(kp, population, offspring, lambdaa):
    combined = np.concatenate((population, offspring))
    newPopulation = list(combined)
    best_fited = np.array([fitness(kp, ind) for ind in combined])
    newPopulation.sort(key = lambda x: fitness(kp,x), reverse = True )
    # print(f'newPopulation:{newPopulation[best_fited]}')
    # print(f'len newPopulation:{len(newPopulation[:lambdaa])}')
    return newPopulation[:lambdaa]

def printKp(kp):
    print(f"\nTHE KNAPSACK PROBLEM: ")
    # print(f"- values: {kp.values}")
    # print(f"- weights: {kp.weights}")
    print(f"- capacity: {kp.capacity}\n")
    heuristic_order = np.arange(len(kp.values))
    heuristic_order_list = list(heuristic_order)
    heuristic_order_list.sort(key=lambda x: kp.values[x] / kp.weights[x], reverse=True)
    heurBest = Individual(kp, 0.0, np.array(heuristic_order_list))
    print(f"HEURISTIC: {fitness(kp, heurBest)}\n" )

if __name__ == '__main__':
    start_time = time.time()
    p = Parameters(200, 5, 50); N = 200  # Lambda | k-Tour | itr | NumbObjects
    kp = KnapsackProblem(N)
    printKp(kp)
    KnapsackEA(kp,p)
    print("--- %s seconds ---" % (time.time() - start_time))

    #checking population & calc fitness of ind
    # population = initialize(kp, p.lambdaa)
    # i = 0
    # for ind in population:
    #     print(f"- Order ind{[i]}: {ind.order}")
    #     fitness(kp, ind)
    #     print(inKnapsack(kp,ind))
    #     i += 1

    # checking if mutation works
    # for NumMuta in range(40):
    #     mutate(population[1])
    #     print(population[1].order)
