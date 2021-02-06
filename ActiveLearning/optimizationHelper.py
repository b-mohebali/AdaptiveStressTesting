from geneticalgorithm import geneticalgorithm as ga
from Sampling import Space

# The wrapper class for the genetic algorithm that solves the exploitation problem:
class GeneticAlgorithmSolver():
    def __init__(self, space: Space, epsilon: float):
        self.space = space
        algoParam = {'max_num_iteration': 100,
                   'population_size':1000,
                   'mutation_probability':0.1,
                   'elit_ratio': 0,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}
        self.gaModel=ga(function = self.objFunction, 
                        dimension = space.dNum, 
                        variable_type = 'real', 
                        variable_boundaries= space.getAllDimensionBounds(),
                        algorithm_parameters=algoParam)
        self.epsilon = epsilon
    
    def objFunction(self, X):
        dist = self.space.nearestPointDistance(X)
        pen = 0
        df = self.space.clf.decision_function(X.reshape(1,len(X)))
        if abs(df) > self.epsilon:
            pen = abs(df) *100
        return -1 * dist + pen 