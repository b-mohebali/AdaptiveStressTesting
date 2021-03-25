from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import Branin
from ActiveLearning.Sampling import *
import platform
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from geneticalgorithm import geneticalgorithm as ga
from ActiveLearning.optimizationHelper import GeneticAlgorithmSolver as gaSolver

from plotter import *

import time
import numpy as np 

import matlab.engine

simConfig = simulationConfig('./yamlFiles/adaptiveTesting.yaml')

figFolder = 'C:/Users/Behshad/Google Drive/codes/ScenarioGenerator/Figures/MATLAB_figures'

print(simConfig.sampleRepo)
eng = matlab.engine.start_matlab()
for sampleNum in range(5,11):
    z1,z2,z3,z4 = eng.runMetrics(simConfig.sampleRepo,
                                figFolder, 
                                sampleNum, 
                                nargout = 4)
    print(z1,z2,z3,z4)
