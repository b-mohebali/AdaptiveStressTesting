#! /usr/bin/python3

from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import time
import matplotlib.pyplot as plt 
from samply.hypercube import halton,cvt, lhs
# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
experFile = './assets/experiments/constrainedSample3.txt'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

designSpace = SampleSpace(variableList=variables)

# Generating a sample for evaluation: 
# n = 1000
# cvtTime = time.time()
# cvtSample = generateInitialSample(designSpace, sampleSize= n, method = InitialSampleMethod.CVT)
# cvtTime = time.time() - cvtTime
# lhsTime = time.time()

# n *= 10000
# lhsSample = generateInitialSample(designSpace, sampleSize= n, method = InitialSampleMethod.LHS)
# lhsTime = time.time() - lhsTime

# print(cvtTime)
# print(lhsTime)

n =100 
dim = 2
cvtSample= cvt(count = n, dimensionality=dim)
lhsSample = lhs(count = n, dimensionality=dim)
haltonSample = halton(count = n, dimensionality=dim)
plt.scatter(cvtSample[:,0], cvtSample[:,1])
plt.grid(True)
plt.show()
