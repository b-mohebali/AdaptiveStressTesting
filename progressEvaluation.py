# #! /usr/bin/python3

from ActiveLearning.dataHandling import readDataset
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import time
import matplotlib.pyplot as plt 
from samply.hypercube import halton,cvt, lhs
import repositories as repos

# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
experFile = './assets/experiments/constrainedSample3.txt'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
repoLoc = repos.adaptRepo10
dataLoc = repoLoc + '/data' 

initialSampleSize = 87 # Due to constraint violating samples being rejected from the initial sample
batchSize = simConfig.batchSize 
adaptiveSampleNum = simConfig.sampleBudget - simConfig.itialSampleSize


designSpace = SampleSpace(variableList=variables)
dimNames = designSpace.getAllDimensionNames()
# Generating the sample : 
n = 100 * 5** designSpace.dNum 
print(n)
evalSample = halton(count = n, dimensionality=designSpace.dNum)
evalLabels = np.zeros(shape = (n,), dtype = float)

# Calculating the initial change measure:
changeMeasureVector = []
iterationNumber = [0]

dataset, labels = readDataset(dataLoc, dimNames= dimNames, 
                    sampleRange=range(1, initialSampleSize+1))
print(dataset, labels)


for sampleNum in range(initialSampleSize + 1, adaptiveSampleNum + initialSampleSize + 1):
    pass 



