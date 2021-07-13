#! /usr/bin/python3


from ActiveLearning.dataHandling import *
from ActiveLearning.Sampling import *
from ActiveLearning.simInterface import *
from ActiveLearning.visualization import *
from metricsRunTest import *
from repositories import *
from yamlParseObjects.variablesUtil import *
from yamlParseObjects.yamlObjects import *

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'
repoLoc = adaptRepo9
dataLoc = repoLoc + '/data'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
designSpace = SampleSpace(variableList = variables)
dimNames = designSpace.getAllDimensionNames()
dataset, labels = readDataset(dataLoc, dimNames=dimNames, sampleRange = range(1,51))
print(labels)

# Getting the mean value of all the dimensions: 
means = np.mean(dataset, axis = 0)
print(means)
stds = np.std(dataset, axis = 0)
print(stds)
normalDataset = (dataset - means) / stds

# Using standard Scaler:
print('--------------------------------------------------')
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler = StandardScaler()
# scaler.fit(dataset)
# print(scaler.mean_)
# print(scaler.scale_)
# normal2 = scaler.transform(dataset)
# diff = abs(normal2 - normalDataset)
# print(diff.T)
# print(dataset[0][:])
# print(scaler.transform(dataset[0][:].reshape(1,4)))

# print('--------------------------------------------------')
# ranges = designSpace.getAllDimensionsRanges()
# bounds = designSpace.getAllDimensionBounds()
# lowerBounds = np.min(dataset, axis = 0)
# upperBounds = np.max(dataset,axis = 0)
# print(bounds)
# ranges = upperBounds - lowerBounds
# print(lowerBounds)
# print(upperBounds)
# stds2 = np.std((dataset-lowerBounds) / ranges, axis = 0)
# print(stds2)

print('--------------------------------------------------')
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(dataset)
fittedData = scaler.transform(dataset)
print(np.min(fittedData, axis = 0))
print(np.max(fittedData, axis = 0))

