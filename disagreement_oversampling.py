#! /usr/bin/python3
from yamlParseObjects.yamlObjects import *
from yamlParseObjects.variablesUtil import *
import os, copy
import matplotlib.pyplot as plt
from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
from ActiveLearning.visualization import * 
from ActiveLearning.optimizationHelper import GA_Exploiter, GA_Explorer
from ActiveLearning.benchmarks import TrainedSvmClassifier
from ActiveLearning.simInterface import *
from repositories import *
from metricsInterface import * 
from multiprocessing import freeze_support

def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

sampleSaveFile = currentDir + '/assets/experiments/disagreement.txt'

variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_restricted.yaml'
simConfig = simulationConfig(currentDir + '/assets/yamlFiles/ac_pgm_conf.yaml')

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

repoLoc = adaptRepo12
dataLoc = repoLoc + '/data'

# Loading the benchmark classifier based on the CVT sample:
benchClfFile = picklesLoc + 'mother_clf_constrained.pickle'
with open(benchClfFile, 'rb') as pickleIn:
    motherClf = pickle.load(pickleIn)
threshold = 0.5 if motherClf.probability else 0 
benchClassifier = TrainedSvmClassifier(motherClf, len(variables), threshold)

# Getting the adaptive classifier based on the data:
designSpace = SampleSpace(variableList = variables)
dimNames = designSpace.getAllDimensionNames()
dataset, labels = readDataset(dataLoc, dimNames)
adaptiveClf = StandardClassifier(kernel = 'rbf', C =1000)
adaptiveClf.fit(dataset, labels)

# Creating the sample for disagreement region detection: 
sampleSize = 20000
initSample = generateInitialSample(space =designSpace, 
                                    sampleSize = sampleSize, 
                                    method = InitialSampleMethod.HALTON,
                                    checkForEmptiness=False,
                                    constraints = consVector,
                                    resample = False)
benchLabels = benchClassifier.predict(initSample)
adaptLabels = adaptiveClf.predict(initSample)
selectedSample = initSample[benchLabels != adaptLabels, :]

print('Length of the selected samples: ', len(selectedSample))

formattedSelected = getSamplePointsAsDict(dimNames, selectedSample)


for sample in formattedSelected:
    print(sample)
saveSampleToTxtFile(formattedSelected, sampleSaveFile)

