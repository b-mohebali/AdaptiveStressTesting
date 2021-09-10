#! /usr/bin/python3

from ActiveLearning.dataHandling import readDataset
import repositories as repo 
import pickle 
from samply.hypercube import cvt 
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import *
from ActiveLearning.benchmarks import * 
from sklearn.metrics import accuracy_score, precision_score, recall_score

simConfigFile = './assets/yamlFiles/adaptiveTesting.yaml'
simConfig = simulationConfig(simConfigFile)
pickleName = f'{simConfig.outputFolder}/71/testClf.pickle'
variablesFile = repo.currentDir + '/assets/yamlFiles/ac_pgm_restricted.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

mySpace = SampleSpace(variableList=variables)
dimNames = mySpace.getAllDimensionNames()
# Loading the two classifiers for comparison: 
disRepo = repo.disagreementRepo 
disData = disRepo + '/data'

dataset, labels = readDataset(dataLoc = disData, dimNames = dimNames)


picklesLoc = repo.picklesLoc
benchClfFile = picklesLoc + 'mother_clf_constrained.pickle'
with open(benchClfFile, 'rb') as pickleIn:
    motherClf = pickle.load(pickleIn)
threshold = 0.5 if motherClf.probability else 0 
benchClassifier = TrainedSvmClassifier(motherClf, len(variables), threshold)

adaptiveRepo = repo.adaptRepo12
adaptiveData = adaptiveRepo + '/data'
adaptDataset, adaptLabels = readDataset(adaptiveData, dimNames = dimNames)
adaptiveClf = StandardClassifier(kernel = 'rbf', C =1000)
adaptiveClf.fit(adaptDataset, adaptLabels)

yPredAdaptive = adaptiveClf.predict(dataset)
yPredBench  = benchClassifier.predict(dataset)
adaptAcc = accuracy_score(labels, yPredAdaptive)
benchAcc = accuracy_score(labels, yPredBench)

print('Adaptive classifier accuracy:', adaptAcc)
print('Benchmark classifier accuracy:', benchAcc)
