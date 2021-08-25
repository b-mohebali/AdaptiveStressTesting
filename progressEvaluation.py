# #! /usr/bin/python3

from ActiveLearning.benchmarks import TrainedSvmClassifier
from ActiveLearning.dataHandling import readDataset
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import time
import matplotlib.pyplot as plt 
from samply.hypercube import halton,cvt, lhs
import repositories as repos
import numpy as np

def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'
experFile = './assets/experiments/constrainedSample3.txt'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
# repoLoc = repos.adaptRepo10

repoLoc = 'E:/Data/adaptiveRepo3'

dataLoc = repoLoc + '/data' 

# figFolder = repos.testRepo15 + '/figures'
figFolder = 'E:Data/prog_figures'

initialSampleSize = 80 # Due to constraint violating samples being rejected from the initial sample
# batchSize = simConfig.batchSize 
batchSize = 1
# adaptiveSampleNum = simConfig.sampleBudget - simConfig.initialSampleSize
adaptiveSampleNum = 366 - 80

designSpace = SampleSpace(variableList=variables)
dimNames = designSpace.getAllDimensionNames()
# Generating the sample : 
n = 100 * 5** designSpace.dNum 
print(n)
evalSample = generateInitialSample(designSpace, sampleSize = n,
                method = InitialSampleMethod.HALTON)
evalLabels = np.zeros(shape = (n,), dtype = float)

# Calculating the initial change measure:
changeMeasureVector = []
iterationNumber = []
f1 = []
iterNum = 0
dataset, labels = readDataset(dataLoc, dimNames= dimNames, 
                    sampleRange=range(1, initialSampleSize+1))
# print(dataset)
# print(labels)
print(len(labels))

samplesUsed = [len(labels)]
clf = StandardClassifier(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)
predLabels = clf.predict(evalSample)
diff = np.sum(np.abs(predLabels - evalLabels))/n
print(diff * 100.0)
print(sum(predLabels)/ n)
print(sum(labels)/initialSampleSize)

# changeMeasureVector.append(diff)
previousLabels = predLabels

picklesLoc = repos.picklesLoc
motherClfPickle = picklesLoc + 'mother_clf.pickle'
import pickle
with open(motherClfPickle, 'rb') as pickleIn:
    motherClf = pickle.load(pickleIn)
threshold = 0.5 if motherClf.probability else 0
classifierBench = TrainedSvmClassifier(motherClf, len(variables),threshold)

# for sampleNum in range(initialSampleSize + 1, adaptiveSampleNum + initialSampleSize + 1, batchSize):
#     iterNum +=1
#     dataset, labels = readDataset(dataLoc, dimNames, 
#                     sampleRange = range(1,sampleNum + 1))
#     clf = StandardClassifier(kernel = 'rbf', C = 1000)
#     clf.fit(dataset, labels)
#     predLabels = clf.predict(evalSample)
#     diff = np.sum(np.abs(predLabels - previousLabels))/ n
    
#     changeMeasureVector.append(diff*100)
#     iterationNumber.append(iterNum)
#     samplesUsed.append(sampleNum)
#     print(f'Iteration {iterNum}, Difference: {diff*100 } %, Number of samples used: {sampleNum}')
#     previousLabels = predLabels

convSample = ConvergenceSample(space = designSpace, constraints = consVector)
sv = classifierBench.classifier.support_vectors_
print(sv)
print(sv.shape)
# for sampleNum in range(initialSampleSize + 1, adaptiveSampleNum + initialSampleSize + 1, batchSize):
# for sampleNum in range(initialSampleSize + 1, 121):
#     iterNum += 1 
#     dataset,labels = readDataset(dataLoc, dimNames, sampleRange = range(1,sampleNum+1))
#     clf = StandardClassifier(kernel = 'rbf', C = 1000)
#     clf.fit(dataset, labels)
#     changeMeasure = convSample.getChangeMeasure(classifier=clf, updateLabels= True, percent = False)
#     f1Score = convSample.getPerformanceMetrics(classifier = clf, benchmark=classifierBench,metricType = PerformanceMeasure.F1_SCORE)
#     f1.append(f1Score)
#     changeMeasureVector.append(changeMeasure)
#     iterationNumber.append(iterNum)
#     samplesUsed.append(sampleNum)
#     print(f'Iteration {iterNum}, Difference: {changeMeasure*100 } %, Number of samples used: {sampleNum}')



# plt.figure(figsize = (10,5))
# plt.plot(iterationNumber[1:], changeMeasureVector[1:])
# plt.grid(True)
# plt.xlabel('Iteration number')
# plt.ylabel('Change Measure %')
# plt.savefig(f'{figFolder}/changeMeasure_vs_Iteration.png')
# plt.close()


# plt.figure(figsize = (10,5))
# plt.plot(samplesUsed[1:], changeMeasureVector)
# plt.grid(True)
# plt.xlabel('Number of samples used')
# plt.ylabel('Change measure %')
# plt.savefig(f'{figFolder}/changeMeasure_samplesUsed.png')
# plt.close()


# plt.figure(figsize = (10,5))
# plt.plot(iterationNumber[1:], f1[1:])
# plt.grid(True)
# plt.xlabel('Iteration number')
# plt.ylabel('f1 score')
# plt.savefig(f'{figFolder}/f1_score_vs_Iteration.png')
# plt.close()