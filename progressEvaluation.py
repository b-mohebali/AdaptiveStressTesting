#! /usr/bin/python3

from ActiveLearning.dataHandling import readDataset
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import time
import matplotlib.pyplot as plt 
from samply.hypercube import halton,cvt, lhs
import repositories as repos
import numpy as np

from activeLearn_PGM import constraint

convVector = [constraint]

# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
experFile = './assets/experiments/constrainedSample3.txt'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
repoLoc = repos.adaptRepo10
dataLoc = repoLoc + '/data' 

figFolder = repos.testRepo15 + '/figures'


initialSampleSize = 87 # Due to constraint violating samples being rejected from the initial sample
# batchSize = simConfig.batchSize 
batchSize = 1
adaptiveSampleNum = simConfig.sampleBudget - simConfig.initialSampleSize


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

convSample = ConvergenceSample(space = designSpace, constraints = convVector)

for sampleNum in range(initialSampleSize + 1, adaptiveSampleNum + initialSampleSize + 1, batchSize):
    iterNum += 1 
    dataset,labels = readDataset(dataLoc, dimNames, sampleRange = range(1,sampleNum+1))
    clf = StandardClassifier(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    changeMeasure = convSample.getChangeMeasure(classifier=clf, updateLabels= True, percent = False)
    f1Score = convSample.getPerformanceMetrics()
    changeMeasureVector.append(changeMeasure)
    iterationNumber.append(iterNum)
    samplesUsed.append(sampleNum)
    print(f'Iteration {iterNum}, Difference: {changeMeasure*100 } %, Number of samples used: {sampleNum}')



plt.figure(figsize = (10,5))
plt.plot(iterationNumber[1:], changeMeasureVector[1:])
plt.grid(True)
plt.xlabel('Iteration number')
plt.ylabel('Change Measure %')
plt.savefig(f'{figFolder}/changeMeasure_vs_Iteration.png')
plt.close()


plt.figure(figsize = (10,5))
plt.plot(samplesUsed[1:], changeMeasureVector)
plt.grid(True)
plt.xlabel('Number of samples used')
plt.ylabel('Change measure %')
plt.savefig(f'{figFolder}/changeMeasure_samplesUsed.png')
plt.close()

