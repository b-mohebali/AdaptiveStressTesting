#! /usr/bin/python3

from ActiveLearning.visualization import setFigureFolder, SaveInformation, plotSpace
from ActiveLearning.benchmarks import TrainedSvmClassifier
from ActiveLearning.dataHandling import getNextSampleNumber, readDataset
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import matplotlib.pyplot as plt 
import repositories as repos
import os 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
    This script is for evaluating the progress of the adaptive sampling scheme on the real time model. 
"""


def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

plotIterations = True 

# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
repoLoc = repos.adaptRepo11

dataLoc = repoLoc + '/data' 
outputFolder = f'{repoLoc}/outputs'
metricFigFolder = f'{setFigureFolder(outputFolder)}/performance_metric_figures'

figFolder = f'{setFigureFolder(outputFolder)}/newFigures'

if not os.path.isdir(figFolder):
    os.mkdir(figFolder)
if not os.path.isdir(metricFigFolder):
    os.mkdir(metricFigFolder)
budgetSize = simConfig.sampleBudget

initialSampleSize = 105 # Due to constraint violating samples being rejected from the initial sample
batchSize = simConfig.batchSize 

initialSampleSize = 80 # Due to constraint violating samples being rejected from the initial sample
# batchSize = simConfig.batchSize 
batchSize = 1
# adaptiveSampleNum = simConfig.sampleBudget - simConfig.initialSampleSize
adaptiveSampleNum = 366 - 80

designSpace = SampleSpace(variableList=variables)
dimNames = designSpace.getAllDimensionNames()

# Calculating the initial change measure:
changeMeasureVector = []
iterationNumber = []
f1 = []
iterNum = 0
dataset, labels = readDataset(dataLoc, dimNames= dimNames, 
                    sampleRange=range(1, initialSampleSize+1))

adaptiveSampleNum = getNextSampleNumber(dataLoc = dataLoc, createFolder=False,count =1)[0] - initialSampleSize-1
print(adaptiveSampleNum)

samplesUsed = []
precision= []
recall = []
clf = StandardClassifier(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)

picklesLoc = repos.picklesLoc
motherClfPickle = picklesLoc + 'mother_clf.pickle'
import pickle
with open(motherClfPickle, 'rb') as pickleIn:
    motherClf = pickle.load(pickleIn)
threshold = 0.5 if motherClf.probability else 0
classifierBench = TrainedSvmClassifier(motherClf, len(variables),threshold)
convSample = ConvergenceSample(space = designSpace, constraints = consVector)
# yTrue = classifierBench.getLabelVec(convSample.samples)

benchRepo = repos.constrainedSample3
benchDataLoc = benchRepo + '/data'
benchData, benchLabels = readDataset(benchDataLoc,dimNames)
yTrue = benchLabels

insigDims = [2,3]
figSize = (32,30)
gridRes = (7,7)
meshRes = 150
print('Figure folder: ', figFolder)
sInfo = SaveInformation(fileName = f'{figFolder}/initial_Plot', 
                        savePDF=False, 
                        savePNG=True)
if plotIterations:                        
    plotSpace(designSpace,
            classifier= clf,
            figsize = figSize,
            meshRes=meshRes,
            showPlot=False,
            showGrid=True,
            gridRes = gridRes,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            constraints=consVector,
            benchmark=None)
    plt.close()

prevClassifier = clf
currentBudget = budgetSize - initialSampleSize
for sampleNum in range(initialSampleSize, budgetSize + 1, batchSize):
    iterNum += 1 
    dataset,labels = readDataset(dataLoc, dimNames, sampleRange = range(1,sampleNum+1))
    clf = StandardClassifier(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    changeMeasure = convSample.getChangeMeasure(classifier=clf, updateLabels= True, percent = False)
    
    yPred = clf.predict(benchData)
    f1Score = f1_score(yTrue, yPred) 
    precisionScore = precision_score(yTrue, yPred)
    recallScore = recall_score(yTrue, yPred) 

    f1.append(f1Score)
    precision.append(precisionScore)
    recall.append(recallScore)
    changeMeasureVector.append(changeMeasure * 100) # Making it percentage.
    iterationNumber.append(iterNum)
    samplesUsed.append(sampleNum)
    print(f'Iteration {iterNum}, Difference: {changeMeasure*100}%, Number of samples used: {sampleNum}')
    currentBudget -= 1
    if plotIterations:
        sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
        plotSpace(designSpace,
                classifier= clf,
                figsize = figSize,
                meshRes=meshRes,
                showPlot=False,
                showGrid=True,
                gridRes = gridRes,
                saveInfo=sInfo,
                insigDimensions=insigDims,
                legend = True,
                constraints=consVector,
                prev_classifier=prevClassifier,
                benchmark=None)
    prevClassifier = clf

    figSize = (7,3)
    # Included in the loop so that we can get the update at any iteration. 
    plt.figure(figsize = figSize)
    plt.plot(iterationNumber[1:], changeMeasureVector[1:])
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Change Measure %')
    plt.savefig(f'{metricFigFolder}/changeMeasure_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = figSize)
    plt.plot(samplesUsed[1:], changeMeasureVector[1:])
    plt.grid(True)
    plt.xlabel('Number of samples used')
    plt.ylabel('Change measure %')
    plt.savefig(f'{metricFigFolder}/changeMeasure_samplesUsed.png')
    plt.close()

    plt.figure(figsize = figSize)
    plt.plot(iterationNumber, f1)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('f1 score')
    plt.savefig(f'{metricFigFolder}/f1_score_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = figSize)
    plt.plot(iterationNumber, precision)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Precision')
    plt.savefig(f'{metricFigFolder}/precision_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = figSize)
    plt.plot(iterationNumber, recall)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Recall')
    plt.savefig(f'{metricFigFolder}/recall_vs_Iteration.png')
    plt.close()
