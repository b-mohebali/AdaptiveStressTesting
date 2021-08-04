#! /usr/bin/python3

from ActiveLearning.visualization import setFigureFolder, SaveInformation, plotSpace
from ActiveLearning.benchmarks import TrainedSvmClassifier
from ActiveLearning.dataHandling import getNextSampleNumber, readDataset
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import matplotlib.pyplot as plt 
import repositories as repos
import os 


def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

# Configs: 
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
repoLoc = repos.adaptRepo10
# repoLoc = 'E:/Data/adaptiveRepo3'

dataLoc = repoLoc + '/data' 
# figFolder = repos.testRepo15 + '/figures'
outputFolder = f'{repoLoc}/outputs'
figFolder = f'{setFigureFolder(outputFolder)}/newFigs'
if not os.path.isdir(figFolder):
    os.mkdir(figFolder)
budgetSize = simConfig.sampleBudget

initialSampleSize = 87 # Due to constraint violating samples being rejected from the initial sample
batchSize = 1

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
yTrue = classifierBench.getLabelVec(convSample.samples)


insigDims = [2,3]
figSize = (32,30)
gridRes = (7,7)
meshRes = 200
print('Figure folder: ', figFolder)
sInfo = SaveInformation(fileName = f'{figFolder}/initial_Plot', 
                        savePDF=True, 
                        savePNG=False)
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
for sampleNum in range(initialSampleSize + 1, adaptiveSampleNum + initialSampleSize + 1, batchSize):
# for sampleNum in range(initialSampleSize+1, 120):
    iterNum += 1 
    dataset,labels = readDataset(dataLoc, dimNames, sampleRange = range(1,sampleNum+1))
    clf = StandardClassifier(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    changeMeasure = convSample.getChangeMeasure(classifier=clf, updateLabels= True, percent = False)
    f1Score = convSample.getPerformanceMetrics(classifier = clf, yTrue = yTrue,metricType = PerformanceMeasure.F1_SCORE)
    precisionScore = convSample.getPerformanceMetrics(classifier = clf, yTrue = yTrue,metricType = PerformanceMeasure.PRECISION)
    recallScore = convSample.getPerformanceMetrics(classifier = clf, yTrue = yTrue,metricType = PerformanceMeasure.RECALL)
    f1.append(f1Score)
    precision.append(precisionScore)
    recall.append(recallScore)
    changeMeasureVector.append(changeMeasure * 100) # Making it percentage.
    iterationNumber.append(iterNum)
    samplesUsed.append(sampleNum)
    print(f'Iteration {iterNum}, Difference: {changeMeasure*100 } %, Number of samples used: {sampleNum}')
    currentBudget -= 1
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

    # Included in the loop so that we can get the update at any iteration. 
    plt.figure(figsize = (10,5))
    plt.plot(iterationNumber, changeMeasureVector)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Change Measure %')
    plt.savefig(f'{figFolder}/changeMeasure_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = (10,5))
    plt.plot(samplesUsed, changeMeasureVector)
    plt.grid(True)
    plt.xlabel('Number of samples used')
    plt.ylabel('Change measure %')
    plt.savefig(f'{figFolder}/changeMeasure_samplesUsed.png')
    plt.close()

    plt.figure(figsize = (10,5))
    plt.plot(iterationNumber, f1)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('f1 score')
    plt.savefig(f'{figFolder}/f1_score_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = (10,5))
    plt.plot(iterationNumber, precision)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Precision')
    plt.savefig(f'{figFolder}/precision_vs_Iteration.png')
    plt.close()

    plt.figure(figsize = (10,5))
    plt.plot(iterationNumber, recall)
    plt.grid(True)
    plt.xlabel('Iteration number')
    plt.ylabel('Recall')
    plt.savefig(f'{figFolder}/recall_vs_Iteration.png')
    plt.close()