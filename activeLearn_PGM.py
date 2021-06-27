#! /usr/bin/python3

from yamlParseObjects.yamlObjects import *
from yamlParseObjects.variablesUtil import *
import os
from profileExample.profileBuilder import * 
from eventManager.eventsLogger import * 
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
from ActiveLearning.visualization import * 
from ActiveLearning.optimizationHelper import GA_Exploiter, GA_Explorer
from ActiveLearning.benchmarks import TrainedSvmClassifier
from sklearn import svm
from ActiveLearning.simInterface import *
from repositories import *
from metricsRunTest import * 

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
modelLoc = repositories.cefLoc + simConfig.modelLocation

"""
Steps of checks for correctness: 
    DONE 1- Run an FFD sample with the 4 variables in a new location with the control object from simulationHelper.py script. -> DONE
    DONE 2- Implement the initial sampling using the combination of the control objects and the developed active learning code. 
    DONE 3- Implement the exploitation part. Save the change measures in each step in case the process is interrupted for any reason.
    DONE 4- Implement the loading of the benchmark classifier trained on the Monte-Carlo data. 
    DONE 5- Run a sample with Visualiation and the benchmark and compare the results.
    6- Calculate the metrics of the classifier such as precision, recall, accuracy, measure of change vs the number of iterations.
    7- Empirically show that the active learner can reach comparable performance with the Monte-Carlo sampling method using a fraction of the process time. 
    8- Improve on the process time using exploration, prallelization, batch sampling.

NOTE 1: Use the currentDir variable from repositories to point to the AdaptiveStressTesting folder. The automation codebase tends to change the working directory during the process and it has to be switched back to use the assets.
"""

print('This is the AC PGM sampling test file. ')
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'


# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

# Setting the main files and locations:
descriptionFile = currentDir + '/assets/yamlFiles/varDescription.yaml'
sampleSaveFile = currentDir + '/assets/experiments/adaptive_sample-400(80)-1.txt'
repoLoc = adaptRepo5
dataLoc = repoLoc + '/data'

# Defining the location of the output files:
outputFolder = f'{repoLoc}/outputs'
figFolder = setFigureFolder(outputFolder)

# Defining the design space and the handler for the name of the dimensions. 
designSpace = SampleSpace(variableList=variables)
dimNames = designSpace.getAllDimensionNames()
initialReport = IterationReport(dimNames)
initialReport.setStart()

#-------------------CREATING SAMPLES BEFORE SIMULATION----------------
# Taking the initial sample based on the parameters of the process. 
initialSamples = generateInitialSample(space = designSpace,
                                        sampleSize=initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False)

### Preparing and running the initial sample: 
formattedSample = getSamplePointsAsDict(dimNames, initialSamples)
saveSampleToTxtFile(formattedSample, sampleSaveFile)
runSample(caseLocation=modelLoc,
            sampleDictList=formattedSample,
            remoteRepo=dataLoc)

#-------------------LOADING SAMPLES----------------------------------
## Loading sample from a pregenerated file in case of interruption:
# print(currentDir)
# formattedSample = loadSampleFromTxtFile(sampleSaveFile)

# runSample(caseLocation=modelLoc,
#             sampleDictList=formattedSample,
#             remoteRepo=dataLoc)
#--------------------------------------------------------------------

#### Running the metrics on the first sample: 
# Forming the sample list which includes all the initial samples:
samplesList = list(range(1, initialSampleSize+1))
### Calling the metrics function on all the samples:
# Using the parallelized metrics evaluation part. 
runBatch(dataLocation=dataLoc,
                sampleGroup=samplesList,
                configFile=simConfig,
                figureFolder=figFolder,
                PN_suggest=4)

# engine = setUpMatlab(simConfig=simConfig)
# getMetricsResults(dataLocation = dataLoc,
#                 eng = engine, 
#                 sampleNumber = samplesList,
#                 metricNames = simConfig.metricNames,
#                 figFolderLoc=figFolder,
#                 procNum = 0)

#### Load the mother sample for comparison:
"""
This part loads a pickled classifier that is trained on the Monte-Carlo sample taken from the 
    model. The purpose for this classifier is to act as a benchmark for the active classifier
    that we are trying to make. 
"""
motherClfPickle = picklesLoc + 'mother_clf.pickle'
classifierBench = None
if os.path.exists(motherClfPickle) and os.path.isfile(motherClfPickle):
    with open(motherClfPickle,'rb') as pickleIn:
        motherClf = pickle.load(pickleIn)
    threshold = 0.5 if motherClf.probability else 0
    classifierBench = TrainedSvmClassifier(motherClf, len(variables), threshold)


#### Load the results into the dataset and train the initial classifier:
dataset, labels = readDataset(dataLoc, dimNames=dimNames)


# updating the space:
designSpace._samples, designSpace._eval_labels = dataset, labels
# Stopping the time measurement for the iteration report:
initialReport.setStop()

#### Iterations of exploitative sampling:
clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)

# Updating the budget:
currentBudget = budget - initialSampleSize


convergenceSample = ConvergenceSample(designSpace)
changeMeasure = [convergenceSample.getChangeMeasure(percent = True,
                        classifier = clf,
                        updateLabels=True)]
samplesNumber = [initialSampleSize]

# Defining the exploiter object:
exploiter = GA_Exploiter(space = designSpace,
                        epsilon = 0.03,
                        batchSize = batchSize,
                        convergence_curve = False,
                        progress_bar = True)

# Defining the explorer object for future use: 
explorer = GA_Explorer(space = designSpace,
                        batchSize=batchSize, 
                        convergence_curve=False,
                        progress_bar=True,
                        beta = 100)

iterationReports = []
# Creating the report object:

iterationReportsFile = f'{outputFolder}/iterationReport.yaml'
iterationNum = 0
# Saving the initial iteration report.
initialReport.iterationNumber = iterationNum 
initialReport.budgetRemaining = currentBudget
initialReport.setChangeMeasure(changeMeasure[0])
initialReport.batchSize = initialSampleSize
initialReport.setMetricResults(labels)
initialReport.setSamples(dataset)

iterationReports.append(initialReport)
saveIterationReport(iterationReports, iterationReportsFile)


## -----------------------------------
# # Setting up the parameters for visualization: 
insigDims = [0,2]
figSize = (12,10)
gridRes = (4,4)
meshRes = 100
sInfo = SaveInformation(fileName = f'{figFolder}/initial_plot', 
                        savePDF=True, 
                        savePNG=True)
"""
TODO: Implementation of the benchmark for this visualizer. 
    The correct way is to use a pickle that contains the classifier 
    trained on the mother sample. Since the results of the evaluation 
    of the mother sample are in the local system.

    NOTE: Done but not tested yet.
"""
plotSpace(designSpace,
            figsize = figSize,
            meshRes = 100,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            benchmark = classifierBench) 
plt.close()

currentSample = formattedSample

## Adaptive Sampling loop:
while currentBudget > 0:
    # Setting up the iteration number:
    iterationNum += 1
    iterReport = IterationReport(dimNames, batchSize = batchSize)
    iterReport.setStart()
    print('Current budget: ', currentBudget, ' samples.')
    
    # Upodating the exploiter object classifier at each iteration. 
    exploiter.clf = clf
    newPointsFound = exploiter.findNextPoints(min(currentBudget, batchSize))
    # Updating the remaining budget:
    currentBudget -= len(newPointsFound)
    # formatting the samples for simulation:
    # NOTE: this is due to the old setting used for the DOE code in the past.
    formattedFoundPoints = getSamplePointsAsDict(dimNames, newPointsFound)
    currentSample.extend(formattedFoundPoints)
    # Getting the number of next samples:
    nextSamples = getNextSampleNumber(dataLoc, 
        createFolder=False, 
        count = len(newPointsFound))
    # running the simulation at the points that were just found:
    """
    TODO: Run all the matlab processes simultaneously. The simulation is done 
            on a point by point basis for now. But the bottleneck of the 
            timing is in the MATLAB metrics calculations. 
    """
    for idx, sample in enumerate(formattedFoundPoints):
        # runSinglePoint(sampleDict = sample,
        #             dFolder = dataFolder,
        #             remoteRepo = dataLoc,
        #             simConfig= simConfig,
        #             sampleNumber = nextSamples[idx],
        #             modelUnderTest=mut)
        ### The new procedure that uses the plasma codebase to run samples:
        runSinglePoint(caseLocation=modelLoc,
                        sampleDict= sample,
                        remoteRepo = dataLoc,
                        sampleNumber = nextSamples[idx])

    # Evaluating the newly simulated samples using MATLAB engine:
    runBatch(dataLocation=dataLoc,
                    sampleGroup=nextSamples,
                    configFile=simConfig,
                    figureFolder=figFolder,
                    PN_suggest=2)
    
    # Updating the classifier and checking the change measure:
    dataset,labels = readDataset(dataLoc, dimNames= dimNames)
    designSpace._samples, designSpace._eval_labels = dataset, labels
    prevClf = clf
    clf = svm.SVC(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    newChangeMeasure = convergenceSample.getChangeMeasure(percent = True,
                        classifier = clf, 
                        updateLabels = True)
    
    # Saving the change measure vector vs the number of samples in each iteration. 
    changeMeasure.append(newChangeMeasure)
    samplesNumber.append(len(labels))
    print('Hypothesis change estimate: ', changeMeasure[-1:], ' %')

    # Visualization of the current state of the space and the classifier
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
    plotSpace(designSpace,
            figsize = figSize,
            meshRes = 80,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            prev_classifier = prevClf,
            benchmark = classifierBench) 
    plt.close() # Just in case. 
    # Saving the iteration report:
    # TODO: Reduce the lines of code that does this job:
    iterReport.setStop()
    iterReport.budgetRemaining = currentBudget
    iterReport.iterationNumber = iterationNum
    iterReport.setMetricResults(labels[-len(newPointsFound):])
    iterReport.setSamples(newPointsFound)
    iterReport.setChangeMeasure(newChangeMeasure)
    iterationReports.append(iterReport)
    saveIterationReport(iterationReports,iterationReportsFile)
    # Saving the experiment file: 
    saveSampleToTxtFile(currentSample, sampleSaveFile)

# Plotting the change measure throughout the process.
plt.figure(figsize = (8,5))
plt.plot(samplesNumber, changeMeasure)
plt.grid(True)
sInfo = SaveInformation(fileName = f'{figFolder}/change_measure', savePDF = True, savePNG = True)
saveFigures(sInfo)
plt.close()



    
    











