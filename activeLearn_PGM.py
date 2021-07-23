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
from metricsRunTest import * 
from multiprocessing import freeze_support

"""
    Constraint format:
        - Input: a numpy vector representing a point in the space. 
        - output: Boolean indicating whether the constraint is respected at that point or not:
            True: Constraint Respected.
            False: Constraint Violated.
        
"""
def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

def main():
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    print(simConfig.name)
    modelLoc = cefLoc + simConfig.modelLocation

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
    variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_restricted.yaml'



    #-------------------SETTINGS OF THE PROCESS---------------------------
    includeBenchmarkSample = False
    loadInitialSample = False
    runInitialSample = True 
    discardEvaluations = False

    #---------------------------------------------------------------------

    # Extracting the hyperparameters of the analysis:
    budget = simConfig.sampleBudget
    batchSize = simConfig.batchSize
    initialSampleSize = simConfig.initialSampleSize

    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

    # Setting the main files and locations:
    descriptionFile = currentDir + '/assets/yamlFiles/varDescription.yaml'
    sampleSaveFile = currentDir + '/assets/experiments/adaptive_sample-400(80)-1_Num10.txt'
    repoLoc = adaptRepo10
    dataLoc = repoLoc + '/data'
    if not os.path.isdir(dataLoc):
        os.mkdir(dataLoc)
    # Copying the config files to the data folder:
    copyDataToremoteServer(dataLoc, variablesFile, isFolder = False)
    copyDataToremoteServer(dataLoc, descriptionFile, isFolder = False)


    # Defining the location of the output files:
    outputFolder = f'{repoLoc}/outputs'
    figFolder = setFigureFolder(outputFolder)

    # Defining the design space and the handler for the name of the dimensions. 
    designSpace = SampleSpace(variableList=variables)
    dimNames = designSpace.getAllDimensionNames()
    dimDescs = designSpace.getAllDimensionDescriptions()
    initialReport = IterationReport(dimDescs)
    initialReport.setStart()

    #-------------------CREATING SAMPLES BEFORE SIMULATION----------------
    # Taking the initial sample based on the parameters of the process. 
    if not loadInitialSample:
        initialSamples = generateInitialSample(space = designSpace,
                                                sampleSize=initialSampleSize,
                                                method = InitialSampleMethod.CVT,
                                                checkForEmptiness=False,
                                                constraints=consVector)

        ### Preparing and running the initial sample: 
        formattedSample = getSamplePointsAsDict(dimNames, initialSamples)
        saveSampleToTxtFile(formattedSample, sampleSaveFile)
        # runSample(caseLocation=modelLoc,
        #             sampleDictList=formattedSample,
        #             remoteRepo=dataLoc)

    #-------------------LOADING SAMPLES----------------------------------
    ## Loading sample from a pregenerated file in case of interruption:
    else: 
        print(currentDir)
        formattedSample = loadSampleFromTxtFile(sampleSaveFile)

    if runInitialSample: 
        runSample(caseLocation=modelLoc,
                    sampleDictList=formattedSample,
                    remoteRepo=dataLoc)
    #--------------------------------------------------------------------

    #### Running the metrics on the first sample: 
    # Forming the sample list which includes all the initial samples:
    samplesList = []
    if discardEvaluations:
        samplesList = list(range(1, initialSampleSize+1))
    else:
        print('Fiding the samples that are not evaluated yet. ')
        samplesList = getNotEvaluatedSamples(dataLoc = dataLoc)
    print('Not evaluated samples:', samplesList)
    ### Calling the metrics function on all the samples:
    # Using the parallelized metrics evaluation part. 
    runBatch(dataLocation=dataLoc,
                    sampleGroup=samplesList,
                    configFile=simConfig,
                    figureFolder=figFolder,
                    PN_suggest=4)

    #### Load the mother sample for comparison:
    """
        This part loads a pickled classifier that is trained on the Monte-Carlo sample taken from the model. The purpose for his classifier is to act as a benchmark for the active classifier that we are trying to make. 

        NOTE: This part is only used when the 'includeBenchmarkSample' setting is activated. Otherwise the benchmark classifier is not included in the plots as well.
    """
    motherClfPickle = picklesLoc + 'mother_clf.pickle'
    classifierBench = None

    if includeBenchmarkSample and os.path.exists(motherClfPickle) and os.path.isfile(motherClfPickle):
        with open(motherClfPickle,'rb') as pickleIn:
            motherClf = pickle.load(pickleIn)
        threshold = 0.5 if motherClf.probability else 0
        classifierBench = TrainedSvmClassifier(motherClf, len(variables), threshold)

    #### Load the results into the dataset and train the initial classifier:
    dataset, labels = readDataset(dataLoc, dimNames=dimNames)

    # updating the space with the new samples:
    designSpace._samples, designSpace._eval_labels = dataset, labels
    # Stopping the time measurement for the iteration report:
    initialReport.setStop()

    #### Iterations of exploitative sampling:
    # using the custom classifier that also trains a standard scaler with the data as well. 
    clf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    clf.fit(dataset, labels)

    # Updating the budget:
    # NOTE: The update must be done based on the number of accepted samples and not the number of initial sample size. 
    lastSimulated = getLastSimulatedSampleNumber(dataLoc = dataLoc)
    currentBudget = budget - lastSimulated

    convergenceSample = ConvergenceSample(designSpace)
    # This vector holds all the values for the change measure and will be used for monitoring and plotting later. 
    changeMeasure = [convergenceSample.getChangeMeasure(percent = True,
                            classifier = clf,
                            updateLabels=True)]
    # This vector is the x axis of the change measure vector in the upcoming plots.
    samplesNumber = [initialSampleSize]

    # Defining the exploiter object:
    exploiter = GA_Exploiter(space = designSpace,
                            epsilon = 0.03,
                            clf = clf,
                            batchSize = batchSize,
                            convergence_curve = False,
                            progress_bar = True,
                            constraints = consVector)

    # Defining the explorer object for future use: 
    # TODO: Include the explorer in the analysis
    # explorer = GA_Explorer(space = designSpace,
    #                         batchSize=batchSize, 
    #                         convergence_curve=False,
    #                         progress_bar=True,
    #                         beta = 100)

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
    insigDims = [2,3]
    figSize = (32,30)
    gridRes = (7,7)
    meshRes = 200
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
            meshRes = meshRes,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            benchmark = classifierBench,
            constraints = consVector) 
    plt.close()

    currentSample = formattedSample

    ## Adaptive Sampling loop:
    while currentBudget > 0:
        # Setting up the iteration number:
        iterationNum += 1
        iterReport = IterationReport(dimDescs, batchSize = batchSize)
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
        print('Next samples:', nextSamples)
        # running the simulation at the points that were just found:
        """
        TODO: Run all the matlab processes simultaneously. The simulation is done 
                on a point by point basis for now. But the bottleneck of the 
                timing is in the MATLAB metrics calculations. 
        """
        for idx, sample in enumerate(formattedFoundPoints):
            ### The new procedure that uses the plasma codebase to run samples:
            runSinglePoint(caseLocation=modelLoc,
                            sampleDict= sample,
                            remoteRepo = dataLoc,
                            sampleNumber = nextSamples[idx])

        # Evaluating the newly simulated samples using MATLAB engine:
        samplesGroup = getNotEvaluatedSamples(dataLoc = dataLoc)
        runBatch(dataLocation=dataLoc,
                sampleGroup=samplesGroup,
                configFile=simConfig,
                figureFolder=figFolder,
                PN_suggest=4)
        
        # Updating the classifier and checking the change measure:
        dataset,labels = readDataset(dataLoc, dimNames= dimNames)
        designSpace._samples, designSpace._eval_labels = dataset, labels
        prevClf = copy.deepcopy(clf)
        clf = StandardClassifier(kernel = 'rbf', C = 1000)
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
                meshRes = meshRes,
                classifier = clf,
                gridRes = gridRes,
                showPlot=False,
                saveInfo=sInfo,
                insigDimensions=insigDims,
                legend = True,
                prev_classifier = prevClf,
                benchmark = classifierBench,
                constraints= consVector) 
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


if __name__=='__main__':
    freeze_support()
    main()







