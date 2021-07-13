#! /usr/bin/python3

from metricsRunTest import runBatch
from ActiveLearning.dataHandling import getNotEvaluatedSamples
from ActiveLearning.Sampling import InitialSampleMethod, SampleSpace, generateInitialSample, getSamplePointsAsDict
import sys
import repositories 
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.simInterface import * 
import os 

def main(run_exper = True, run_eval = True, load_sample = False):
    repoLoc = repositories.monteCarlo2500
    dataLoc = repoLoc + '/data'
    if not os.path.isdir(dataLoc):
        os.mkdir(dataLoc)
    figLoc = repoLoc + '/figures'
    print(repositories.currentDir)
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    modelLoc = repositories.cefLoc + simConfig.modelLocation
    variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'
    experFile = './assets/experiments/monteCarlo2500.txt'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
    designSpace = SampleSpace(variableList=variables)
    dimNames = designSpace.getAllDimensionNames()
    initialSampleSize = 2500 
    # Creating the large sample using CVT method:
    if load_sample:
        formattedSample = loadSampleFromTxtFile(experFile)
    else:
        initialSample = generateInitialSample(space = designSpace,
                                            sampleSize=initialSampleSize,
                                            method = InitialSampleMethod.CVT,
                                            checkForEmptiness=False)
        formattedSample = getSamplePointsAsDict(dimNames, initialSample)
        saveSampleToTxtFile(formattedSample, experFile)
    # Running the sample:
    if run_exper:
        runSample(caseLocation = modelLoc,
                sampleDictList=formattedSample,
                remoteRepo = dataLoc)

    # Evaluation of the sample points in parallel
    processNum = 4
    sampleList = getNotEvaluatedSamples(dataLoc = dataLoc)
    print('Samples not evaluated yet:', sampleList)
    if run_eval:
        runBatch(dataLocation=dataLoc,
                sampleGroup = sampleList,
                configFile = simConfig,
                figureFolder = figLoc,
                PN_suggest=processNum)
        




if __name__=='__main__':
    main(run_exper = False, run_eval=False, load_sample = False)
