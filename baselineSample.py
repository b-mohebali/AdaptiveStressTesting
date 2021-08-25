#! /usr/bin/python3

from metricsRunTest import runBatch
from ActiveLearning.dataHandling import getNotEvaluatedSamples
from ActiveLearning.Sampling import InitialSampleMethod, SampleSpace, generateInitialSample, getSamplePointsAsDict
import sys
import repositories 
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.simInterface import * 
import os 

def constraint(X):
    freq = X[0]
    pulsePower = X[1]
    rampRate = X[2]
    cons = (2 * pulsePower * freq) < rampRate
    return cons
consVector = [constraint]

def main(run_exper = True, run_eval = True, load_sample = False):
    repoLoc = repositories.constrainedSample3
    dataLoc = repoLoc + '/data'
    if not os.path.isdir(dataLoc):
        os.mkdir(dataLoc)
    figLoc = repoLoc + '/figures'
    print(repositories.currentDir)
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    modelLoc = repositories.cefLoc + simConfig.modelLocation
    variablesFile = './assets/yamlFiles/ac_pgm_restricted.yaml'
    experFile = './assets/experiments/constrainedSample3.txt'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
    designSpace = SampleSpace(variableList=variables)
    dimNames = designSpace.getAllDimensionNames()
    print(dimNames)
    initialSampleSize = 2500
    # Creating the large sample using CVT method:
    if load_sample:
        formattedSample = loadSampleFromTxtFile(experFile)
        print(len(formattedSample))
    else:
        initialSample = generateInitialSample(space = designSpace,
                                            sampleSize=initialSampleSize,
                                            method = InitialSampleMethod.CVT,
                                            checkForEmptiness=False,
                                            constraints=consVector,
                                            resample=True)
        formattedSample = getSamplePointsAsDict(dimNames, initialSample)
        saveSampleToTxtFile(formattedSample, experFile)
    # Running the sample:
    if run_exper:
        sampleGroup = list(range(2178,len(formattedSample)+1))
        runSample(caseLocation = modelLoc,
                sampleDictList=formattedSample,
                remoteRepo = dataLoc,
                sampleGroup=sampleGroup)

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
    main(run_exper = False, 
        run_eval=True, 
        load_sample = True)
