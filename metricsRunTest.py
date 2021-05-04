#! /usr/bin/python

from yamlParseObjects.yamlObjects import * 
import os,sys
import string
import time
import yaml 
matlabPath = './Matlab'
matlabConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(matlabConfig.name)
for p in matlabConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from repositories import *
import matlab.engine
    
# Defining matlab eng as a global variable:
eng = None

"""
    NOTE: Defining the matlab engine as a global object to be used for the metrics.
    The metrics function does not have to start the matlab engine every time it is 
    called. Detection and connecting the matlab engine to the process fails after 2
    calls to the getMetricsResults function.
"""

# eng = matlab.engine.start_matlab()
# assert len(matlab.engine._engines) > 0
# eng.addpath(matlabPath)
# print('MATLAB engine started.')

def setUpMatlab(simConfig: simulationConfig = matlabConfig):
    eng = matlab.engine.start_matlab()
    assert len(matlab.engine._engines) > 0
    if simConfig is not None:
        paths = simConfig.matlabPaths
        for p in paths:
            eng.addpath(eng.genpath(p))
            print(f'Directory {p} was added to matlab path.')
    print(type(eng))
    print('MATLAB engine started.')
    return eng

# Testing the function that runs the metrics and saves the label.
def getMetricsResults(dataLocation: str,
                        eng,
                        sampleNumber,
                        metricNames, 
                        figFolderLoc: str = None,
                        proc_num = 1):
    # Setting the default location of saving the plots that come out of the metrics
    # evaluation.
    # NOTE: The MATLAB code makes sure that the figures folder exists. If not,
    #       if will create that folder in the specified location.
    if figFolderLoc is None:
        figFolderLoc = dataLocation + '/figures'
    
    # Checking to see if the samples are a list or a single 
    if isinstance(sampleNumber, list):
        labels = []
        for sampleNum in sampleNumber:
            l = getMetricsResults(dataLocation,eng = eng,
                                 sampleNumber = sampleNum,
                                 metricNames=metricNames, 
                                 figFolderLoc = figFolderLoc)
            labels.append(l)
        return labels 
    
    # Check to see if the dataLocation is a valid path
    assert os.path.exists(dataLocation)
    
    # Calling the matlab script that handles the metrics run and calculating the 
    # time it took.
    startTime = time.time()
    output = eng.runMetrics(dataLocation, sampleNumber, figFolderLoc, nargout = 4)
    endTime = time.time()
    elapsed = endTime - startTime
    print('Time taken to calculate the metrics: ', elapsed)
    # capturing the label as an integer. 
    label = int(output[0])
    # Unpacking the results of other metrics coming from the matlab evaluation:
    metricValues = list(output[1:]) if len(output) > 1 else []
    # TODO: Capturing the rest of the metric values that may be useful for the 
    # factor screening in case we do it on the python side.

    # saving the results to a yaml report file.
    sampleLoc = f'{dataLocation}/{sampleNumber}'
    reportDict = {}
    reportDict['elapsed_time_sec'] = float('{:.5f}'.format(elapsed))
    with open(f'{sampleLoc}/variableValues.yaml') as varValues:
        values = yaml.load(varValues, Loader= yaml.FullLoader)
    reportDict['variables'] = values
    reportDict['result_label'] = label
    # capturing the values of the performance metrics:
    for idx, metricName in enumerate(metricNames):
        print(metricName, ': ', metricValues[idx])
        reportDict[metricName] = metricValues[idx]
    with open(f'{sampleLoc}/finalReport.yaml','w') as reportYaml:
        yaml.dump(reportDict, reportYaml)
    
    return label

# Main function for "manual" setting of the range of the samples to be evaluated.
    # TODO: Get the ranges of the samples from the command line parameters instead 
    # of hardcoding it.
def main():
    engine = setUpMatlab()
    # dataLocation = 'E:/Data/adaptiveRepo1'
    dataLocation = adaptRepo2
    figFolder = dataLocation + '/figures'
    startingSample = 1 
    finalSample = 2
    sampleNumbers = list(range(startingSample,finalSample+1))
    getMetricsResults(dataLocation,eng = engine, 
                        sampleNumber = sampleNumbers,
                        metricNames = matlabConfig.metricNames, 
                        figFolderLoc=figFolder)

if __name__=='__main__':
    main()

