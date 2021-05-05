#! /usr/bin/python

from yamlParseObjects.yamlObjects import * 
import os,sys
import string
import time
import yaml 
from repositories import *
import matlab.engine
import math
from multiprocessing import Process

"""
    NOTE: MATLAB engine is started by and passed to the script that needs the MATLAB 
        engine for anything. The script keeps the engine alive for as long as it 
        needs it. 
        Global definition of MATLAB in this script was tried but did not work.
"""

def setUpMatlab(simConfig: simulationConfig):
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
                    figFolderLoc: str = None):
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
            l = getMetricsResults(dataLocation,
                                eng = eng,
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
"""
    This class contains the information needed for running a set of MATLAB metrics
    evaluation. 
    NOTE: Each of the metric processes starts its own MATLAB engine. It is not 
        possible to start a matlab engine and pass it to the started process
        as MATLAB engine object cannot be serialized using pickle.
"""
class MetricsProcess:
    def __init__(self, 
                dataLocation, 
                sampleGroup, 
                figFolder, 
                matlabConfig):
        self.dataLocation = dataLocation
        self.sampleGroup = sampleGroup
        self.metricNames = matlabConfig.metricNames
        self.figFolder = figFolder
        self.config = matlabConfig

    def runMetrics(self):
        start = time.perf_counter()
        self.engine = setUpMatlab(simConfig=self.config)
        finish = time.perf_counter()
        print(f'MATLAB engine for process {os.getpid()} started in {round(finish - start, 2)} seconds')
        getMetricsResults(dataLocation = self.dataLocation, 
                eng = self.engine,
                sampleNumber = self.sampleGroup, 
                metricNames = self.metricNames, 
                figFolderLoc=self.figFolder)

"""
    This function runs the metrics on a group of samples by 
    starting processes, starting separate matlab engines for each 
    and calling the metrics run function. 
"""
def runMetricsBatch(dataLocation, 
                    sampleGroup, 
                    configFile,
                    figureFolder = None, 
                    processNumber = 1):
    samplePerProc = math.ceil(len(sampleGroup)/processNumber)
    sampleGroups = []
    # Dividing the samples into groups, one for each process:
    for _ in range(processNumber):
        sampleGroups.append(sampleGroup[_*samplePerProc:min((_+1)*samplePerProc,len(sampleGroup))])
    # Instantiating the metrics process objects, one for each sample group:
    metricProcesses = [
        MetricsProcess(dataLocation = dataLocation, 
                        sampleGroup=sg, 
                        figFolder= figureFolder, 
                        matlabConfig = configFile) for sg in sampleGroups]
    processes = []
    # Starting all the processes that call the runMetrics method of the 
    #   metrics process objects:
    for metricProc in metricProcesses:
        p = Process(target = metricProc.runMetrics)
        p.start()
        processes.append(p)
    # Processes join here: 
    for p in processes:
        p.join()
    return 

    
    
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

