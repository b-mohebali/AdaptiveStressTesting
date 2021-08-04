# #! /usr/bin/python

from yamlParseObjects.yamlObjects import * 
import os,sys
import time
import yaml 
import matlab.engine
from multiprocessing import Process
from ActiveLearning.dataHandling import resultFileName

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
for p in simConfig.codeBase:
    sys.path.insert(0,p)
    print(p + ' has been added to the path.')

from repositories import *

"""
    NOTE: MATLAB engine is started by and passed to the script that needs the MATLAB engine for anything. The script keeps the engine alive for as long as it needs it. Global definition of MATLAB in this script was tried but did not work.
"""

def setUpMatlab(simConfig: simulationConfig):
    eng = matlab.engine.start_matlab()
    assert len(matlab.engine._engines) > 0
    if simConfig is not None:
        paths = simConfig.matlabPaths
        for p in paths:
            eng.addpath(eng.genpath(p))
            print(f'Directory {p} was added to matlab path.')
    print('MATLAB engine started.')
    return eng

# Testing the function that runs the metrics and saves the label.
def getMetricsResults(dataLocation: str,
                    eng,
                    sampleNumber,
                    metricNames, 
                    figFolderLoc: str = None,
                    procNum: int = 0):
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
            print(f'Running sample at {dataLocation}\n\tSample number: {sampleNum}\n\tFigureFolder:{figFolderLoc}')
            l = getMetricsResults(dataLocation,
                                eng = eng,
                                sampleNumber = sampleNum,
                                metricNames = metricNames, 
                                figFolderLoc = figFolderLoc,
                                procNum = procNum)
            labels.append(l)
        return labels 
    
    # Check to see if the dataLocation is a valid path
    assert os.path.exists(dataLocation)
    
    # Calling the matlab script that handles the metrics run and calculating the 
    # time it took.
    startTime = time.time()
    # NOTE: The number of outputs is 1 plus the number of metrics. 1 is for the label of the sample.
    output = eng.runMetrics(dataLocation, sampleNumber, figFolderLoc, nargout = 1 + len(metricNames))
    endTime = time.time()
    elapsed = endTime - startTime
    processIndicator = f'Process {procNum}: ' if procNum!=0 else ''
    print(f'{processIndicator}Time taken to calculate the metrics: {elapsed}')
    # capturing the label as an integer. 
    label = int(output[0])
    # Unpacking the results of other metrics coming from the matlab evaluation:
    metricValues = list(output[1:]) if len(output) > 1 else []

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
        print(processIndicator, metricName, ': ', float('{:.5f}'.format(metricValues[idx])))
        reportDict[metricName] = metricValues[idx]
    with open(f'{sampleLoc}/{resultFileName}','w') as reportYaml:
        yaml.dump(reportDict, reportYaml)    
    return label


class MetricsProcess:
    """
        This class contains the information needed for running a set of MATLAB metrics
        evaluation. 
        NOTE: Each of the metric processes starts its own MATLAB engine. It is not possible to start a matlab engine and pass it to the started process as MATLAB engine object cannot be serialized using pickle.
    """
    def __init__(self, 
                dataLocation, 
                sampleGroup, 
                figFolder, 
                matlabConfig,
                procNum = 0):
        print(f'Initiating the Processor object for process {procNum}.')
        self.dataLocation = dataLocation
        self.sampleGroup = sampleGroup
        self.metricNames = matlabConfig.metricNames
        self.figFolder = figFolder
        self.config = matlabConfig
        self.processNum = procNum
        print(f'Initialization completed for process {self.processNum}.')

    # Running this function will go through all the samples assigned to this process, evaluating them one by one until all are evaluated and then the process will join with the main process. 
    def runMetrics(self):
        print(f'Starting to run the metrics for process {self.processNum}.')
        start = time.perf_counter()
        # Starting the MATLAB engine by calling the setup function:
        self.engine = setUpMatlab(simConfig=self.config)
        finish = time.perf_counter()
        print(f'MATLAB engine for process {os.getpid()} started in {round(finish - start, 2)} seconds')
        # Calling the MATLAB function that evaluates the sample:
        getMetricsResults(dataLocation = self.dataLocation, 
                eng = self.engine,
                sampleNumber = self.sampleGroup, 
                metricNames = self.metricNames, 
                figFolderLoc = self.figFolder,
                procNum = self.processNum)
        # Shutting down the engine after the work is done:
        self.engine.quit()


def runBatch(dataLocation: str, 
                    sampleGroup: list, 
                    configFile: simulationConfig,
                    figureFolder: str = None, 
                    PN_suggest: int = 1) -> None:
    """
        This function runs the metrics on a group of samples by 
        starting processes, starting separate matlab engines for each 
        and calling the metrics run function. 
    """
    processNumber = min(PN_suggest, len(sampleGroup))
    print(f'The metrics evaluations will be done with {processNumber} processes.')
    sampleGroups = []
    # Dividing the samples into groups, one for each process:
    for proc in range(processNumber): 
        sInd = range(proc, len(sampleGroup), processNumber)
        # Building the sample groups for each process as lists:
        sg = [sampleGroup[ind] for ind in sInd] 
        print(f'Process #{proc+1} sample numbers: {sg}', 'Number of samples:', len(sg))
        sampleGroups.append(sg)
    # Instantiating the metrics process objects, one for each sample group:
    metricProcesses = [
        MetricsProcess(dataLocation = dataLocation, 
                        sampleGroup=sg, 
                        figFolder= figureFolder, 
                        matlabConfig = configFile,
                        procNum = p+1) for p,sg in enumerate(sampleGroups)]
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
# TODO: Get the ranges of the samples from the command line parameters instead of hardcoding it.
def main():
    # dataLocation = 'E:/Data/monteCarlo400'
    dataLocation = 'E:/Data/Sample101/data'
    figFolder = dataLocation + '/figures'
    startingSample = 1
    finalSample = 16
    sampleNumbers = list(range(startingSample,finalSample+1))
    matlabConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    processNumber = 4
    runBatch(dataLocation = dataLocation,
                    sampleGroup=sampleNumbers,
                    configFile=matlabConfig,
                    figureFolder=figFolder,
                    PN_suggest=processNumber)

if __name__=="__main__":
    main()

