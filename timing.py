from matplotlib.pyplot import figure
from ActiveLearning.dataHandling import loadMetricValues, loadVariableValues, readDataset, reconstructDesignMatrix
import repositories
from ActiveLearning.simInterface import runExperiment
from yaml.loader import SafeLoader 
from yamlParseObjects.variablesUtil import * 
from yamlParseObjects.yamlObjects import * 
from metricsRunTest import *
from multiprocessing import freeze_support
import time
from scipy.linalg import solve
import matplotlib.pyplot as plt
from ActiveLearning.visualization import *
import sys 

original_stdout = sys.stdout

def main():
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
    dimNames = [var.name for var in variables]
    repoLoc = 'E:/Data/adaptiveRepo4'
    dataLoc = repoLoc + '/data'
    figLoc = repoLoc + '/outputs/Figures'
    batchSizes = [1,2,4,8,16]
    sampleGroup = list(range(1,81))
    overallTimes = []
    sampleTimeAvg = []
    avgTimePerSample = []
    for batchSize in batchSizes:
        startTime = time.time()
        runBatch(dataLocation=dataLoc,
                        sampleGroup=sampleGroup,
                        configFile=simConfig,
                        figureFolder=figLoc,
                        PN_suggest=batchSize)
        finishTime = time.time()
        elapsed = finishTime - startTime
        overallTimes.append(elapsed)
        avgTimePerSample.append(elapsed/80)

        dataset,labels, times = readDataset(dataLoc, dimNames=dimNames, includeTimes=True)        
        sampleTimeAvg.append(np.mean(times))
        with open(repoLoc + '/timeReport.txt', 'a') as reportFile: 
            sys.stdout = reportFile
            print('------------------------------------------')
            print('Number of processes:', batchSize)
            print('Overall time:', elapsed)
            print('avg time per sample: ', elapsed / 80)
            print('Sample time Avg:', np.mean(times))
            sys.stdout = original_stdout
            print('------------------------------------------')
            print('Number of processes:', batchSize)
            print('Overall time:', elapsed)
            print('avg time per sample: ', elapsed / 80)
            print('Sample time Avg:', np.mean(times))
    origTime = overallTimes[0]
    speedups = [_/origTime for _ in overallTimes]
    plt.figure()
    plt.plot(batchSizes, speedups)
    plt.grid(True)
    plt.xlabel('Number of processes')
    plt.ylabel('Speed up')
    plt.show()
    return 



if __name__=='__main__':
    main()