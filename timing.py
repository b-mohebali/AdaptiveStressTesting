#! /usr/bin/python3

from ActiveLearning.dataHandling import readDataset
from yamlParseObjects.variablesUtil import * 
from yamlParseObjects.yamlObjects import * 
from metricsRunTest import *
from multiprocessing import freeze_support
import time
import matplotlib.pyplot as plt
from ActiveLearning.visualization import *
import repositories as repo
import sys 

original_stdout = sys.stdout

"""
    Constraint format:
        - Input: a numpy vector representing a point in the space. 
        - output: Boolean indicating whether the constraint is respected or not:
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
    variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_restricted.yaml'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
    dimNames = [var.name for var in variables]
    repoLoc = repo.constrainedSample2
    dataLoc = repoLoc + '/data'
    dataset,labels = readDataset(dataLoc, dimNames=dimNames)
    print(labels)
    print(sum(labels))

    behchmarkClf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    behchmarkClf.fit(dataset, labels)



    repoLoc = repo.adaptRepo10
    dataLoc = repoLoc + '/data'
    dataset,labels = readDataset(dataLoc, dimNames=dimNames)
    print(labels)
    print(sum(labels))

    clf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    clf.fit(dataset, labels)


    designSpace = SampleSpace(variableList = variables)
    designSpace._samples, designSpace._eval_labels = dataset, labels 
    insigDims = [2,3]
    figSize = (32,30)
    gridRes = (7,7)
    meshRes = 200
    outputFolder = f'{repoLoc}/outputs'
    figFolder = setFigureFolder(outputFolder)
    print('Figure folder: ', figFolder)
    sInfo = SaveInformation(fileName = f'{figFolder}/testPlot', 
                            savePDF=True, 
                            savePNG=True)
    plotSpace(designSpace,
            classifier= clf,
            figsize = figSize,
            meshRes=meshRes,
            showPlot=False,
            showGrid=False,
            gridRes = gridRes,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            constraints=consVector,
            benchmark=behchmarkClf)
    plt.close()

    # repoLoc = 'E:/Data/adaptiveRepo4'
    # dataLoc = repoLoc + '/data'
    # figLoc = repoLoc + '/outputs/Figures'
    # batchSizes = [1,2,4,8,16]
    # sampleGroup = list(range(1,81))
    # overallTimes = []
    # sampleTimeAvg = []
    # avgTimePerSample = []
    # for batchSize in batchSizes:
    #     startTime = time.time()
    #     runBatch(dataLocation=dataLoc,
    #                     sampleGroup=sampleGroup,
    #                     configFile=simConfig,
    #                     figureFolder=figLoc,
    #                     PN_suggest=batchSize)
    #     finishTime = time.time()
    #     elapsed = finishTime - startTime
    #     overallTimes.append(elapsed)
    #     avgTimePerSample.append(elapsed/80)

    #     dataset,labels, times = readDataset(dataLoc, dimNames=dimNames, includeTimes=True)        
    #     sampleTimeAvg.append(np.mean(times))
    #     with open(repoLoc + '/timeReport.txt', 'a') as reportFile: 
    #         sys.stdout = reportFile
    #         print('------------------------------------------')
    #         print('Number of processes:', batchSize)
    #         print('Overall time:', elapsed)
    #         print('avg time per sample: ', elapsed / 80)
    #         print('Sample time Avg:', np.mean(times))
    #         sys.stdout = original_stdout
    #         print('------------------------------------------')
    #         print('Number of processes:', batchSize)
    #         print('Overall time:', elapsed)
    #         print('avg time per sample: ', elapsed / 80)
    #         print('Sample time Avg:', np.mean(times))
    # origTime = overallTimes[0]
    # speedups = [_/origTime for _ in overallTimes]
    # plt.figure()
    # plt.plot(batchSizes, speedups)
    # plt.grid(True)
    # plt.xlabel('Number of processes')
    # plt.ylabel('Speed up')
    # plt.show()
    # return 



if __name__=='__main__':
    freeze_support()
    main()