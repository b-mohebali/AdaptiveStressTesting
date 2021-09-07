#! /usr/bin/python3

from ActiveLearning.dataHandling import readDataset
from yamlParseObjects.variablesUtil import * 
from yamlParseObjects.yamlObjects import * 
from metricsInterface import *
from multiprocessing import freeze_support
import time
import matplotlib.pyplot as plt
from ActiveLearning.visualization import *
import repositories as repo
import sys 
import pickle
from sklearn.metrics import accuracy_score

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
    repoLoc = repo.constrainedSample3
    dataLoc = repoLoc + '/data'
    benchDataset,benchLabels = readDataset(dataLoc, dimNames=dimNames)


    benchmarkClf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    benchmarkClf.fit(benchDataset, benchLabels)
    pickleName = repo.picklesLoc + 'mother_clf_constrained.pickle'
    with open(pickleName, 'wb') as motherClf:
        pickle.dump(benchmarkClf, motherClf)

    yPred = benchmarkClf.predict(benchDataset)
    acc = accuracy_score(benchLabels,yPred)
    print('Accuracy:', acc * 100)

    repoLoc = repo.adaptRepo12
    dataLoc = repoLoc + '/data'


    # for _ in range(105, 401):
    #     dataset,labels = readDataset(dataLoc, dimNames=dimNames, sampleRange = range(1,_+1))
    #     clf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    #     clf.fit(dataset, labels)
    #     yPred = clf.predict(benchDataset)
    #     acc = accuracy_score(benchLabels, yPred)
    #     print(f'Obtained accuracy with {_} samples:', acc * 100)
    dataset, labels = readDataset(dataLoc = dataLoc, dimNames = dimNames)
    clf = StandardClassifier(kernel = 'rbf', C = 1000, probability=False)
    clf.fit(dataset, labels)



    designSpace = SampleSpace(variableList = variables)
    designSpace._samples, designSpace._eval_labels = dataset, labels 
    insigDims = [2,3]
    figSize = (47,45)
    gridRes = (11,11)
    meshRes = 100
    outputFolder = f'{repoLoc}/outputs'






    sInfo = SaveInformation(fileName = f'{outputFolder}/testPlot', 
                            savePDF=False, 
                            savePNG=True)
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
            benchmark=benchmarkClf)
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
