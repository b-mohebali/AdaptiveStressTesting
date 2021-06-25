# #! /usr/bin/python3

from matplotlib.pyplot import figure
from ActiveLearning.dataHandling import loadMetricValues, loadVariableValues, reconstructDesignMatrix
import sys

from yaml.loader import SafeLoader 
from yamlParseObjects.variablesUtil import * 
from yamlParseObjects.yamlObjects import * 
from metricsRunTest import *
from multiprocessing import freeze_support
import glob
import yaml
from scipy.linalg import solve
import matplotlib.pyplot as plt
from ActiveLearning.visualization import * 

def analyseFactorScreening(repoLoc, figFolder, metNames, include_bias = False):
    dataLoc = repoLoc + '/data'
    varDescFile = glob.glob(dataLoc + '/*.yaml')[0]
    varNames, descs = loadVars(varDescFile)
    sInfo = SaveInformation(fileName = '', savePDF=True, savePNG=True)
    metVals = loadMetricValues(dataLoc = dataLoc, metricNames=metNames)
    varValues = loadVariableValues(dataLoc=dataLoc, varNames = varNames)
    if include_bias:
        varNames.insert(0,'bias')
        descs['bias'] = 'Bias term'

    H = reconstructDesignMatrix(varValues)
    for metricName in metNames:
        metVal = metVals[metricName]
        x = solve(np.matmul(H.T,H), np.matmul(H.T, np.array(metVal).reshape(len(metVal,))))  
        print(metricName, x) 
        if not include_bias:
            x = x[1:] # Removing the first element that is related to the bias term.
        width = 0.55  # the width of the bars
        b = np.arange(len(varNames))  # the label locations
        plt.figure()
        fig, ax = plt.subplots()
        indices = np.argsort(abs(x))
        print(indices)
        barlist = ax.barh(b,abs(x)[indices], width)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Main effects')
        ax.set_title(metricName)
        ax.set_yticks(b)
        ax.set_yticklabels([descs[varNames[_]] for _ in indices])
        posLabel = False
        negLabel = False
        for _,idx in enumerate(indices):
            if x[idx] < 0 and not negLabel:
                barlist[_].set_label('Negative effect')
                negLabel = True
            elif not posLabel:
                barlist[_].set_label('Positive effect')
                posLabel = True
            barlist[_].set_color('r' if x[idx] < 0 else 'b')
        plt.xticks(rotation=45)
        ax.legend(loc ='lower right')
        plt.grid(True)
        fig.tight_layout()
        if not os.path.isdir(f'{figFolder}/finalResults'):
            os.mkdir(f'{figFolder}/finalResults')
        sInfo.fileName = f'{figFolder}/finalResults/{"".join(c for c in metricName if c.isalnum())}'
        saveFigures(sInfo)
        plt.close()


def main():
    # Setting pu the locations: 
    repoLoc = 'E:/Data/testSample'
    dataLoc = repoLoc + '/data'
    figLoc = repoLoc + '/figures'
    files = glob.glob(dataLoc + '/*.txt')
    experFile = files[0]
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    metNames = simConfig.metricNames

    exper = loadSampleFromTxtFile(experFile)
    print(exper)

    # Normalizing the experiment matrix:
    sampleGroup = list(range(1,17))
    batchSize = 4

    runBatch(dataLocation=dataLoc,
                    sampleGroup=sampleGroup,
                    configFile=simConfig,
                    figureFolder=figLoc,
                    PN_suggest=batchSize)

    # Analysis of the results:
    analyseFactorScreening(repoLoc=repoLoc, 
                        figFolder=figLoc,
                        metNames = metNames, 
                        include_bias=False)
   

def loadVars(varDescFile):
    with open(varDescFile, 'r') as fp:
        yamlString = fp.read()
    vars = []
    descs = {}
    yamlObj =  yaml.load(yamlString, Loader = SafeLoader)
    for var in yamlObj:
        vars.append(var)
        descs[var]= yamlObj[var]
    return vars, descs

# Since we are using multiprocessing we need to have this here: 
if __name__=="__main__":
    freeze_support()
    main()