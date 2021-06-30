# # ! /usr/bin/python3

from ActiveLearning.dataHandling import loadMetricValues, loadVariableValues, readDataset, reconstructDesignMatrix
import repositories
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
    # Plotting the dot diagrams for each variable: 
    for var in varNames:
        varDesc = descs[var]
        varFigDir = repoLoc + f'/figures/{var}'
        if not os.path.isdir(varFigDir):
            os.mkdir(varFigDir)
        varVal = varValues[var]
        varMax = max(varValues[var])
        varMin = min(varValues[var])
        for metName in metNames:
            met = metVals[metName]
            minMean,maxMean = np.mean(met[varVal==varMin]),np.mean(met[varVal==varMax])
            plt.figure(figsize = (10,5))
            plt.scatter(varVal, met, s = 4, label = 'Data points')
            plt.scatter([varMin, varMax],[minMean, maxMean], color = 'r', s = 20, label='Mean value')
            plt.plot([varMin, varMax],[minMean, maxMean], color = 'k', label = 'Interpolation')
            plt.legend(loc='upper center')
            plt.grid(True)
            plt.xlabel(varDesc)
            plt.ylabel(metName)
            sInfo = SaveInformation(
                fileName = f'{varFigDir}/{"".join(ch for ch in metName if ch.isalnum())}', savePDF=True, savePNG=True)
            saveFigures(sInfo)
            plt.close()


    # Calculating the main effect coefficients
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
            elif x[idx] > 0 and not posLabel:
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


def main(run_exp:bool = True, run_eval:bool=True, run_analysis:bool = True):
    # Simulation phase:
    repoLoc = 'C:/Data/testSample'
    samplesLoc = repoLoc + '/data'
    figLoc = repoLoc + '/figures'
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    modelLoc = repositories.cefLoc + simConfig.modelLocation
    variablesFile = './assets/yamlFiles/variables_ac_pgm.yaml'
    descFile = './assets/yamlFiles/varDescription.yaml'
    experFile = './assets/experiments/FFD_new_Framework.txt'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR,
                span = 0.65) 
    for v in variables: 
        print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
    if run_exp:
        from ActiveLearning.simInterface import runExperiment
        timeIndepVars = getTimeIndepVars(variables, omitZero=True) 
        exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)
        saveSampleToTxtFile(samples = exper, fileName = './assets/experiments/FFD_sample.txt')
        for idx, ex in enumerate(exper): 
            print(f'{idx+1}: ', ex)
        runExperiment(modelLoc= modelLoc, 
                        variables = timeIndepVars, 
                        simRepo = samplesLoc,
                        experiment=exper,
                        experFile=experFile,
                        descFile= descFile)

    # Evaluation of the samples
    if run_eval:
        sampleGroup = list(range(1,17))
        batchSize = 4
        runBatch(dataLocation=samplesLoc,
                        sampleGroup=sampleGroup,
                        configFile=simConfig,
                        figureFolder=figLoc,
                        PN_suggest=batchSize)

    # Analysis of the results:
    if run_analysis:
        metNames = simConfig.metricNames
        analyseFactorScreening(repoLoc=repoLoc, 
                            figFolder=figLoc,
                            metNames = metNames, 
                            include_bias=False)
   




# Since we are using multiprocessing we need to have this here: 
if __name__=="__main__":
    freeze_support()
    repoLoc = 'C:/Data/testSample'
    dataLoc = repoLoc + '/data'
    varDescFile = glob.glob(dataLoc + '/*.yaml')[0]
    varNames, descs = loadVars(varDescFile)
    print(varNames)
    data, labels = readDataset(repoLoc = dataLoc, dimNames = varNames)
    print(labels)
    print(data)
    # main(run_eval=True, run_exp=False, run_analysis = True)
    