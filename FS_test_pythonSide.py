#! /usr/bin/python3

from ActiveLearning.dataHandling import getNotEvaluatedSamples, loadMetricValues, loadVariableValues, reconstructDesignMatrix
from ActiveLearning.dataHandling import readDataset
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
from enum import Enum 
from scipy.interpolate import interp1d

def printName(s):
    printName = [c for c in s if c.isalnum()]
    return ''.join(printName)

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
                fileName = f'{varFigDir}/{metName}', savePDF=True, savePNG=True)
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
        sInfo.fileName = f'{figFolder}/finalResults/{metricName}'
        saveFigures(sInfo)
        plt.close()
    return 

def InterpMetrics(varValues, metricValues, n):
    newVals = np.linspace(min(varValues),max(varValues), num=n, endpoint = True)
    newMet = interp1d(varValues, metricValues, kind = 'cubic')
    newMetVals = newMet(newVals)
    return newVals, newMetVals

def analyseVariableSweep(repoLoc, figFolder, metricNames):
    """
        This function analyzes the data from the variable sweep experiment that uses the verification sample generation procedure. The output is a set of plots, one for each metric value, vs the variable that is used for the sweep

        Inputs:
            - repoLoc: Location of the repository that includes the data folder, that includes the sample folders.
            - figFolder: The location of the figure folder that is going to store the results of the analysis and the plots from evaluation of the samples. 
            - metNames: The list of the metric names coming out of the evaluation. It is used both for loading the sweeping data and the plotting of the results.
        
        Outputs: 
            - NOTE: The output is a set of plots for the metrics. No value is returned from this function. 
    """
    dataLoc = repoLoc + '/data'
    varDescFile = glob.glob(dataLoc + '/*.yaml')[0]
    varNames, descs = loadVars(varDescFile)
    if len(varNames) > 1 or len(descs) > 1:
        raise ValueError('More than one variable was used in the sweeping experiment.')
    varName = varNames[0]
    desc = descs[varName]
    metVals = loadMetricValues(dataLoc = dataLoc, metricNames= metricNames)
    varValues = loadVariableValues(dataLoc=dataLoc, varNames = varNames)[varName]
    # Grabbing the folder where the final results are going to be saved.
    resultsFolder = f'{figFolder}/finalResults'
    if not os.path.isdir(resultsFolder):
        os.mkdir(resultsFolder)
    
    _, labels = readDataset(dataLoc, dimNames = [varName])
    # Analysis based on the metric names:
    for metName in metricNames:
        met = metVals[metName]
        plt.figure(figsize = (10,5))
        # Adding the smooth interpolation of the points:
        x_inter, y_inter = InterpMetrics(varValues, met,n=200)
        plt.semilogx(x_inter, y_inter, color = 'k', label = 'Interpolation')
        plt.scatter(varValues[labels==1], met[labels==1], label='Infeasible samples', color = 'r', s = 20)
        plt.scatter(varValues[labels==0], met[labels==0], label='Feasible samples', color = 'b', s = 20)
        plt.legend()
        # Touch ups and saving the results:
        plt.grid(b=True, which='both')
        plt.xlabel(desc)
        plt.ylabel(metName)
        sInfo = SaveInformation(
            fileName= f'{resultsFolder}/{metName}', savePDF=True, savePNG=True)
        saveFigures(sInfo)
        plt.close()
    return 

class ExperimentType(Enum):
    STRICT_OAT = 0
    STANDARD_OAT = 1
    FFD = 2
    VERIFICATION = 3
    SWEEP = 4

def main(run_exp:bool = True, 
        run_eval:bool=True, 
        run_analysis:bool = True,
        experType: ExperimentType = ExperimentType.FFD,
        include_bias = False):
    # Simulation phase:
    # repoLoc = 'C:/Data/testSample'
    repoLoc = testRepo10
    samplesLoc = repoLoc + '/data'
    print(samplesLoc)
    # Creating the data folder if it does not exist:
    if not os.path.isdir(samplesLoc):
        os.mkdir(samplesLoc)
    figLoc = repoLoc + '/figures'
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
    modelLoc = repositories.cefLoc + simConfig.modelLocation

    variablesFile = './assets/yamlFiles/test.yaml'
    # variablesFile = './assets/yamlFiles/variables_ac_pgm.yaml'
    
    descFile = './assets/yamlFiles/varDescription.yaml'
    # experFile = './assets/experiments/FFD_new_Framework.txt'
    experFile = './assets/experiments/Frequency_Sweep.txt'
    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR,
                span = 0.65) 
    for v in variables: 
        print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
        from ActiveLearning.simInterface import runExperiment
    timeIndepVars = getTimeIndepVars(variables, omitZero=True) 
    # Doing the experiment based on what was asked for. 
    if experType == ExperimentType.FFD:
        exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)
    elif experType == ExperimentType.VERIFICATION:
        exper = generateVerifSample(variables=timeIndepVars, initValue = False)
    elif experType == ExperimentType.STRICT_OAT:
        exper = strictOATSampleGenerator(variables = timeIndepVars)
    elif experType == ExperimentType.STANDARD_OAT:
        exper = standardOATSampleGenerator(variables = timeIndepVars)       
    elif experType == ExperimentType.SWEEP:
        # Sweep only done for the first variable in the config
        varName = 'QXUfro'
        exper = generateSweepSample(variables,varName = varName, num = 50, scaleType=Scale.LOGARITHMIC)

    saveSampleToTxtFile(samples = exper, fileName = experFile)

    if run_exp:
        runExperiment(modelLoc= modelLoc, 
                        variables = timeIndepVars, 
                        simRepo = samplesLoc,
                        experiment=exper,
                        experFile=experFile,
                        descFile= descFile)

    # Evaluation of the samples (parallelized):
    if run_eval:
        # Only loads the samples that are not already evaluated. 
        # TODO: Put an option for discarding the old evaluation and run it anew.
        sampleGroup = getNotEvaluatedSamples(dataLoc = samplesLoc)
        print(sampleGroup)
        batchSize = 4
        runBatch(dataLocation=samplesLoc,
                        sampleGroup=sampleGroup,
                        configFile=simConfig,
                        figureFolder=figLoc,
                        PN_suggest=batchSize)

    # Analysis of the results:
    if run_analysis:
        metNames = simConfig.metricNames
        if experType != ExperimentType.SWEEP:
            analyseFactorScreening(repoLoc=repoLoc, 
                            figFolder=figLoc,
                            metNames = metNames, 
                            include_bias=include_bias)
        else:
            analyseVariableSweep(repoLoc= repoLoc,
                            figFolder = figLoc, 
                            metricNames=metNames)

# Since we are using multiprocessing we need to have this here: 
if __name__=="__main__":
    freeze_support()
    experType = ExperimentType.SWEEP
    main(run_exp=True, run_eval=True, run_analysis = True,
        experType = experType, include_bias = False)
    