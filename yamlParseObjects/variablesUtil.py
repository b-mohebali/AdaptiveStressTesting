from .yamlObjects import *
from typing import List
import numpy as np 

import yaml
import os
import shutil
import matplotlib.pyplot as plt
import subprocess
from scipy.linalg import hadamard
import ast
import random



def trunkatedExponential(u, a,b):
    beta = np.e/(np.e-1.)
    xHat = -1 * np.log(1 - u /beta)
    x = a + xHat * (b-a)
    return x

trunkatedExponentialVec = np.vectorize(trunkatedExponential)

def getTimeDepVariables(variables):
    vars = []
    for var in [v for v in variables if v.varType.lower() != 'timeindep']: 
        vars.append(var)
    return vars

def getTimeindepVariablesDict(variables):
    varMap = {}
    for var in [v for v in variables if v.varType.lower() == 'timeindep']: 
        varMap[var.name] = var
    return varMap    

def getVariablesDict(variables):
    varMap = {}
    for var in [v for v in variables if v.varType.lower() != 'timeindep']: 
        varMap[var.name] = var
    return varMap    

def getTimeIndepVarsDict(variables, omitZero = True):
    varMap = {}
    for var in getTimeIndepVars(variables, omitZero=omitZero):
        varMap[var.name] = var
    return varMap


# this function returns a list of all the variables in the yaml file. It can shuffle the list or omit the ones 
# that have a initial value of zero. 
def getTimeIndepVars(variables, shuffle = False, omitZero = True):
    varList = [v for v in variables if v.varType.lower() == 'timeindep']
    if omitZero:
        varList = [v for v in varList if v.initialState != 0.0]
    if shuffle:
        random.shuffle(varList)
    return varList

# This function gets a list of variables config and returns a random value for each one within
# its specified range of available values. 
def randomizeVariables(variables):
    randomValues = {}
    values = np.random.rand(len(variables))
    for idx,key in enumerate(variables):
        var = variables[key]
        range = var.upperLimit - var.lowerLimit
        randomValues[var.name] = values[idx] * range + var.lowerLimit
    return randomValues

# This function creates a list of random samples for the given variables. 
# The creation of all the random samples for the factors can give the user 
# the control to change the sampling method more conveniently.
# Also this function uses stratified sampling technique to give a more uniform 
# distribution.
# An addition to this function will make it able to sample from a trunkated 
# exponential distribution instead of a uniform one. This is needed for the 
# time when the limits of the variation range are defined as logarithmic instead 
# of linear. 
def randomizeVariablesList(variables, sampleNum, subIntervals,scalingScheme = Scale.LINEAR, saveHists=False):
    randomLists = {}
    # Making the random lists with stratified 
    for var in variables:
        # var = variables[key]
        randomValues = np.random.rand(sampleNum)
        if scalingScheme == Scale.LINEAR:
            interval = var.upperLimit - var.lowerLimit
            d = interval / subIntervals
            randomVar = [rv*d + var.lowerLimit+(idx%subIntervals)*d for idx,rv in enumerate(randomValues)] 
        elif scalingScheme == Scale.LOGARITHMIC:
            d = 1. / subIntervals
            stratified = [rv*d + d*(idx%subIntervals) for idx, rv in enumerate(randomValues)] 
            randomVar = trunkatedExponentialVec(stratified, var.lowerLimit, var.upperLimit)
        # Shuffling is needed for disruption of correlation between the factors.
        np.random.shuffle(randomVar)
        randomLists[var.name] = randomVar
        # Saving the plots of the distributions. 
        if saveHists:
            plt.figure()
            plt.hist(x=randomVar, bins=subIntervals, range=(min(var.lowerLimit, var.upperLimit),max(var.lowerLimit, var.upperLimit)), rwidth=0.95)
            plt.title(var.name)
            plt.xlabel('Bins')
            plt.ylabel('Number of samples in the bin')
            plt.savefig(f'./figures/histo_{var.name}.png')
            plt.close()
            plt.figure()
            plt.scatter(range(1,1 + sampleNum),randomVar,marker='.',s=2)
            plt.title(f'Scatter plot of {var.name} distribution')
            plt.savefig(f'./figures/scatter_{var.name}.png')
            plt.xlabel('Number of sample')
            plt.ylabel('Sample value')
            plt.close()
    randValuesList = []
    for counter in range(sampleNum):
        randomSample = {}
        for var in variables:
            # var = variables[key]
            randomSample[var.name] = randomLists[var.name][counter]
        randValuesList.append(randomSample)
    return randValuesList

# This function creates the samples for STRICT One-At-A-Time sensitivity analysis
# The simulation will run the model for the factors at the extremes of their 
# range. In the sample set, each factor has k simulation when it is at its 
# minimum and k simulation when it is at its maximum. 
def strictOATSampleGenerator(varDict, addMiddle = False):
    randValuesList = []
    currentSample = {}
    middleSample = {}
    for key in varDict:
        var = varDict[key]
        varLow = max(var.initialState * (1-variableSpan), var.lowerLimit)
        varHigh = min(var.initialState * (1+variableSpan), var.upperLimit)
        currentSample[var.name] = varLow
        middleSample[var.name] = varLow + (varHigh - varLow)/2.
    for key in varDict:
        var = varDict[key]
        varHigh = min(var.initialState * (1+variableSpan), var.upperLimit)
        currentSample[var.name] = varHigh
        randValuesList.append(currentSample.copy())
    for key in varDict:
        var = varDict[key]
        varLow = max(var.initialState * (1-variableSpan), var.lowerLimit)
        currentSample[var.name] = varLow
        randValuesList.append(currentSample.copy())
    # Adding the middle sample:
    if addMiddle:
        randValuesList.append(middleSample.copy())
    return randValuesList

def standardOATSampleGenerator(varDict, repeat = False):
    copyNumber = 2 if repeat else 1
    valuesList = []
    standardSample = {}
    for key in varDict:
        var = varDict[key]
        standardSample[var.name]= var.initialState
    for key in varDict:
        var = varDict[key]
        varLow = max(var.initialState * (1-variableSpan), var.lowerLimit)
        varHigh = min(var.initialState * (1+variableSpan), var.upperLimit)
        for i in range(copyNumber):
            nextSample = standardSample.copy()
            nextSample[var.name] = varHigh
            valuesList.append(nextSample.copy())
            nextSample = standardSample.copy()
            nextSample[var.name] = varLow
            valuesList.append(nextSample.copy())
    return valuesList


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


# This function takes a LIST of the time independent variable. 
# The output of the function is an experiment in the form of a series of 
# simulation parameters that are generated using the Hadamard matrices. 
# To know more about the details please refer to Saltelli's Global Sensitivity
# Analysis, a primer book, chapter 2.  
def fractionalFactorialExperiment(timeIndepVariables,res4=False):
    h = getFFHMatrix(timeIndepVariables, res4 = res4)
    valueList = []
    simNumber = h.shape[0]
    for simIndex in range(simNumber):
        currentSample = {}
        for varIndex,var in enumerate(timeIndepVariables):
            currentSample[var.name] = h[simIndex, varIndex]
        valueList.append(currentSample.copy())
    return valueList

def getFFHMatrix(vars, res4 = False, dtype = float):
    k = len(vars)
    h = hadamard(next_power_of_2(k+1),dtype = float)
    if res4:
        h = np.concatenate((h,-h),axis = 0)
    for idx,var in enumerate(vars):
        varLow = max(var.initialState * (1-variableSpan), var.lowerLimit)
        varHigh = min(var.initialState * (1+variableSpan), var.upperLimit)
        h[h[:,idx+1]==1,idx+1] = varHigh
        h[h[:,idx+1]==-1,idx+1] = varLow
    return h[:,1:k+1] # Only returning the part of the H matrix that is used as simulation parameters.


def getVariablesInitialValueDict(variables):
    initials = {}
    for var in variables:
        initials[var.name] = var.initialState
    return initials

# This function gets a dictionary of variables and their values and 
# saves them in a yaml file.
def saveVariableValues(varDict, fileName):
    print("This is the variables map: " , varDict)
    v={}
    for key in varDict:
        v[key] = float(varDict[key]) 
    with open(fileName, 'w') as outFile:
        yaml.dump(v, outFile, default_flow_style=False)
    return

def saveVariableDescription(variables, fileName):
    v = {}
    for var in variables:
        v[var.name] = var.description
    with open(fileName, 'w') as outFile:
        yaml.dump(v, outFile, default_flow_style=False)
    return

def createNewDatafolder(parent):
    allFolders = os.listdir(parent)
    s = [int(f) for f in allFolders if f.isdigit()]
    nextOne = 1 if len(s)==0 else max(s) + 1 
    newFolderPath = f'{parent.rstrip("/")}/{nextOne}/'
    os.mkdir(newFolderPath)
    return newFolderPath

def createSpecificDataFolder(parent, folderNum):
    newFolderPath = f'{parent.rstrip("/")}/{folderNum}/'
    shutil.rmtree(newFolderPath, ignore_errors=True)
    os.mkdir(newFolderPath)
    return newFolderPath


def copyDataToNewLocation(newLocation, dataFolder):
    allFiles = [f for f in os.listdir(dataFolder) if not os.path.isdir(f'{dataFolder.rstrip("/")}/{f}') and f.endswith('.mat')]
    for f in allFiles:
        filePath = f'{dataFolder.rstrip("/")}/{f}'
        newPath = f'{newLocation.rstrip("/")}/{f}'
        shutil.copyfile(filePath, newPath)
    return

def copyDataToremoteServer(dataRepo, dataAddress, isFolder = True):
    cmd = f'scp {"-r" if isFolder else ""} {dataAddress} {dataRepo}' # Adds -r if the data is a folder
    print(cmd)
    os.system(cmd)
    return   




# This function will remove the old data folders (that are already copied to the remote 
# server) so that the number of data folders in the source repo stays 
# at maxSize specified in the function signature. 
def removeExtraFolders(dataFolder, maxSize):
    s = [int(f) for f in os.listdir(dataFolder) if f.isdigit()]
    while len(s) > maxSize:
        toRemove = f'{dataFolder.rstrip("/")}/{min(s)}'
        print(f'Removing {toRemove}')
        shutil.rmtree(toRemove)
        s = [int(f) for f in os.listdir(dataFolder) if f.isdigit()]
    return 

def emptyFolder(folderName):
    s = [int(f) for f in os.listdir(folderName) if f.isdigit()]
    for f in s:
        toRemove = f'{folderName.rstrip("/")}/{f}'
        print(f'Removing {toRemove}')
        shutil.rmtree(toRemove)
    return 

    
def copyMetricScript(scriptName, location, newFolder):
    filePath = f'{location.rstrip("/")}/{scriptName}'
    newFilePath = f'{newFolder.rstrip("/")}/{scriptName}'
    
    shutil.copyfile(filePath, newFilePath)
    return


def saveSampleToTxtFile(samples, fileName):
    if not os.path.isfile(fileName):
        pass
    with open(fileName,'w') as f:
        for sample in samples:
            f.write(sample.__str__() + '\n')

def loadSampleFromTxtFile(fileName):
    output = []
    with open(fileName, 'r') as sampleFile:
        output = [ast.literal_eval(l) for l in sampleFile]
    return output 

# This function will generate the samples needed for verification of the 
# samples (FFD and OAT). The idea is to check the variation of the output
# around the initial point with logarithmic scales. 
def generateVerifSample(variables):
    varInitial = getVariablesInitialValueDict(variables)
    varDict = getTimeindepVariablesDict(variables)
    sampleList = []
    sampleList.append(varInitial.copy())
    scales = [0.2,0.5,2,5]
    lastSample = None
    for key in varInitial:
        var = varDict[key]
        for scale in scales:
            currentSample = varInitial.copy()
            currentSample[key] = currentSample[key] * scale
            if currentSample[key] > var.upperLimit: 
                currentSample[key] = var.upperLimit
            if currentSample[key] < var.lowerLimit:
                currentSample[key] = var.lowerLimit
            if currentSample != lastSample:
                sampleList.append(currentSample.copy())
                lastSample = currentSample.copy()
    return sampleList

def findDifferentValue(baseLine, sample):
    for key in baseLine:
        if baseLine[key] != sample[key]:
            return key
    return None

