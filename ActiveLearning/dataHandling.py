import numpy as np 
import os, sys, platform, shutil
import yaml
import string
from typing import List
from yamlParseObjects.yamlObjects import *
from repositories import * 
import pickle
import string

class BadExtention(Exception):
    pass


resultFileName = 'finalReport.yaml'

def readDataset(repoLoc, dimNames, includeTimes = False, sampleRange = None):
    """
        Reading a dataset from a group of evaluated samples.
        
        Input: 
            - Location of the samples
            - List of the variables configurations for the order of their names
            - IncludeTimes: Boolean determining whether the process time of each is needed and should be included in the output or not. 
            - sampleRange: If not None, only the samples in the entered range
                will be read for dataset loading. If None, all samples in the repository will be read for dataset loading. 

        Output:
            - dataset: Dataset in the form of data points as a list of lists
            - labels: Label of each sample in the same order as the samples in the dataset
            - times: (Optional) The time each sample took to be evaluated.


        Note: Assumes that the samples are in folders with numerical names. Ignores
        all the other types of names (alphabetical or symbolic characters)

    """
    sampleFolders = [name for name in os.listdir(repoLoc) if name.isdigit()]
    # This makes sure that only the samples that actually exist in the 
    #   repo will be loaded into the dataset.
    if sampleRange is not None:
        sampleFolders  = [_ for _ in sampleFolders if _ in sampleRange]
    labels = []
    dataset = []
    elapsed_times = []
    for sampleFolder in sampleFolders: 
        d,l,t = readSingleSample(repoLoc, dimNames, sampleFolder)
        labels.append(l)
        dataset.append(d)
        elapsed_times.append(t)
    if includeTimes:
        return dataset, labels, elapsed_times
    return dataset, labels

def readSingleSample(repoLoc,dimNames, sampleNumber):
    """
        This function read the results of a single sample from the repo.

        Inputs:
            - Location of the samples (repoLoc)
            - List of the dimension names (dimNames)
            - Sample Number (sampleNumber)

        Output:
            - list of the values for the dimensions of the sample (varList)
            - Label of the sample as it was evaluated by the metrics (label)
            - The time the sample took to be evaluated.
    """
    yamlFile = repoLoc + f'/{sampleNumber}/{resultFileName}'
    reportObj = FinalReport(yamlFile)
    varList = []
    for dimName in dimNames:
        varList.append(reportObj.variables[dimName])
    return varList, reportObj.label, reportObj.elapsed_time

def getNextSampleNumber(repoLoc, createFolder:bool = False, count = 1):
    """
        Gets the location of the repository and determines what is the number of the next sample. 

        Returns a list of sample numbers if count > 1
    """
    sampleFolders = [int(name) for name in os.listdir(repoLoc) if name.isdigit()]
    lastSampleNumber = max(sampleFolders)
    nextSampleNumber = lastSampleNumber + 1
    nextSamples = [_ for _ in range(nextSampleNumber, nextSampleNumber+count)]
    if createFolder:
        [os.mkdir(repoLoc + f'/{_}') for _ in nextSamples]
    return nextSamples

"""
    Takes a classifier and a name for the saved file. Saves the classifier object as a pickle in the directory allocated to the trained classifiers.

    Inputs:
        - cls: Classifier that is going to be saved.
        - pickleName: The name of the file being used to save the classifier
"""
def saveClassifierAsPickle(cls, pickleName: str):
    # Checking if the file has the right extension:
    if not pickleName.endswith('.pickle'):
        raise BadExtention("The file name has to end with .pickle")
    # Building the absolute path of the saved file:
    fileName = picklesLoc + pickleName
    # Saving the pickle:
    with opne(fileName, 'wb'):
        pickle.dump(cls, fileName)

