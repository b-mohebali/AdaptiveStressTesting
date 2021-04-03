import numpy as np 
import os, sys, platform, shutil
import yaml
import string
from typing import List
from yamlParseObjects.yamlObjects import *


resultFileName = 'finalReport.yaml'

def readDataset(repoLoc, variables:List[variableConfig]):
    """
        Reading a dataset from a group of evaluated samples.
        Input: 
            - Location of the samples
            - List of the variables configurations for the order of their names

        Output:
            - Dataset in the form of data points as a list of lists
            - Label of each sample in the same order as the samples in the dataset


        Note: Assumes that the samples are in folders with numerical names. Ignores
        all the other types of names (alphabetical or symbolic characters)

    """
    sampleFolders = [name for name in os.listdir(repoLoc) if name.isdigit()]
    labels = []
    dataset = []
    for sampleFolder in sampleFolders: 
        d,l = readSingleSample(repoLoc, variables, sampleFolder)
        labels.append(l)
        dataset.append(d)
    return dataset, labels


def readSingleSample(repoLoc,variables: List[variableConfig], sampleNumber):
    """
        This function read the results of a single sample from the repo.

        Inputs:
            - Location of the samples (repoLoc)
            - List of varaibles configuration (variables)
            - Sample Number (sampleNumber)

        Output:
            - list of the values for the dimensions of the sample (varList)
            - Label of the sample as it was evaluated by the metrics (label)

    """
    yamlFile = repoLoc + f'/{sampleNumber}/{resultFileName}'
    reportObj = FinalReport(yamlFile)
    varList = []
    for var in variables:
        varList.append(reportObj.variables[var.name])
    return varList, reportObj.label

