from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import Branin
from ActiveLearning.Sampling import *
import platform
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum


from plotter import *

import time
import numpy as np 


simConfig = simulationConfig('./yamlFiles/adaptiveTesting.yaml')
variablesFiles = './yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFiles, scalingScheme=Scale.LINEAR)


# Setting up the design space or the sampling space:
mySpace = Space(variableList = variables,initialSampleCount = 20)

myBench = Branin(8)

x1 = [-2,4,6]
x2 = [10,0, 10]
l = myBench.getLabelVec(x1,x2)
print(l)
mySpace.generateInitialSample()
print(mySpace.samples)
