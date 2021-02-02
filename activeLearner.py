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
from sklearn import svm

from plotter import *

import time
import numpy as np 


simConfig = simulationConfig('./yamlFiles/adaptiveTesting.yaml')
variablesFiles = './yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFiles, scalingScheme=Scale.LINEAR)


# Setting up the design space or the sampling space:

# mySpace = Space(variableList = variables,initialSampleCount = 30)
# # myBench = Hosaki(-1)
# myBench = Branin(12)
# x1 = [1,2,4,4,3]
# x2 = [2,4,1,3,4]
# l = myBench.getLabelVec(x1,x2)
# print(l)
# s = myBench.scoreVec(x1,x2)
# print(s)
# mySpace.generateInitialSample()
# mySpace.getBenchmarkLabels(myBench)

# print(mySpace.eval_labels)

# # plotSpace2D(mySpace, legend = True)

# clf = svm.SVC(kernel = 'rbf', C =1000)
# clf.fit(mySpace.samples, mySpace.eval_labels)

# plotSpace(mySpace,classifier = clf, legend = True)

a = np.array([[5,5,0,2],[1,1,1,4]])
myBench = DistanceFromOrigin(threshold = 5, inputDim = 4, root = False)
print(myBench.scoreVec(*a.T))