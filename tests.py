import repositories as repo 
import pickle 
from samply.hypercube import cvt 
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import *
from ActiveLearning.benchmarks import * 

simConfigFile = './assets/yamlFiles/adaptiveTesting.yaml'
simConfig = simulationConfig(simConfigFile)
pickleName = f'{simConfig.outputFolder}/71/testClf.pickle'
varsFile = './assets/yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=varsFile, scalingScheme=Scale.LINEAR)
n = 100000
with open(pickleName, 'rb') as pickleIn:
    testClf = pickle.load(pickleIn)

sample = halton(count = n, dimensionality=2) 
space = SampleSpace(variableList=variables)
dims = space.dimensions
for dimIndex, dimension in enumerate(dims):
    sample[:,dimIndex] *= dimension.range
    sample[:,dimIndex] += dimension.bounds[0] 

myBench = Hosaki(threshold = -1)

actualLabels = myBench.getLabelVec(sample)
hypoLabels = testClf.predict(sample) 
selected = sample[actualLabels!= hypoLabels,:]
print(len(selected))

import matplotlib.pyplot as plt 
plt.scatter(sample[:,0], sample[:,1], s=2)
plt.scatter(selected[:,0], selected[:,1], s=2,color='red')

plt.grid(True)
plt.show()
