from ActiveLearning.Sampling import SampleSpace
import pickle

from matplotlib import scale
from yamlParseObjects.yamlObjects import Scale, getAllVariableConfigs, simulationConfig 
import matplotlib.pyplot as plt 



# metricsFiles = [
# './assets/outputReports/53/data/metrics.pickle',
# './assets/outputReports/59/data/metrics.pickle',
# './assets/outputReports/60/data/metrics.pickle',
# './assets/outputReports/61/data/metrics.pickle',
# # './assets/outputReports/63/data/metrics.pickle', # This batch did not find the disjointed region and exploration failed to find it later. 
# './assets/outputReports/64/data/metrics.pickle',

# ]

# labels = [
#     'batch size = 1',
#     'batch size = 10, dynamic RA active', 
#     'batch size = 5, dynamic RA active',        
#     'batch size = 5, dynamic RA inactive',        
#     # 'batch size = 20, dynamic RA inactive',    # 63    
#     'batch size = 20, dynamic RA inactive',        
# ]

# plt.figure()

# for idx, metricsFile in enumerate(metricsFiles): 
#     with open(metricsFile, 'rb') as pickleIn:
#         metricsReport = pickle.load(pickleIn)
#     plt.plot(metricsReport.sampleCount, metricsReport.acc, label = labels[idx])
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy %')
# plt.grid(True)
# plt.legend()
# plt.show()

from samply.hypercube import cvt 
from ActiveLearning.optimizationHelper import GA_Voronoi_Explorer
simConfigFile = './assets/yamlFiles/adaptiveTesting.yaml'
simConfig = simulationConfig(simConfigFile)
variableFiles = './assets/yamlFiles/normal_vars.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variableFiles, scalingScheme=Scale.LINEAR)

mySpace = SampleSpace(variableList = variables)


explorer = GA_Voronoi_Explorer(space=mySpace, batchSize = 1, convergence_curve=False, progress_bar=True)


samples = cvt(count = 10, dimensionality=2)
mySpace.addSamples(samples, [1]*10)

from collections import namedtuple 

FoundPoint = namedtuple('FoundPoint', ('point', 'batchNumber'))


# plt.figure(figsize = (10,10))
# plt.text(samples[0,0],samples[0,1],1)
for idx in range(2,21):
    newSamples = explorer.findNextPoints(pointNum = 1)
    mySpace.addSamples(newSamples, labels = [1])
    print(len(mySpace._samples))
    for idx,s in enumerate(mySpace._samples[:10]):        
        plt.text(s[0],s[1],0)
    for idx,s in enumerate(mySpace._samples[10:]):        
        plt.text(s[0],s[1],idx+1)
    plt.scatter(mySpace.samples[:,0], mySpace.samples[:,1], s=6)
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
    plt.grid(True)
    plt.show()
