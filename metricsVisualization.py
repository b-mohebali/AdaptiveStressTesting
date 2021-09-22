from ActiveLearning.Sampling import ConvergenceSample, SampleSpace
import pickle

from matplotlib import scale
from yamlParseObjects.yamlObjects import Scale, getAllVariableConfigs, simulationConfig 
import matplotlib.pyplot as plt 
import numpy as np 

'''
    This file contains the code that gets a set of metrics pickle files and plots the saved metrics in several plots, one plot per metric typr. 

    For example, there is one plot for Accuracy that contains the accuracy curves of all the pickles. Same goes for recall and precision.

'''

def putInOnePlot(metricsFile, movingAvgN = 5):
    with open(metricsFile, 'rb') as pickleIn:
        metricsReport = pickle.load(pickleIn)
    plt.figure()
    plt.plot(metricsReport.sampleCount, metricsReport.acc, label = 'Accuracy')
    plt.plot(metricsReport.sampleCount, metricsReport.precision, label = 'Precision')
    plt.plot(metricsReport.sampleCount, metricsReport.recall, label = 'Recall')
    plt.xlabel('Sample count')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    plt.figure()
    iterVector = np.arange(len(metricsReport.sampleCount))

    plt.plot(iterVector, metricsReport.changeMeasure, label = 'Change Measure (actual)')
    
    m = ConvergenceSample.movingAverageVec(metricsReport.changeMeasure, movingAvgN)
    plt.plot(iterVector, m, label = f'Moving average, n={movingAvgN}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Change measure')

    plt.show()




metricsFiles = [
# './assets/outputReports/89/data/metrics.pickle',
# './assets/outputReports/92/data/metrics.pickle',
# './assets/outputReports/93/data/metrics.pickle',
'./assets/outputReports/95/data/metrics.pickle',
'./assets/outputReports/97/data/metrics.pickle', # This batch did not find the disjointed region and exploration failed to find it later. 
# './assets/outputReports/64/data/metrics.pickle',

]

labels = [
    # 'sample size = 500',
    'sample size = 1200'
    # 'batch size = 10, dynamic RA active', 
    # 'batch size = 5, dynamic RA active',        
    # 'batch size = 5, dynamic RA inactive',        
    # # 'batch size = 20, dynamic RA inactive',    # 63    
    # 'batch size = 20, dynamic RA inactive',        
]


putInOnePlot(metricsFiles[0])
putInOnePlot(metricsFiles[1])

# ---------------Plotting the accuracy:
plt.figure()
for idx, metricsFile in enumerate(metricsFiles): 
    with open(metricsFile, 'rb') as pickleIn:
        metricsReport = pickle.load(pickleIn)
    plt.plot(metricsReport.sampleCount, metricsReport.acc, label = labels[idx])
plt.xlabel('Sample Count')
plt.ylabel('Accuracy %')
plt.grid(True)
plt.legend()

# --------------Plotting the Precision:
plt.figure()
for idx, metricsFile in enumerate(metricsFiles): 
    with open(metricsFile, 'rb') as pickleIn:
        metricsReport = pickle.load(pickleIn)
    plt.plot(metricsReport.sampleCount, metricsReport.precision, label = labels[idx])
plt.xlabel('Sample Count')
plt.ylabel('Precision %')
plt.grid(True)
plt.legend()



# --------------Plotting the Recall:

plt.figure()
for idx, metricsFile in enumerate(metricsFiles): 
    with open(metricsFile, 'rb') as pickleIn:
        metricsReport = pickle.load(pickleIn)
    plt.plot(metricsReport.sampleCount, metricsReport.recall, label = labels[idx])
plt.xlabel('Sample Count')
plt.ylabel('Recall %')
plt.grid(True)
plt.legend()

# --------------Plotting the change measure:

plt.figure()
for idx, metricsFile in enumerate(metricsFiles): 
    with open(metricsFile, 'rb') as pickleIn:
        metricsReport = pickle.load(pickleIn)
    plt.plot(metricsReport.sampleCount[1:], metricsReport.changeMeasure[1:], label = labels[idx])
plt.xlabel('Sample Count')
plt.ylabel('Change measure %')
plt.grid(True)
plt.legend()


# Testing the moving mean for the change measure: 
t= metricsReport.changeMeasure
n = 5 
m1 = np.zeros(shape = (len(t),), dtype= float)
for _ in range(len(t)):
    m1[_] = np.mean(t[max(0,_-n+1):_+1])

m = ConvergenceSample.movingAverageVec(t, n)

plt.figure()
iterVector = np.arange(len(metricsReport.sampleCount))
plt.plot(iterVector, m, label = f'Moving average, n={n}')
plt.plot(iterVector,t,label='Change Measure')
plt.grid(True)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Change measure %')


# plt.show()
