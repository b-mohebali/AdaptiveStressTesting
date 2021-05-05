# #! /usr/bin/python

from yamlParseObjects.yamlObjects import * 
import os,sys
import string
import time
import yaml 
matlabPath = './Matlab'
matlabConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')

import matlab.engine
from metricsRunTest import * 
from multiprocessing import Process
import pickle
def f(x): 
    return x*x

def print_name(name, number):
    print('hello ', name, number)

def info(title):
    print(title)
    print('module name', __name__)
    print('parent process:', os.getppid())
    print('process ID:', os.getpid())

def theLoop():
    pid = os.getpid()
    print('starting from proc', pid)
    for _ in range(100000000):
        pass
    print('Finished',pid)

class TestObject:
    def __init__(self, dataLoc, sampleNum, metricNames, figFolder):
        self.dl = dataLoc
        self.sampleNum = sampleNum
        self.metricNames = metricNames
        self.figFolder = figFolder
    
    def runMetrics(self):
        # Each of these objects has to start its own MATLAB engine.
        # The time it takes to start the engine was measured to evaluate the trade-off.
        start = time.perf_counter()
        self.engine = setUpMatlab(simConfig=matlabConfig)
        finish = time.perf_counter()
        print(f'MATLAB engine for process {os.getpid()} started in {round(finish - start, 2)} seconds')
        # The call to the MATLAB metrics evaluation after the engine started.
        getMetricsResults(self.dl, self.engine, self.sampleNum, self.metricNames, self.figFolder)
        # The MATLAB engine is shut down after the evaluation is done.
        self.engine.quit()

    def doSomething(self, seconds):
        print(f'Sleeping for {seconds} second')
        time.sleep(seconds)
        print(f'This is process {os.getpid()}')
        print(f'Done sleeping for {seconds} seconds...')

    

def doSomething(seconds):
    print(f'Sleeping for {seconds} second')
    time.sleep(seconds)
    print(f'This is process {os.getpid()}')
    print(f'Done sleeping for {seconds} seconds...')

if __name__=='__main__':
    dataLocation = 'E:/Data/motherSample2'
    figFolder = dataLocation + '/figures'
    sGroup = list(range(20,1800))
    runMetricsBatch(dataLocation=dataLocation, 
                sampleGroup = sGroup, 
                configFile = matlabConfig,
                figureFolder = figFolder,
                processNumber = 4)

