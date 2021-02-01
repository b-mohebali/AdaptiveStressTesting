from yamlParseObjects.yamlObjects import * 
from typing import List
from samply.hypercube import cvt

class Dimension():
    def __init__(self, varConfig: variableConfig):
        self.name = varConfig.name
        self.bounds = [varConfig.lowerLimit, varConfig.upperLimit]
        self.range = self.bounds[1] - self.bounds[0]


class Space():
    def __init__(self, variableList: List[variableConfig], initialSampleCount = 20):
        self.initialSampleCount = initialSampleCount
        self.dimensions = []
        for varConfig in variableList:
            self.dimensions.append(Dimension(varConfig = varConfig))
        self.dNum = len(self.dimensions)
        self.samples = []
        self.labels = []
    
    def getAllDimensionNames(self):
        return [dim.name for dim in self.dimensions]
    
    def getAllDimensionRanges(self):
        return [dim.bounds for dim in self.dimensions]


    def generateInitialSample(self):
        # First sample. All the dimensions are between 0 and 1
        samples = cvt(count = self.initialSampleCount, dimensionality=self.dNum)
        # all the samples are then scaled to their appropriate range:
        for dimIndex, dimension in enumerate(self.dimensions):
            samples[:,dimIndex] *= dimension.range # Samples times the range of the dimension
            samples[:,dimIndex] += dimension.bounds[0] # Samples shifted by the lower bound.
        self.samples = samples
        return
