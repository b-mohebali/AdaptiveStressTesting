from yamlParseObjects.yamlObjects import *
import csv
import matplotlib.pyplot as plt 
from ActiveLearning.Sampling import Space
from typing import List

def plotSpace(space: Space, 
              classifier = None, 
              forth_dimension:str = None, 
              fDim_values: List[int] = None) -> None:
    """This function plots the samples in a Space object.
    - Options: 3D and 2D spaces.
    - TO-DO: Higher dimensions implementation by generating a set 
        of plots  in which the 4th dimension is fixed at a point. 
        The set of the fixed values are passed to this function. 
    ****
    Inputs:
        - space: the space object containing the samples and their labels
        - classifier: The trained classifier for the decision boundary 
            plotting
        - forth_dimension: The dimension that is not shown in the plots 
            but is fixed in each plot.
        - fDim_values: The set of values at which the forth dimension factor 
            is fixed in each plot. 
    """
    if space.dNum ==2:
        plotSpace2D(space, classifier)
    elif space.dNum == 3: 
        plotSpace3D(space, classifier)
    
    
    return 

def plotSpace3D(space: Space, classifier = None):

    return 


def plotSpace2D(space: Space, classifier = None):
    return 