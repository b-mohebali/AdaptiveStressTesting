from yamlParseObjects.yamlObjects import *
import csv
import matplotlib.pyplot as plt 
from ActiveLearning.Sampling import Space
from typing import List
import numpy as np 
from collections import namedtuple

class SaveInformation():
    def __init__(self, fileName, savePDF = False, savePNG = False):
        self.fileName = fileName
        self.savePDF = savePDF
        self.savePNG = savePNG



def plotSpace(space: Space, 
              figsize=(6,6),
              meshRes = 100,
              classifier = None, 
              forth_dimension:str = None, 
              fDim_values: List[int] = None,
              legend = True,
              newPoint = None,
              saveInfo: SaveInformation = None,
              showPlot = True) -> None:
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
            but is fixed in each plot. Only for spaces with d>3.
        - fDim_values: The set of values at which the forth dimension factor 
            is fixed in each plot. Only for spaces with d>3.
    """
    if space.dNum ==2:
        plotSpace2D(space = space, 
                    figsize = figsize, 
                    classifier = classifier,
                    meshRes = meshRes, 
                    legend = legend,
                    newPoint = newPoint,
                    saveInfo = saveInfo,
                    showPlot = showPlot)
    elif space.dNum == 3: 
        plotSpace3D(space, classifier)
    return 

def plotSpace3D(space: Space, 
                showPlot = True,
                classifier = None, 
                figsize = (6,6), 
                meshRes = 100,
                newPoint = None):
    plt.figure(figsize=figsize)
    if space.labels:
        pass
    return 


def plotSpace2D(space: Space, 
                classifier = None, 
                figsize = None, 
                meshRes=100, 
                legend=True,
                newPoint = None,
                saveInfo:SaveInformation = None,
                showPlot = True):
    points = space.samples
    labels = space.eval_labels
    fig,ax = plt.subplots(figsize = figsize)  
    ranges = space.getAllDimensionBounds()
    dimensionNames = space.getAllDimensionNames()
    x1range = ranges[0]
    x2range = ranges[1]
    if len(labels) > 0: 
        ax.scatter(points[labels==0,0], points[labels==0,1], s=12, c = 'r', label= '- Data points')
        ax.scatter(points[labels==1,0], points[labels==1,1], s=12, c = 'b', label= '+ Data points')
        
    else:
        ax.scatter(points[:,0], points[:,1], s = 10, c = 'black', label = 'samples')
    # TODO: This part only works with SVM classifiers because only those have support vectors
    ax.set_xlim(x1range)
    ax.set_ylim(x2range)
    ax.set_xlabel(dimensionNames[0])
    ax.set_ylabel(dimensionNames[1])
    if space.benchmark is not None or classifier is not None:
        xx = np.linspace(start = x1range[0], stop = x1range[1], num = meshRes)
        yy = np.linspace(start = x2range[0], stop = x2range[1], num = meshRes)
        YY,XX = np.meshgrid(yy,xx)
        # Transposed to make the dimensions match the convention. 
        xy = np.vstack([XX.ravel(),YY.ravel()]).T 
    if space.benchmark is not None:
        scores = space.benchmark.getScoreVec(xy).reshape(XX.shape)
        cs = ax.contour(XX,YY,scores, colors='g', levels = [space.benchmark.threshold], 
                alpha = 1, linestyles = ['dashed']) 
        cslabels = ['Actual Boundary']
        ax.clabel(cs, inline=1, fontsize=10) 
        for i in range(len(cslabels)):
            cs.collections[i].set_label(cslabels[i])     
    """ Plotting the contours of the decision function indicating the
        decision boundary and the margin. 
    """
    if classifier is not None:
        ax.scatter(classifier.support_vectors_[:,0], classifier.support_vectors_[:,1], s=80, 
                    linewidth = 1, facecolors = 'none', edgecolors = 'orange', label='Support vectors')
        decisionFunction = classifier.decision_function(xy).reshape(XX.shape)
        cs2 = ax.contour(XX, YY, decisionFunction, colors='k', levels=[-1,0,1], alpha=1,linestyles=['dashed','solid','dotted'])
        csLabels2 = ['DF=-1','DF=0 (hypothesis)','DF=+1']
        # ax.clabel(cs2, inline=1, fontsize=10)
        for i in range(len(csLabels2)):
            cs2.collections[i].set_label(csLabels2[i])                 
    
    # If the new point(s) is (are) passed to the function it will be shown differently than the evaluated points:
    if newPoint is not None:
        newPoint = np.array(newPoint)
        newPoint = newPoint.reshape(1,len(newPoint)) if len(newPoint.shape) <2 else newPoint
        legendLabel = 'Next point' + ('s' if newPoint.shape[0]>1 else '')
        print('label: ' + legendLabel)
        ax.scatter(newPoint[:,0], newPoint[:,1], marker = 's',s = 20, label = legendLabel, color = 'm' )
    if legend:
        # ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop={'size':12})
        plt.legend(loc = 'upper left',bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    if saveInfo is not None:
        if saveInfo.savePDF:
            plt.savefig(fname = f'{saveInfo.fileName}.pdf',
                        facecolor='w', edgecolor = 'w', transparent = False, bbox_inches='tight')
        if saveInfo.savePNG:
            plt.savefig(fname = f'{saveInfo.fileName}.png',
                        facecolor='w', edgecolor = 'w', transparent = False, bbox_inches='tight')
    if showPlot:
        plt.show()
    return 