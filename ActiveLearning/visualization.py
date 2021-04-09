from yamlParseObjects.yamlObjects import *
import csv
import matplotlib.pyplot as plt 
from ActiveLearning.Sampling import Space
from typing import List
import numpy as np 
from collections import namedtuple
from skimage import measure
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from ActiveLearning.benchmarks import Benchmark 
import os

class SaveInformation():
    def __init__(self, fileName, savePDF = False, savePNG = False):
        self.fileName = fileName
        self.savePDF = savePDF
        self.savePNG = savePNG
    
    def __str__(self):
        descriptor = f'''File Name: {self.fileName}
            Saves PDF: {'Yes' if self.savePDF else 'No'}
            Saves PNG: {'Yes' if self.savePNG else 'No'}
        '''
        return descriptor.__str__()

def saveFigurePickle(saveInfo: SaveInformation):
    pass

def plotSpace(space: Space, 
              figsize=(6,6),
              meshRes = 100,
              classifier = None, 
              forth_dimension:str = None, 
              fDim_values: List[int] = None,
              benchmark:Benchmark = None,
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
        _plotSpace2D(space = space, 
                    figsize = figsize, 
                    classifier = classifier,
                    meshRes = meshRes, 
                    legend = legend,
                    benchmark = benchmark,
                    newPoint = newPoint,
                    saveInfo = saveInfo,
                    showPlot = showPlot)
    elif space.dNum == 3: 
        _plotSpace3D(space = space,
                    showPlot= showPlot,
                    classifier = classifier,
                    figsize = figsize,
                    benchmark = benchmark,
                    meshRes = meshRes,
                    newPoint = newPoint,
                    saveInfo=saveInfo)
    return 

def _plotSpace3D(space: Space, 
                showPlot = True,
                classifier = None, 
                figsize = (6,6),
                legend = True, 
                benchmark = None,
                meshRes = 100,
                saveInfo:SaveInformation = None,
                newPoint = None):
    
    clf = classifier if classifier is not None else space.clf
    labels = space.eval_labels
    ranges = space.getAllDimensionBounds()
    x1range = ranges[0]
    x2range = ranges[1]
    x3range = ranges[2]
    fig = plt.figure(figsize = figsize)
    ax = plt.gca(projection = '3d')
    dimensionNames = space.getAllDimensionDescriptions()

    # Scattering the sample points colored based on their labels: 
    if len(labels) > 0: 
        points = space.samples[:len(labels),:]
        ax.scatter(points[labels==0,0], points[labels==0,1], points[labels==0,2], s=10, c = 'r', label= '- Data points')
        ax.scatter(points[labels==1,0], points[labels==1,1], points[labels==1,2], s=10, c = 'b', label= '+ Data points')
    else:
        points = space.samples
        ax.scatter(points[:,0], points[:,1], s = 10, c = 'black', label = 'samples')
    # Setting the limits of the dimensions:
    ax.set_xlim(x1range)
    ax.set_ylim(x2range)
    ax.set_zlim(x3range)
    # Labeling the dimensions of the plot:
    ax.set_xlabel(dimensionNames[0])
    ax.set_ylabel(dimensionNames[1])   
    ax.set_zlabel(dimensionNames[2])   
    # Creating the mesh for the benchmark evaluation:
    if benchmark is not None or clf is not None:
        xx = np.linspace(start = x1range[0],stop = x1range[1], num = meshRes)
        yy = np.linspace(start = x2range[0],stop = x2range[1], num = meshRes)
        zz = np.linspace(start = x3range[0],stop = x3range[1], num = meshRes)
        XX,YY,ZZ = np.meshgrid(xx,yy,zz, indexing = 'ij')
        XYZ = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    r1= x1range[1] - x1range[0]
    r2= x2range[1] - x2range[0]
    r3= x3range[1] - x3range[0]
    
    # Evaluating the benchmark and adding it to the plot:
    if benchmark is not None:
        scores = space.benchmark.getScoreVec(XYZ).reshape(XX.shape)
        out = measure.marching_cubes(scores,level = benchmark.threshold)
        verts = out[0]
        faces = out[1]
        verts = verts * [r1,r2,r3] / meshRes
        verts = verts + [x1range[0], x2range[0],x3range[0]]  
        mesh = Poly3DCollection(verts[faces], facecolor = 'green', edgecolor = 'blue', alpha = 0.5)
        ax.add_collection3d(mesh)

    # Evaluating the decision function and adding it to the plot: 
    t = clf.decision_function(XYZ).reshape(XX.shape)
    out = measure.marching_cubes(t,level = 0)
    verts = out[0]
    faces = out[1]
    verts = verts * [r1,r2,r3] / meshRes
    verts = verts + [x1range[0], x2range[0],x3range[0]]  
    mesh = Poly3DCollection(verts[faces], facecolor = 'orange', edgecolor = 'gray', alpha = 0.5)
    ax.add_collection3d(mesh)
    if legend:
        plt.legend(loc = 'upper left',bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()

    if newPoint is not None:
        newPoint = np.array(newPoint)
        newPoint = newPoint.reshape(1,len(newPoint)) if len(newPoint.shape) <2 else newPoint
        legendLabel = 'Next point' + ('s' if newPoint.shape[0]>1 else '')
        ax.scatter(newPoint[:,0], newPoint[:,1], newPoint[:,2], marker = 's',s = 20, label = legendLabel, color = 'green' )
    if saveInfo is not None:
        saveFigures(saveInfo=saveInfo)
    if showPlot:
        plt.show()
        return 
    plt.close()
    return 

def _plotSpace2D(space: Space, 
                classifier, 
                figsize = None, 
                meshRes=100, 
                benchmark = None,
                legend=True,
                newPoint = None,
                saveInfo:SaveInformation = None,
                showPlot = True):
    labels = space.eval_labels
    
    _,ax = plt.subplots(figsize = figsize)  
    ranges = space.getAllDimensionBounds()
    dimensionNames = space.getAllDimensionDescriptions()
    x1range = ranges[0]
    x2range = ranges[1]
    if len(labels) > 0: 
        points = space.samples[:len(labels),:]
        ax.scatter(points[labels==0,0], points[labels==0,1], s=12, c = 'r', label= '- Data points')
        ax.scatter(points[labels==1,0], points[labels==1,1], s=12, c = 'b', label= '+ Data points')
    else:
        points = space.samples
        ax.scatter(points[:,0], points[:,1], s = 10, c = 'black', label = 'samples')
    # TODO: This part only works with SVM classifiers because only those have support vectors
    ax.set_xlim(x1range)
    ax.set_ylim(x2range)
    ax.set_xlabel(dimensionNames[0])
    ax.set_ylabel(dimensionNames[1])
    if benchmark is not None or classifier is not None:
        xx = np.linspace(start = x1range[0], stop = x1range[1], num = meshRes)
        yy = np.linspace(start = x2range[0], stop = x2range[1], num = meshRes)
        YY,XX = np.meshgrid(yy,xx)
        # Transposed to make the dimensions match the convention. 
        xy = np.vstack([XX.ravel(),YY.ravel()]).T 
    if benchmark is not None:
        scores = benchmark.getScoreVec(xy).reshape(XX.shape)
        cs = ax.contour(XX,YY,scores, colors='g', levels = [benchmark.threshold], 
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
        for i in range(len(csLabels2)):
            cs2.collections[i].set_label(csLabels2[i])                 
    
    # If the new point(s) is (are) passed to the function it will be shown differently than the evaluated points:
    if newPoint is not None:
        newPoint = np.array(newPoint)
        newPoint = newPoint.reshape(1,len(newPoint)) if len(newPoint.shape) <2 else newPoint
        legendLabel = 'Next point' + ('s' if newPoint.shape[0]>1 else '')
        ax.scatter(newPoint[:,0], newPoint[:,1], marker = 's',s = 20, label = legendLabel, color = 'm' )
    if legend:
        plt.legend(loc = 'upper left',bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    if saveInfo is not None:
        saveFigures(saveInfo=saveInfo)
    if showPlot:
        plt.show()
    return 


def saveFigures(saveInfo):
    if saveInfo.savePDF:
        plt.savefig(fname = f'{saveInfo.fileName}.pdf',
                    facecolor='w', edgecolor = 'w', transparent = False, bbox_inches='tight')
    if saveInfo.savePNG:
        plt.savefig(fname = f'{saveInfo.fileName}.png',
                    facecolor='w', edgecolor = 'w', transparent = False, bbox_inches='tight')

def setFigureFolder(outputReportsLoc):
    figFolder = f'{outputReportsLoc}/Figures'
    if not os.path.isdir(outputReportsLoc):
        os.mkdir(outputReportsLoc)
        os.mkdir(figFolder)
    if not os.path.isdir(figFolder):
        os.mkdir(figFolder)
    return figFolder

