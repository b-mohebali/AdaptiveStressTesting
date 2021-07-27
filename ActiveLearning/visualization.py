from yamlParseObjects.yamlObjects import *
import csv
import matplotlib.pyplot as plt 
from ActiveLearning.Sampling import SampleSpace
from typing import List
import numpy as np 
from collections import namedtuple
from skimage import measure
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ActiveLearning.benchmarks import Benchmark 
import os
from ActiveLearning.Sampling import StandardClassifier

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

def plotSpace(space: SampleSpace, 
              figsize=(6,6),
              meshRes = 100,
              classifier = None, 
              benchmark:Benchmark = None,
              legend = True,
              showGrid = False, 
              newPoints = None,
              explorePoints = None,
              saveInfo: SaveInformation = None,
              showPlot = True,
              insigDimensions = [2,3], 
              gridRes = None,
              prev_classifier =None,
              comparison_classifier = None,
              constraints = []) -> None:
    """This function plots the samples in a Space object.
    - Options: 3D and 2D spaces. (4D coming up)
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
    # Three options for the dimension of the design space are available: 
    if space.dNum ==2:
        _plotSpace2D(space = space, 
                    figsize = figsize, 
                    classifier = classifier,
                    meshRes = meshRes, 
                    legend = legend,
                    showGrid = showGrid,
                    benchmark = benchmark,
                    newPoints = newPoints,
                    saveInfo = saveInfo,
                    showPlot = showPlot,
                    explorePoints = explorePoints,
                    constraints = constraints)
    elif space.dNum == 3: 
        _plotSpace3D(space = space,
                    showPlot= showPlot,
                    classifier = classifier,
                    figsize = figsize,
                    benchmark = benchmark,
                    meshRes = meshRes,
                    legend = legend,
                    showGrid=showGrid,
                    newPoints = newPoints,
                    explorePoints = explorePoints,
                    saveInfo=saveInfo)
    elif space.dNum == 4:
        _plotSpace4D(space = space, 
                    showPlot = showPlot,
                    classifier=classifier,
                    figsize=figsize,
                    legend = legend,
                    benchmark=benchmark,
                    meshRes = meshRes,
                    showGrid = showGrid,
                    saveInfo=saveInfo,
                    insigDimensions = insigDimensions,
                    gridRes = gridRes,
                    prev_classifier = prev_classifier, 
                    comparison_classifier = comparison_classifier,
                    constraints = constraints)
    return 


"""

    NOTE: The 4D version of the visualizer does not have the feature to show the data points or the newly found points.
"""
def _plotSpace4D(space: SampleSpace,
                insigDimensions,
                showPlot = True, 
                classifier:StandardClassifier = None,
                figsize = (6,6),
                legend = True,
                showGrid = False,
                benchmark = None,
                meshRes = 100,
                gridRes = (4,4),
                saveInfo: SaveInformation = None,
                prev_classifier = None,
                comparison_classifier = None,
                constraints = []):
    ### Getting the information about the significant and insignificant dimensions. 
    
    ## Implementation of the 4D visualization: 
    allDims = list(range(4))
    sigDims = [_ for _ in allDims if _ not in insigDimensions]
    ranges = space.getAllDimensionBounds()
    dimNames = space.getAllDimensionNames()
    sigDim1Range = ranges[sigDims[0]]
    sigDim2Range = ranges[sigDims[1]]
    insigDim1Range = ranges[insigDimensions[0]]
    insigDim2Range = ranges[insigDimensions[1]]

    insigDim1Vals = np.linspace(start = insigDim1Range[0],
                                stop = insigDim1Range[1],
                                num = gridRes[0],
                                endpoint = True)
    insigDim2Vals = np.linspace(start = insigDim2Range[0],
                                stop = insigDim2Range[1],
                                num = gridRes[1],
                                endpoint = True)

    xx = np.linspace(start = sigDim1Range[0],
                stop = sigDim1Range[1],
                num = meshRes)
    yy = np.linspace(start = sigDim2Range[0],
                stop = sigDim2Range[1],
                num = meshRes)

    XX,YY = np.meshgrid(xx,yy, indexing = 'ij')
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    dataVec = np.zeros(shape = (YY.size, 4), dtype = float)

    onesVec = np.ones((XX.size,),dtype = float)
    insigVec1 = onesVec * insigDim1Vals[0]
    insigVec2 = onesVec * insigDim2Vals[0]

    dataVec[:,insigDimensions[0]] = insigVec1
    dataVec[:,insigDimensions[1]] = insigVec2
    dataVec[:,sigDims[0]] = xy[:,0]
    dataVec[:,sigDims[1]] = xy[:,1]

    plotNum = 1
    fig,ax = plt.subplots(nrows = gridRes[0], ncols = gridRes[1], figsize = figsize)
    # This variable is used to make sure that the legend for only one constraint is shown
    constraintsLabeled = False
    """
        This loop goes through the values of the "insignificant" dimensions that are discretized and 
        creates a grid of 2D plots 
    """
    
    for rowNum in range(gridRes[0]):
        for colNum in range(gridRes[1]):
            dataVec[:,insigDimensions[0]] = onesVec * insigDim1Vals[rowNum]
            dataVec[:,insigDimensions[1]] = onesVec * insigDim2Vals[colNum]
            ax = plt.subplot(gridRes[0],gridRes[1],plotNum)
            decisionFunction = classifier.decision_function(dataVec).reshape(XX.shape)
            lvls = [0.4,0.5,0.6] if classifier.probability else [-1,0,1]
            cs2 = ax.contour(XX, YY, decisionFunction, colors='k', levels=lvls, alpha=1,linestyles=['dashed','solid','dotted'])
            csLabels2 = ['DF=-1','DF=0 (hypothesis)','DF=+1']
            if plotNum ==1:
                for i in range(len(csLabels2)):
                    cs2.collections[i].set_label(csLabels2[i])
            
            ### Tagging and labeling the axes:
            if plotNum <= gridRes[1]:
                ax.set_title(f'{dimNames[insigDimensions[1]]} = {insigDim2Vals[colNum]:.4f}')
            if plotNum%gridRes[1]==0:
                ax2 = ax.twinx()
                ax2.set_ylabel(f'{dimNames[insigDimensions[0]]} = {insigDim1Vals[rowNum]:.4f}')
                ax2.set_yticklabels([])
            if plotNum%gridRes[1]==1:
                ax.set_ylabel(dimNames[sigDims[1]])
            if (plotNum+gridRes[1]) > (gridRes[0]*gridRes[1]):
                ax.set_xlabel(dimNames[sigDims[0]])

            ### Plotting the benchmark classifier
            if benchmark is not None:
                scores = benchmark.getScoreVec(dataVec).reshape(XX.shape)
                cs = ax.contour(XX,YY,scores, colors='r', levels = [benchmark.threshold], 
                    alpha = 1, linestyles = ['solid']) 
                cslabels = ['Actual Boundary']
                if plotNum==1:
                    for i in range(len(cslabels)):
                        cs.collections[i].set_label(cslabels[i])
            # Plotting the previous boundary for comparison:
            if prev_classifier is not None:
                prev_DF = prev_classifier.decision_function(dataVec).reshape(XX.shape)
                prev_thresh = 0.5 if prev_classifier.probability else 0
                cs = ax.contour(XX,YY,prev_DF, colors='g', levels = [prev_thresh], 
                    alpha = 1, linestyles = ['solid'])
                cslabels = ['Previous iteration']
                if plotNum==1:
                    for i in range(len(cslabels)):
                        cs.collections[i].set_label(cslabels[i])
            # Adding the comparison classifier:
            if comparison_classifier is not None:
                comp_Df = comparison_classifier.decision_function(dataVec).reshape(XX.shape)
                comp_thresh = 0.5 if comparison_classifier.probability else 0
                cs = ax.contour(XX,YY,comp_Df, colors='m', levels = [comp_thresh], 
                    alpha = 1, linestyles = ['solid'])
                cslabels = ['Compared']
                if plotNum==1:
                    for i in range(len(cslabels)):
                        cs.collections[i].set_label(cslabels[i])
            # Applying the space constraints if there is any:
            if len(constraints) > 0:
                results = np.array([np.apply_along_axis(cons, axis=1,arr=dataVec) for cons in constraints]).T
                taking = np.apply_along_axis(all, axis = 1,arr = results).astype(int)
                takingGrid = taking.reshape(XX.shape)
                reducedIdx = range(0,len(taking),23)
                cs3 = ax.contour(XX,YY,takingGrid, colors='orange',levels=[0.5],alpha =1, linestyles=['dashdot'])
                
                # Scatter plotting the violating, feasible and infeasible regions:
                labels = classifier.predict(dataVec)
                reducedLabels = labels[reducedIdx]
                reducedTaking = taking[reducedIdx]
                redXy = xy[reducedIdx,:]
                feasXy = redXy[reducedLabels==0,:]
                infeasXy = redXy[reducedLabels==1,:]
                noResXy = redXy[reducedTaking==0,:]
                if not constraintsLabeled:
                    constraintsLabeled = True
                    cs3.collections[0].set_label('Constraint(s)')
                    ax.scatter(feasXy[:,0], feasXy[:,1], s=0.2,color='lime',label='Feeasible Region')
                    ax.scatter(infeasXy[:,0], infeasXy[:,1], s=0.2,color='gold',label='Infeasible Region')
                    ax.scatter(noResXy[:,0], noResXy[:,1], s=0.2,color='orangered',label='Violationg constraints')
                else:
                    ax.scatter(feasXy[:,0], feasXy[:,1], s=0.2,color='lime')
                    ax.scatter(infeasXy[:,0], infeasXy[:,1], s=0.2,color='gold')
                    ax.scatter(noResXy[:,0], noResXy[:,1], s=0.2,color='orangered')
                    
            ax.grid(showGrid)
            # Updating the plot number, needed for locating the subplots.
            plotNum += 1
            
    if legend:
        fig.legend(loc = 'upper left',bbox_to_anchor=(1.03, 1.0))
    fig.tight_layout()
    if saveInfo is not None:
        saveFigures(saveInfo = saveInfo)
    if showPlot:
        plt.show()
        return 
    plt.close()
    return 
    
def _plotSpace3D(space: SampleSpace, 
                showPlot = True,
                classifier = None, 
                figsize = (6,6),
                legend = True, 
                benchmark = None,
                meshRes = 100,
                showGrid = False,
                saveInfo:SaveInformation = None,
                newPoints = None,
                explorePoints = None):
    
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

    if newPoints is not None:
        newPoints = np.array(newPoints)
        newPoints = newPoints.reshape(1,len(newPoints)) if len(newPoints.shape) <2 else newPoints
        legendLabel = 'Exploitative points' + ('s' if newPoints.shape[0]>1 else '')
        ax.scatter(newPoints[:,0], newPoints[:,1], newPoints[:,2], marker = 's',s = 20, label = legendLabel, color = 'm' )
    if explorePoints is not None:
        explorePoints = np.array(explorePoints)
        explorePoints = explorePoints.reshape(1,len(explorePoints)) if len(explorePoints.shape) <2 else explorePoints
        legendLabel = 'Exploratory point' + ('s' if explorePoints.shape[0]>1 else '')
        ax.scatter(explorePoints[:,0], explorePoints[:,1], explorePoints[:,2], marker = 's',s = 20, label = legendLabel, color = 'green' )
    ax.grid(showGrid)
    if saveInfo is not None:
        saveFigures(saveInfo=saveInfo)
    if showPlot:
        plt.show()
        return 
    plt.close()
    return 

def _plotSpace2D(space: SampleSpace, 
                classifier:StandardClassifier, 
                figsize = None, 
                meshRes=100, 
                benchmark = None,
                legend=True,
                newPoints = None,
                showGrid = False,
                explorePoints = None,
                saveInfo:SaveInformation = None,
                showPlot = True,
                constraints = []):
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
        # Inverse transforming the support vectors of the classifier since the SVs are a subset of the transformed data points. 
        suppoerVectots = classifier.getSupportVectors(standard=True)
        ax.scatter(suppoerVectots[:,0], suppoerVectots[:,1], s=80, 
                    linewidth = 1, facecolors = 'none', edgecolors = 'orange', label='Support vectors')
        decisionFunction = classifier.decision_function(xy).reshape(XX.shape)
        cs2 = ax.contour(XX, YY, decisionFunction, colors='k', levels=[-1,0,1], alpha=1,linestyles=['dashed','solid','dotted'])
        csLabels2 = ['DF=-1','DF=0 (hypothesis)','DF=+1']
        for i in range(len(csLabels2)):
            cs2.collections[i].set_label(csLabels2[i])                 
    
    # If the new point(s) is (are) passed to the function it will be shown differently than the evaluated points:
    if newPoints is not None:
        newPoints = np.array(newPoints)
        newPoints = newPoints.reshape(1,len(newPoints)) if len(newPoints.shape) <2 else newPoints
        legendLabel = 'Exploitative point' + ('s' if newPoints.shape[0]>1 else '')
        ax.scatter(newPoints[:,0], newPoints[:,1], marker = 's',s = 20, label = legendLabel, color = 'm' )
    if explorePoints is not None:
        explorePoints = np.array(explorePoints)
        explorePoints = explorePoints.reshape(1,len(explorePoints)) if len(explorePoints.shape) <2 else explorePoints
        legendLabel = 'Exploitative point' + ('s' if explorePoints.shape[0]>1 else '')
        ax.scatter(explorePoints[:,0], explorePoints[:,1], marker = 's',s = 20, label = legendLabel, color = 'g')
    # Drawing the constraints:
    if len(constraints)>0:

        results = np.array([np.apply_along_axis(cons, axis=1,arr=xy) for cons in constraints]).T
        taking = np.apply_along_axis(all, axis = 1,arr = results).astype(int)
        takingGrid = taking.reshape(XX.shape)
        reducedIdx = range(0,len(taking),int(meshRes/8))
        cs3 = ax.contour(XX,YY,takingGrid, colors='orange',levels=[0.5],alpha =1, linestyles=['dashdot'])
        cs3.collections[0].set_label('Constraint(s)')
        
        # Scatter plotting the violating, feasible and infeasible regions:
        labels = classifier.predict(xy)
        reducedLabels = labels[reducedIdx]
        reducedTaking = taking[reducedIdx]
        redXy = xy[reducedIdx,:]
        feasXy = redXy[reducedLabels==0,:]
        infeasXy = redXy[reducedLabels==1,:]
        noResXy = redXy[reducedTaking==0,:]
        ax.scatter(feasXy[:,0], feasXy[:,1], s=0.2,color='lime',label='Feeasible Region')
        ax.scatter(infeasXy[:,0], infeasXy[:,1], s=0.2,color='gold',label='Infeasible Region')
        ax.scatter(noResXy[:,0], noResXy[:,1], s=0.2,color='orangered',label='Violationg constraints')

    ax.grid(showGrid)
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

