import matplotlib.pyplot as plt 
import pickle 
from ActiveLearning.visualization import plotSpace
from ActiveLearning.Sampling import SampleSpace


pickleName= 'results_outcome_35.pickle'
dataFile = f'C:/Users/bm12m/Google Drive/codes/ScenarioGenerator/assets/outputReports/130/{pickleName}'


with open(dataFile, 'rb') as pickleIn:
    data = pickle.load(pickleIn)
print(data)
clf = data['clf'] 
bench = data['benchmark']
space = data['space'] 

meshRes = 100 
figSize = (9,7)
gridRes = (7,7)

plotSpace(space, 
        classifier=clf, 
        figsize = figSize, 
        meshRes=meshRes,
        legend = True,
        gridRes = gridRes, 
        showPlot=True, 
        benchmark = bench,
        extraLevels = [-4.79])
