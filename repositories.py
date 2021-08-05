import os
from yamlParseObjects.yamlObjects import simulationConfig
import sys

dataFolder = '/home/caps/.wine/drive_c/SCRATCH/Mohebali/Data/AC_PGM_LogFiles'

# This is the number of the rack that will be used for the simulation: 
rackNum = 18
##_______________________________________________________
# Location of the assets local to the codebase:
assetsLoc = './assets/'
yamlFilesLoc = assetsLoc + 'yamlFiles/'
picklesLoc = assetsLoc + 'pickles/'
experimentsLoc = assetsLoc + 'experiments/'

##_______________________________________________________
# Assets on Plasma:


plasmaLoc = '/home/caps/.wine/drive_c/'
cefLoc = plasmaLoc + 'cef/'

class outputLocation:
    mounted = '/home/caps/.wine/drive_c/SCRATCH/Mohebali/Data/AC_PGM_LogFiles'
    absolute = r'\\plasma.caps.fsu.edu\SCRATCH\Mohebali\Data\AC_PGM_LogFiles'

codebasePath = plasmaLoc + 'HIL-TB/py'

def addCodeBaseToPath():
    sys.path.append(codebasePath)
    return 

##_______________________________________________________
# Location of the Linux machine assets and data files:
# Repo for the Monte-Carlo sample 
dataRepo = '/home/caps/.wine/drive_c/SCRATCH/mohebali/Data/SensAnalysis/'

repoRoot = 'caps@10.146.64.68:/home/caps/SensAnalysis/'
repoRoot2 = 'caps@10.146.64.69:/home/caps/SensAnalysis/'
localRepo = '/home/caps/SenseAnalysisTemp/'
newLocalRepo = '/home/caps/Data/SensAnalysis/'
newAdaptiveRepo = '/home/caps/Data/AdaptiveSamplingRepo/'
newTestRepo = '/home/caps/Data/Tests/'
adaptiveSamplingLoc = '/home/caps/AdaptiveSamplingRepo/'

# New FFD sample with larger variable list. 
remoteRepo1 = repoRoot + 'sample1'
# Repo for the first OAT design sample. 
remoteRepo2 = repoRoot + 'sample2'
# Repo for res 4 fractional factorial design sample with 20% variations. 
remoteRepo3 = repoRoot + 'sample3'
# repo 4 is for the second OAT sample
remoteRepo4 = repoRoot + 'sample4'
# repo 5 is for the third OAT sample
remoteRepo5 = repoRoot + 'sample5'
# repo 6 for Res 4 FF Design with 10% variations
remoteRepo6 = repoRoot + 'sample6'
# repo 7 High dimension OAT sample
remoteRepo7 = repoRoot + 'sample7'
# repo 8 for Monte-Carlo Samples with limited variable space
remoteRepo8 = repoRoot + 'sample8'
# repo 9 for Monte-Carlo Samples with limited variable space with logarithmic range.
remoteRepo9 = repoRoot + 'sample9'
# repo 10 for FFD with larger limits (50%) and the new scenario (just high load with a fixed length)
remoteRepo10 = repoRoot + 'sample10'
# repo 11 for FFD with larger limits (50%) and the new scenario (just high load with a fixed length)
# Also the filter parameters are unified for both windings
remoteRepo11 = repoRoot + 'sample11'
# This sample is mean for verification of the results for other samples.
remoteRepo12 = repoRoot + 'sample12'
# Verification sample
remoteRepo13 = repoRoot + 'sample13'
# Verification sample after 3 sec delay was added to the variable setting function.
remoteRepo14 = repoRoot + 'sample14'
# New fractional factorial design dataset
remoteRepo15 = repoRoot + 'sample15'
# Verification sample with new factor list 16 + 1
remoteRepo16 = repoRoot + 'sample16'
# Strict OAT sample with 17 factors
remoteRepo17 = repoRoot + 'sample17'

## -------------- Set of samples with 26 factors:
# FFD after adding the time constants for the machine and the exciter to the list. 
remoteRepo18 = repoRoot + 'sample18'
# Standard OAT after adding the time constants for the machine and the exciter to the list. 
remoteRepo19 = repoRoot + 'sample19'
# Strict OAT after adding the time constants for the machine and the exciter to the list. 
remoteRepo20 = repoRoot + 'sample20'
# Fractional factorial design sample with shuffled factor list
remoteRepo21 = repoRoot + 'sample21'
# Verification sample with 26 factors
remoteRepo22 = repoRoot + 'sample22'
# Another shuffled FFD sample because sample 18 and 21 did not match as expected. 
remoteRepo23 = repoRoot + 'sample23'

# Verification sample after changing the filter values with only filter factors after Rc reduction
remoteRepo24 = repoRoot + 'sample24'
# FFD sample only for the filter factors after Rc reduction
remoteRepo25 = repoRoot + 'sample25'
# FFD sample with time constants without the 0-sequence factors. +-20%
remoteRepo26 = repoRoot + 'sample26'
# Standard OAT sample with repeats for verification of the FFD sample
remoteRepo27 = repoRoot + 'sample27'
# FFD sample similar to sample26 with shuffled factor list
remoteRepo28 = repoRoot + 'sample28'
# FFD sample similar to sample26 with shuffled factor list
remoteRepo29 = repoRoot + 'sample29'
# FFD with extended initialization period and ONLY FILTER factor list.
remoteRepo30 = repoRoot + 'sample30'
# FFD with extended initialization period and filter+0sequence+Neut-gnd res factor list.
remoteRepo31 = repoRoot + 'sample31'
# FFD with filter+ and added phase current signals for 0-sequence current metric.
remoteRepo32 = localRepo + 'sample32'
# Standard OAT with the test filter (connected to an AC source only)
remoteRepo33 = localRepo + 'sample33'
# Same as sample33 but taken again. The capacitor component was changed with the cap found on the search. The unit of the draft variable was set as seconds (by default)
remoteRepo34 = localRepo + 'sample34'
# Same as sample33 and sample34 but the draft var unit is uF.
remoteRepo35 = localRepo + 'sample35'
# Same as sample33 with the same draft variable but different component taken from the search. 
remoteRepo36 = localRepo + 'sample36'
# Same but the range of the new draft variable changed. 
remoteRepo37 = localRepo + 'sample37'
# FFD sample after the ranges for the cap draft variable are changed to cover the whole variation range. 
remoteRepo38 = localRepo + 'sample38'
# Standard OAT with L and C and the signals of the test filter.
remoteRepo39 = localRepo + 'sample39'
# Same as 39 but with C_filter (different draft variable) governing the capacitance for the test filter and actual filters. 
remoteRepo40 = localRepo + 'sample40'
# Standard OAT with C_filter moved inside the draft variable block.
remoteRepo41 = localRepo + 'sample41'
# Standard OAT sampel 2ith 3+1 factors. Changing C_filter name to GaMaCf
remoteRepo42 = localRepo + 'sample42'
# Standard OAT like 41
remoteRepo43 = localRepo + 'sample43'
# FFD After changing the name of the capacitor draft variable.
remoteRepo44 = localRepo+ 'sample44'
# Standard OAT after creating GaMaRfil for the Capacitor resistance
remoteRepo45 = localRepo + 'sample45'

# Repository for test data
remoteRepoTest = localRepo + 'testSample'

# FFD sample after adding the factors for governor
remoteRepo46 = localRepo + 'sample46'
# FFD sample after Tc and Tb values were corrercted
remoteRepo47 = localRepo + 'sample47'
# Standard OAT sample with 40 factors
remoteRepo48 = localRepo + 'sample48'
# Strict OAT sample with 40 samples
remoteRepo49 = localRepo + 'sample49'

# FFD sample with smaller variation to see if the instability seen in sample 40 can be avoided while the sensitivity of the model can be analyzed. 
remoteRepo50 = localRepo + 'sample50'
# FFD sample with Turbine factors and 15% variation range (a mistake. We needed 20% but I did not delete the sample. I got another one)
remoteRepo51 = localRepo + 'sample51'
# FFD sample with turbine factors and 20% variation range for all the factors. 
remoteRepo52 = localRepo + 'sample52'

# FFD sample with some added factors for Exciter, variable span was the same as the last sample at 20%
remoteRepo53 = localRepo + 'sample53'
# Standard OAT sample with the same factor list as 53
remoteRepo54 = localRepo + 'sample54'
# Strict OAT sample with the same factor list as 53
remoteRepo55 = localRepo + 'sample55'
# FFD sample with complete factor list (44 + 1) and 15% variable span.
remoteRepo56 = localRepo + 'sample56'
# FFD sample with complete factor list (44 + 1) and 15% variable span and shuffled factor list. 15% variation
remoteRepo57 = localRepo + 'sample57'
# FFD sample with complete factor list (44 + 1) and 15% variable span and shuffled factor list. 20% variation
remoteRepo58 = localRepo + 'sample58'

# Testing the SA module after a long time!
remoteRepo59 = localRepo + 'sample59'


# First repo for AC PGM:
remoteRepo60 = localRepo + 'sample60'
# AC PGM with GB load excluded:
remoteRepo61 = localRepo + 'sample61'
# AC PGM changed the limits of the variables
remoteRepo62 = localRepo + 'sample62'

# 10 Samples with the same settings to check the effect of noise on the process:
remoteRepo63 = localRepo + 'sample63'
remoteRepo64 = localRepo + 'sample64'
remoteRepo65 = localRepo + 'sample65'
remoteRepo66 = localRepo + 'sample66'
remoteRepo67 = localRepo + 'sample67'
remoteRepo68 = localRepo + 'sample68'
remoteRepo69 = localRepo + 'sample69'
remoteRepo70 = localRepo + 'sample70'
remoteRepo71 = localRepo + 'sample71'
remoteRepo72 = localRepo + 'sample72'

# Pushing the Pulse load init to .6 pu and steady state power UL to .9 pu
remoteRepo73 = localRepo + 'sample73'
remoteRepo74 = localRepo + 'sample74'

# Pushing the sum of the powers to 4 MW at their maximum
    # Ps = 2, Pp = 0.7
remoteRepo75 = localRepo + 'sample75'
    # Ps = 1, Pp = 1.7
remoteRepo76 = localRepo + 'sample76'

# Pusing the sum of the powers to 5 MW:
    # Ps = 2, Pp = 1.4
remoteRepo77 = localRepo + 'sample77'
    # Ps = 1.5, Pp = 1.9
remoteRepo78 = localRepo + 'sample78'

# Same as 77 but with the stabilization period extended from 10s to 25s. 
remoteRepo79 = localRepo + 'sample79'
# Same as 79 (for testing the effect of noise)
remoteRepo80 = localRepo + 'sample80'

# Same as 77 but on Rack 17:
remoteRepo81 = localRepo + 'sample81'

# test repo:
remoteRepo83 = localRepo + 'sample83'

# FFD Sample with span = 0.5 after droop was turned off on the AC PGM model:
remoteRepo84 = localRepo + 'sample84'
# FFD Sample with span = 0.65 after droop was turned off on the AC PGM model:
remoteRepo85 = localRepo + 'sample85'
# FFD with the model stabilization time span extended to 25s from 10s.
remoteRepo86 = localRepo + 'sample86'
# FFD sample after applying the constraints on the frequency and power of the pulse load:
remoteRepo87 = localRepo + 'sample87'
# Sample 87 was incorrect due to the limit setting that was done in the sampling process. 
#   Sample 88 does the same thing with the limit setting corrected: 
remoteRepo88 = localRepo + 'sample88'
# Same as sample 88 but the pulse is enabled after 15 seconds 
remoteRepo89 = localRepo + 'sample89'
# Drastically increased the range of variation for the variables. Especially the pulse load ramp rate and its frequency
remoteRepo90 = localRepo + 'sample90'

# Testing the new code base for the automation:
remoteRepo91 = newLocalRepo + 'sample91'
# Same as 91 but the vars with init=0 omitted:
remoteRepo92 = newLocalRepo + 'sample92'
remoteRepo93 = newLocalRepo + 'sample93'
remoteRepo94 = newLocalRepo + 'sample94'
remoteRepo95 = newLocalRepo + 'sample95'
remoteRepo96 = newLocalRepo + 'sample96'

# Monte Carlo sample using CVT sampling method:
remoteRepo97 = newLocalRepo + 'sample97'

# FFD with corrected limits: 
remoteRepo98 = newLocalRepo + 'sample98'
# FFD after the frequency reduced to 20 Hz
remoteRepo99 = newLocalRepo + 'sample99'
# Increased the upper limit of the steady state load to push the total power out of nominal value
remoteRepo100 = newLocalRepo + 'sample100'
# Increased the upper limit for the pulse load power to push the total power out of the nominal bound
remoteRepo101 = newLocalRepo + 'sample101'
# First repo to integrate the analysis and simulation steps into the same script:
remoteRepo102 = newLocalRepo + 'sample102'
# Sample with the metrics including the modulation
remoteRepo103 = newLocalRepo + 'sample103'
# Changed the range of variation. Took out the irrelevant variables from the analysis. Added the new metric "Voltage modulation".:
remoteRepo104 = newLocalRepo + 'sample104'



"""
    Adaptive Samples notation: 
    sample {Total sample size}({Initial sample size})-{Batch size}
    Example : 400(80)-1 -> A sample of 400 simulations with 80 samples taken initially and batches of size 1.
"""
# First adaptive sampling repo. Test: initial sample + metrics evaluation + initial classifier training and visualization.
adaptRepo1 = adaptiveSamplingLoc + 'adaptiveRepo1'
# A large sample that will act as the benchmark for the algorithm (2500 samples):
motherSample = adaptiveSamplingLoc + 'motherSample'
# A larger sample that will act as a benchmark for the algorithm (5000 samples lhs):
# NOTE: reason for use of LHS over CVT is that we can add to this sample 
#   later but not to the CVT sample. 
motherSample2 = adaptiveSamplingLoc + 'motherSample2'
# Large sample used as a benchmark dataset with 2500 samples (CVT) in a 4D space. 
monteCarlo2500 = newAdaptiveRepo + 'monteCarlo2500'
# Large sample with constrained space. The constraint is based on the pulse load power, frequnecy and the ramp rate.
constrainedSample = newAdaptiveRepo + 'constrainedSample'
# Another constrained sample after the first one came out all infeasible:
constrainedSample2 = newAdaptiveRepo + 'constrainedSample2'
# Benchmark sample after the bounds are tested again
constrainedSample3 = newAdaptiveRepo + 'constrainedSample3'

# Adaptive Sampling 300(80)-1
adaptRepo2 = adaptiveSamplingLoc + 'adaptiveRepo2'
# Adaptive Sampling 400(80)-1 Droop control on the voltage and frequency was engaged. 
adaptRepo3 = newAdaptiveRepo + 'adaptiveRepo3'
# Monte Carlo sample of size 400 taken as the baseline for the adaptive samples of the same size:
monteCarlo400 = adaptiveSamplingLoc + 'monteCarlo400'
# Adaptive Sampling 400(80)-1 Droop control disengaged. 
adaptRepo4 = newAdaptiveRepo + 'adaptiveRepo4'
# Adaptive Sampling 400(80)-1 Droop control engaged again. The range of variation for the variables are changed too.
adaptRepo5 = newAdaptiveRepo + 'adaptiveRepo5'
# Adaptive Sampling 400(80)-1 Droop control disengaged and voltage modulation added to the metrics. 
adaptRepo6 = newAdaptiveRepo + 'adaptiveRepo6'
# The last initial sample had all the points fail the metrics. So I relaxed the bounds for the variables and ran it again. Also changed the time steps to 50 uS
adaptRepo7 = newAdaptiveRepo + 'adaptiveRepo7'
# Same problem with the last sample. Changed the range of the variables and tried again. Also changed the L-L RMS voltage to 4160 v
adaptRepo8 = newAdaptiveRepo + 'adaptiveRepo8'
# Adaptive Sample after the range of the variables were established through frequency sweep experiments. 
adaptRepo9 = newAdaptiveRepo + 'adaptiveRepo9'
# Adaptive Sample 400(80)-1 after the mother sample was taken with 2425 samples. The rejection of the constrained area was added to the algorithm.
adaptRepo10 = newAdaptiveRepo + 'adaptiveRepo10'
# Adaptive Sample 400(100)-4. The batch sampling is enabled for the adaptive part. Only the metrics evalaution is done in parallel. The sample simulation is done sequentially. The exploration is NOT engaged yet. The initial sample is resampled due to the rejection effect.
adaptRepo11 = newAdaptiveRepo + 'adaptiveRepo11'
# Adaptive sample 400(100)-4 with batch sampling and exploration enabled. The exploration is one sample per 3 exploitative sample. The resource allocation part is not implemented yet. 
adaptRepo12 = newAdaptiveRepo + 'adaptiveRepo12'
 



"""
    Miscellanous tests with different purposes that are not meant for sensitivity analysis or adaptive stress testing. 

"""
# Test for seeing whether the metrics are calculated correctly. Pulse load disabled here. 
testRepo1 = newTestRepo + 'test1'
# Same test with droop disabled: 
testRepo2 = newTestRepo + 'test2'
# Same test but the time step was changed to 25 uS because the loggers had some issue with recording at 28.5uS
testRepo3 = newTestRepo + 'test3'
# Same but with Vb = 4160v
testRepo4 = newTestRepo + 'test4'
# To find a point of feasibility in the whole space.
testRepo5 = newTestRepo + 'test5'
# Adding more variables to the verification test:
testRepo6 = newTestRepo + 'test6'
# Adding all the variables of the model
testRepo7 = newTestRepo + 'test7'
# A 30 point frequency sweep on the model to find the regions of vulnerability.
testRepo8 = newTestRepo + 'test8'
# Frequency sweep with 50 points between 0.1 Hz and 7 Hz. Pulse load = 0.5 MW (10% nominal)
testRepo9 = newTestRepo + 'test9'
# Frequency sweep with 50 points between 0.1 Hz and 7 Hz. Pulse load = 0.25 MW (5% nominal), Steady state load = 2 MW 
testRepo10 = newTestRepo + 'test10'
# Frequency sweep with 50 points between 0.1 Hz and 10 Hz pulse load  0.25 MW, Steady state load = 1 MW. 
testRepo11 = newTestRepo + 'test11'
# Repeating test11:
testRepo12 = newTestRepo + 'test12'
# Frequency sweep with same frequency and pulse load levels as 11 and 10 but steady state load = 1.5 MW 
testRepo13 = newTestRepo + 'test13'
# Frequency sweep. SteadyState = 1.5MW, Pulse load = 0.5MW, Frequency: 0.1-10 Hz
testRepo14 = newTestRepo + 'test14'
# just to test the test for repo location
testRepo15 = newTestRepo + 'test15'

currentDir = os.getcwd()
isRepoRemote = True