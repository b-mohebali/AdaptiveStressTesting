import os
import case_Setup

# Repo for the Monte-Carlo sample 
dataRepo = '/home/caps/.wine/drive_c/SCRATCH/mohebali/Data/SensAnalysis/'
dataFolder = case_Setup.LOGGER_OUTPUT

repoRoot = 'caps@10.146.64.68:/home/caps/SensAnalysis/'
repoRoot2 = 'caps@10.146.64.69:/home/caps/SensAnalysis/'
localRepo = '/home/caps/SenseAnalysisTemp/'
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

# First adaptive sampling repo. Test: initial sample + metrics evaluation + initial classifier training and visualization.
adaptRepo1 = adaptiveSamplingLoc + 'adaptiveRepo1'
# test repo:
remoteRepo83 = localRepo + 'sample83'

# A large sample that will act as the benchmark for the algorithm:
motherSample = adaptiveSamplingLoc + 'motherSample'
# Adaptive Sampling repo with budget=300 and initial sample size=80
adaptRepo2 = adaptiveSamplingLoc + 'adaptiveRepo2'
# Adaptive Sampling repo with budget = 400 samples and initial sample size=80
adaptRepo3 = adaptiveSamplingLoc + 'adaptiveRepo3'



currentDir = os.getcwd()
isRepoRemote = True