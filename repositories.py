import case_Setup
import os

# Repo for the Monte-Carlo sample 
dataRepo = '/home/caps/.wine/drive_c/SCRATCH/mohebali/Data/SensAnalysis/'
dataFolder = case_Setup.LOGGER_OUTPUT
remoteRepo  = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample1'
# Repo for the first OAT design sample. 
remoteRepo2 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample2'
# Repo for res 4 fractional factorial design sample with 20% variations. 
remoteRepo3 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample3'
# repo 4 is for the second OAT sample
remoteRepo4 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample4'
# repo 5 is for the third OAT sample
remoteRepo5 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample5'
# repo 6 for Res 4 FF Design with 10% variations
remoteRepo6 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample6'
# repo 7 for Standard OAT Samples
remoteRepo7 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample7'
# repo 8 for Monte-Carlo Samples with limited variable space
remoteRepo8 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample8'
# repo 9 for Monte-Carlo Samples with limited variable space with logarithmic range.
remoteRepo9 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample9'
# repo 10 for FFD with larger limits (50%) and the new scenario (just high load with a fixed length)
remoteRepo10 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample10'
# repo 11 for FFD with larger limits (50%) and the new scenario (just high load with a fixed length)
# Also the filter parameters are unified for both windings
remoteRepo11 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample11'
# This sample is mean for verification of the results for other samples.
remoteRepo12 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample12'
# Verification sample
remoteRepo13 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample13'
# Verification sample after 3 sec delay was added to the variable setting function.
remoteRepo14 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample14'
# New fractional factorial design dataset
remoteRepo15 = 'caps@10.146.64.67:/home/caps/SensAnalysis/sample15'
# Test repo
testRepo = 'caps@10.146.64.67:/home/caps/SensAnalysis/testRepo'


currentDir = os.getcwd()
isRepoRemote = True