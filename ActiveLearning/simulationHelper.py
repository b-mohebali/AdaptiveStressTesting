
from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import * 
import platform
from repositories import *
import logging
import os
from typing import Dict


from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
import simulation




# This is added from my cubicle machine.
class SaveType(Enum):
    SAVE_ALL = 1
    SAVE_ONE = 2

class OATSampleMetod(Enum):
    STANDARD = 1
    STRICT = 2

class StorageInfo():
    def __init__(self, dFolder, rRepo):
        self.dFolder = dFolder
        self.rRepo = rRepo

class VariableNotFoundInModel(Exception):
    pass


class PGM_control(Control):
    NAME = 'PGM_control'
    def __init__(self, ctrl_str, controls_dir, configFile):
        super().__init__(ctrl_str, controls_dir)
        logging.info('Instantiating the PGM control')
        self.start_file_name = None
        self.ctrl_str        = ctrl_str
        self.start_file      = None
        self.controls_dir    = controls_dir
        self.simConfig = configFile
        self.folder = f'{case_Setup.CEF_BASE_DIR}/{self.simConfig.modelLocation}'
        print(f'File folder: {self.folder}')
        self.simulation = self.pull_case(self.folder+'/')
        self.dft_file = self.simulation.dft_file
        self.simulation.set_run_script('Start_Case.scr')
        self.rtds_sys = rtds.RtdsSystem.from_dft(self.dft_file.str())
        self.simulation.set_int_control(internal_ctrl = True)
    
    def _setVariableValues(self, randValues, saveOriginal = False):
        print('------------------------------------------------------------')
        print('Setting draft variables to specific values within their range')
        draftVars = self.rtds_sys.get_draftvars()
        sliders = self.rtds_sys.get_sliders()
        for dftVar in draftVars:
            if dftVar['Name'] in randValues:
                print(f'Variable name: {dftVar["Name"]}, value before change: {dftVar["Value"]}, Value after change: {randValues[dftVar["Name"]]}')
                dftVar['Value'] = randValues[dftVar['Name']]
        for sldr in sliders:
            if sldr['Name'] in randValues:
                print(f'Slider name: {sldr["Name"]}, value before change: {sldr["Init"]}, Value after change: {randValues[sldr["Name"]]}')
                sldr['Init'] = randValues[sldr['Name']]
        outfile = f'{case_Setup.DRIVE_C}/testing/fileman/{self.simConfig.modelName}.dft'
        self.rtds_sys.save_dft(fpath = outfile)
        if saveOriginal:
            self.rtds_sys.save_dft(fpath = f'{self.folder}/{self.simConfig.modelName}.dft')
        # Waiting for the draft file to be saved:
        print('------------------------------------------------------------')
        return outfile

    # This function will set all the variables to their nominal values.
    def setVariablesToInitialState(self, variables):
        inits = getVariablesInitialValueDict(variables)
        outfile = self._setVariableValues(inits, saveOriginal=True)
        return outfile
    
    def setVariablesToRandom(self, variables, variableFile = SaveType.SAVE_ALL):
        timeIndepVars = getTimeIndepVarsDict(variables)
        randVars = randomizeVariables(timeIndepVars)
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        outfile = self._setVariableValues(randVars)
        return outfile
    
    # This function gets the already randomized list of the variables
    # and their values.
    def setVariables(self, randVars, variableFile = SaveType.SAVE_ALL):
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        outfile = self._setVariableValues(randVars)
        return outfile
        

#-------------------------------------------------------------
def runSample(sampleDictList, 
            dFolder, 
            remoteRepo, 
            sampleGroup = None, 
            simConfig: simulationConfig = None):
    indexGroup = range(1, len(sampleDictList)+1)
    myControl = PGM_control('', './', configFile=simConfig)   
    if sampleGroup is not None: 
        indexGroup = sampleGroup
    for sampleIndex in indexGroup:
        sample = sampleDictList[sampleIndex - 1]
        outfile = myControl.setVariables(sample)
        testDropLoc = Trial.init_test_drop(myControl.NAME)
        ctrl = myControl
        ctrl.initialize()
        trial = Trial(ctrl, ctrl.simulation, testDropLoc)
        # # HACK. This checks if it has to do fm metrics. 
        case_Setup.fm = False 
        trial.runWithoutMetrics()
        ### This is where the output is copied to a new location. 
        newF = createSpecificDataFolder(remoteRepo, sampleIndex)
        shutil.copyfile(f"{currentDir}/variableValues.yaml", f'{newF.rstrip("/")}/variableValues.yaml')
        copyDataToNewLocation(newF, dFolder)
        copyDataToremoteServer(newF, outfile, isFolder = False)
        print('removed the extra folders from the source repository.')
        print(f'Done with the experiment {sampleIndex} and copying files to the repository.')
    print('This is the working directory after the sample is done: ',os.getcwd())
    os.chdir(currentDir)
    return
 

def runSampleFrom(sampleDictList, dFolder, remoteRepo = None, fromSample = None):
    N = len(sampleDictList)
    if fromSample is not None:
        sampleGroup = range(fromSample, N+1)
    else:
        sampleGroup = range(N)
    print('Starting sample: ')
    print(sampleDictList[fromSample-1])
    runSample(sampleDictList, dFolder, remoteRepo = remoteRepo, sampleGroup=sampleGroup)
    return

"""
    Runs a single point in the design space and saves the data:

    NOTE: This is copied from the function that runs a series of samples on the same 
        model. The runSample function may have to be changed to just loop through 
        samples using this function as a building block.
"""
def runSinglePoint(sampleDict: Dict[str, float],
                dFolder: str, 
                remoteRepo: str,
                simConfig: simulationConfig,
                sampleNumber: int,
                modelUnderTest = None):
    if modelUnderTest is None:
        modelUnderTest = PGM_control('','./', configFile=simConfig)
    outfile = modelUnderTest.setVariables(sampleDict)
    testDropLoc = Trial.init_test_drop(modelUnderTest.NAME)
    modelUnderTest.initialize()
    trial = Trial(modelUnderTest, modelUnderTest.simulation, testDropLoc)
    case_Setup.fm = False
    trial.runWithoutMetrics()
    newF = createSpecificDataFolder(remoteRepo, sampleNumber)
    shutil.copyfile(f'{currentDir}/variableValues.yaml', f'{newF.rstrip("/")}/variableValues.yaml')
    copyDataToNewLocation(newF, dFolder)
    copyDataToremoteServer(newF, outfile, isFolder = False)
    print('removed the extra folders from the source repository.')
    print(f'Done with the experiment {sampleNumber} and copying files to the repository.')
    return newF


