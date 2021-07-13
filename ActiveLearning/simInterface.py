#! /usr/bin/python3

import sys 
import repositories    
repositories.addCodeBaseToPath()

import os
import shutil
import logging
from typing import Dict
from yamlParseObjects.variablesUtil import *
log = logging.getLogger(__name__)
import contextlib

from rscad import *
import time

defaultRackNum = 18

class Case: 
    def __init__(self, caseLocation, 
                rackNum, 
                compile = False) -> None:
        self.caseLocation = caseLocation
        self.rackNum = rackNum

        self.draft = DftFile.from_directory(self.caseLocation)
        print(self.draft)
        self.sibFile = self.draft.sib()
        print(self.sibFile)
        self.loggerNumber = self.draft.nr_gloggers()
        self.draft.set_rack(self.rackNum)
        print(f"Parsing dft file: '{self.draft}'")
        rtds_sys = rtds.RtdsSystem.from_dft(self.draft)
        rtds_sys = self.draft.parse()
        rtds_sys.save_dft(self.draft)
        if compile: 
            self.draft.compile()
        
        self.draft.configure_rscad_startup_directories()

    def runCase(self,logDirectory):
        with glog.RscadConnection() as conn:
            self._runCase(conn, logDirectory=logDirectory)

    def _runCase(self,conn, logDirectory):
        with contextlib.suppress(FileNotFoundError, OSError):
            shutil.rmtree(logDirectory)
        os.makedirs(logDirectory)
        os.chdir(logDirectory)

        self.conn = conn
        self.conn.LoadBatch(self.sibFile.as_win())
        self.draft.parent.cd()
        
        # Initializing the loggers:
        self.loggers = glog.Gloggers()
        self.initializeLoggers(logDir = logDirectory)
        self.glog_files = []

        # doing trial 
        with self.startCase():
            time.sleep(5)
            self.setPulseLoad(enable = True)
            time.sleep(5)
            with self.dataRecord(logDir = logDirectory):
                time.sleep(20)

        print('Data record is done. waiting for the log files to arrive.')
        self.loggers.wait_on_dst_files()

        for logger in self.loggers:
            self.glog_files.append(fs.File(logger.dst_file))

    @contextlib.contextmanager
    def dataRecord(self, logDir):
        log.info('Starting data record...')
        self.initializeLoggers(logDir)
        self.loggers.enable()
        USE_RSCAD_PLOTS = False 
        if USE_RSCAD_PLOTS:
            self.conn.UnlockPlots()
        yield
        self.conn.LockPlots()
        self.loggers.disable()
        self.loggers.set_defaults()

    @contextlib.contextmanager
    def startCase(self):
        print('Starting the case....')
        startAttempts = 20
        for _ in range(startAttempts):
            self.conn._sync()
            locked = self.conn.rack_is_locked(self.rackNum)
            if locked:
                if locked != 'mark':
                    raise rscad.RackLockedException(f"Rack {self.rackNum} is locked by {locked}, exiting." )
                self.conn.UnlockRack(self.rackNum,'rtds')
                time.sleep(5)
                self.conn._sync()
                log.warning(f'Rack {self.rackNum} was locked before being unlocked. Retrying to start the case...')
                time.sleep(5)
            started = self.conn.Start(abort_on_error = False)
            self.conn._sync()
            log.info(started)
            locked = self.conn.rack_is_locked(self.rackNum)
            if started and locked:
                break
            log.warning(f'Case did not start. {_+1} of {startAttempts} retried...')
        else:
            raise Exception('Case did not start!')
        yield
        self.conn._sync()
        time.sleep(1)
        self.conn.Stop()
        self.conn._sync()

    def setSwitch(self, switchName, state = 1):
        self.conn.SetSwitch(f"Subsystem #1 : CTLs : Inputs : {switchName}", state)
    
    def setLogger(self, enable):
        self.setSwitch('AUsen', 1 if enable else 0)

    def setPulseLoad(self, enable = True):
        self.setSwitch('Pulse_En', 1 if enable else 0)

    def getLogger(self, logDirectory, rackPort = 10, id = 'A'):
        logger = glog.Glogger(
            rack_nb = self.rackNum,
            rack_port = rackPort,
            id = id,
            log_signals_file = f'Scripts/{id}.txt'
        )
        logger.cfg_str = logger._cfg_defaults()
        logger.dst_file = fr'{logDirectory}\rack{self.rackNum}-port{rackPort}.mat'
        logger.cfg_str = glog.Glogger.get_config(bit_nr = 0, move_to=logger.dst_file)
        logger.enable = self.setLogger
        return logger

    def initializeLoggers(self,logDir):
        self.conn._sync()
        print(fs.File(os.getcwd()).abs_str())
        self.loggers.append(self.getLogger(logDir, 10,'A'))
        self.loggers.cfg_before_start(self.conn)
        self.loggers.disable()
        for logger in self.loggers:
            fs.File(logger.dst_file).rm(verbose = False)
        return 


def setVariables(draftFile, 
                variableDict, 
                inplace = True, 
                varFileName = 'variableValues.yaml'):
    '''
        This function gets a draft file object and a dictionary in the form of (key,Value)=(variable name: str, variable value: float) and:
            - Sets the variables in the draft file according to the values in the dictionary.
            - Saves the draft file in place.
            - Saves a yaml file with the name of the variables and their new values.

        Returns:
            - The absolute path to the yaml file that describes the variable values.
    '''
    rtdsSys = rtds.RtdsSystem.from_dft(draftFile)
    draftVars = rtdsSys.get_draftvars()
    sliders = rtdsSys.get_sliders()
    # Forming the draft variables 
    draftVarsDict = {}
    slidersDict = {}
    for var in draftVars:
        draftVarsDict[var['Name']] = var
    for slider in sliders: 
        slidersDict[slider['Name']] = slider

    # Looking for the variables in the draft file before setting their new values: 
    for varName in variableDict:
        varValue = variableDict[varName]
        print(varName, varValue)
        if varName in draftVarsDict:
            print('Found ', varName, ' in draft variables')
            draftVarsDict[varName]['Value'] = varValue
            continue
        if varName in slidersDict:
            print('found ', varName, ' in sliders')
            slidersDict[varName]['Init'] = varValue
            continue
        log.warning(f'Variable {varName} was not found in the draft file. Moving on...')
    # Saving the new draft file in place:
    if inplace:
        print('Saving to ', draftFile.str())
        rtdsSys.save_dft(fpath = draftFile)
        draftFile.compile()
    # Saving the yaml file that describes the variable values:
    os.chdir(repositories.currentDir)
    saveVariableValues(variableDict, varFileName)
    absPath = f'{repositories.currentDir}/{varFileName}'
   
    return absPath

def runSample(caseLocation,
            sampleDictList, 
            remoteRepo, 
            sampleGroup = None):
    dFolder = repositories.outputLocation.mounted
    print(dFolder)
    indexGroup = range(1,len(sampleDictList)+1) if sampleGroup is None else sampleGroup
    print([_ for _ in indexGroup])
    realTimeCase = Case(caseLocation=caseLocation,
                    rackNum = repositories.rackNum)
    for sampleIndex in indexGroup:
        sample = sampleDictList[sampleIndex - 1]
        outFile = realTimeCase.draft
        varDescYaml = setVariables(outFile, sample, inplace = True)
        realTimeCase.runCase(repositories.outputLocation.absolute)
        newF = createSpecificDataFolder(remoteRepo, sampleIndex)
        # shutil.copyfile(varDescYaml, f'{newF.rstrip("/")}/{os.path.basename(varDescYaml)}')
        copyDataToremoteServer(newF, varDescYaml, isFolder = False)
        copyDataToNewLocation(newF, dFolder)
        copyDataToremoteServer(newF, outFile, isFolder = False)
        print('removed the extra folders from the source repository.')
        print(f'Done with the experiment {sampleIndex} and copying files to the repository.')
    print('This is the working directory after the sample is done: ',os.getcwd())
    os.chdir(repositories.currentDir)
    return

def runSampleFrom(caseLocation, sampleDictList, remoteRepo = None, fromSample = None):
    dFolder = repositories.outputLocation.mounted
    N = len(sampleDictList)
    # Setting up the sample group:
    if fromSample is not None: sampleGroup = range(fromSample, N+1)
    else: sampleGroup = range(N)
    
    print('Starting sample: ')
    print(sampleDictList[fromSample-1])
    runSample(caseLocation=caseLocation,sampleDictList=sampleDictList,dFolder=dFolder,remoteRepo=remoteRepo,sampleGroup=sampleGroup)
    return

def runSinglePoint(caseLocation: str,
                sampleDict,
                remoteRepo: str,
                sampleNumber: int):
    dFolder = repositories.outputLocation.mounted
    realTimeCase = Case(caseLocation = caseLocation, 
            rackNum = repositories.rackNum)
    outFile = realTimeCase.draft
    varDescYaml = setVariables(outFile, sampleDict, inplace = True)
    realTimeCase.runCase(logDirectory=repositories.outputLocation.absolute)
    newF = createSpecificDataFolder(remoteRepo, sampleNumber)
    copyDataToremoteServer(newF, varDescYaml, isFolder = False)
    copyDataToNewLocation(newF, dFolder)
    copyDataToremoteServer(newF, outFile, isFolder = False)
    print('removed the extra folders from the source repository.')
    print(f'Done with the experiment {sampleNumber} and copying files to the repository.')
    return newF

def FFD(modelLoc, variables,simRepo,res4 = True):
    '''
        This function uses the util tools in the codebase to perform a fractional factorial design set of experiments on a given model. 

        Returns:
            - True if the operation is successful
            - False if otherwise.
    '''
    print(f'Running the FFD experiment for model at {modelLoc}')
    experFile = repositories.experimentsLoc + 'FFD.txt'
    descFile = repositories.currentDir + '/varDescription.yaml'
    exper = fractionalFactorialExperiment(variables, res4 = res4)
    return runExperiment(modelLoc, variables, simRepo, exper, experFile, descFile)
    
def strictOAT(modelLoc, variables, simRepo):
    '''
        This function uses the util tools in the codebase to perform a strict OAT set of experiments on a given model. 

        Returns:
            - True if the operation is successful
            - False if otherwise.
    '''
    experFile = repositories.experimentsLoc + 'strict_OAT.txt'
    descFile = repositories.currentDir + '/varDescription.yaml'
    exper = strictOATSampleGenerator(variables)
    return runExperiment(modelLoc, variables, simRepo, exper, experFile, descFile)
    
def standardOAT(modelLoc, variables, simRepo):
    '''
        This function uses the util tools in the codebase to perform a standard OAT set of experiments on a given model. 

        Returns:
            - True if the operation is successful
            - False if otherwise.
    '''
    experFile = repositories.experimentsLoc + 'standard_OAT.txt'
    descFile = repositories.currentDir + '/varDescription.yaml'
    exper = standardOATSampleGenerator(variables)
    return runExperiment(modelLoc, variables, simRepo, exper, experFile, descFile)

def runExperiment(modelLoc, variables,simRepo, experiment, experFile, descFile):
    saveSampleToTxtFile(experiment, fileName = experFile)
    saveVariableDescription(variables, descFile)
    copyDataToremoteServer(simRepo, experFile, isFolder=False)
    copyDataToremoteServer(simRepo,descFile, isFolder=False)
    runSample(caseLocation=modelLoc,
                sampleDictList=experiment,
                remoteRepo=simRepo,
                sampleGroup = [8])
    return True

if __name__=='__main__':
    pass