import logging
import os


def getLoggerFileAddress(absolutePath='./', fileName = 'app'):
    logDirectory = absolutePath + 'log/'
    fileNameEx = fileName + ".log"
    if os.path.isdir(logDirectory) is False:
        os.mkdir(logDirectory)
    fileAddress = logDirectory + fileNameEx
    return fileAddress

def logCommand(cmd):
    cmdString = 'Running "'
    for pars in cmd:
        cmdString += pars + ' '
    logging.info(cmdString + '"')


