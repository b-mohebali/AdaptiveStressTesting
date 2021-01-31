from yamlParseObjects.yamlObjects import * 
import logging
import os,sys
import subprocess

import platform
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum

import time

simConfig = simulationConfig('./yamlFiles/simulation.yaml')

print(simConfig)
