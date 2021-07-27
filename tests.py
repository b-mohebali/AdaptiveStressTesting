#! /usr/bin/python3

import repositories as repos 
repos.addCodeBaseToPath()

from rscad import * 
from rscad.fs import *

scratchPath = get_plasma_scratch_dir()
print(scratchPath)
s = glog.Glogger.cfg_options(18,10)
print(s)