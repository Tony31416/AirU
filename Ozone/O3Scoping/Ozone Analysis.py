# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:30:28 2018

@author: jbuss
"""

import os
import pandas as pd
import numpy as np

path = "C:\\Users\\jbuss\\Google Drive\\AirU Folder\\scripts\AirU (github)\\AirU\\Ozone\\O3Scoping\\" # insert path
ozone = [] 
for subdir, dirs, files in os.walk(path): #create list of pathways to ozone analysis files
    for file in files:
        if file.startswith("O3AnalysisOutput"):
            ozone.append(os.path.join(subdir,file))
dfs = [] 
for filename in ozone: # reates list of opened csv files
    dfs.append(pd.read_csv(filename,))


Master = pd.concat(dfs, ignore_index = True) # puts csv files into one data frame
Master.drop('Unnamed: 0', axis = 1 , inplace = True)
Master

    