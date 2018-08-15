# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 14:17:28 2016

@author: celin
"""

import os
from subprocess import call

rootdir = os.getcwd()

for root, dirs, files in os.walk(".", topdown=False):
    bashfile = [i for i in files if i.split('.')[1]=='sh']
    if len(bashfile)>0:
        for b in bashfile:
            os.chdir(root)
            call(['sbatch', b])
#            print os.path.join(root, b)
#            print os.getcwd()
            os.chdir(rootdir)