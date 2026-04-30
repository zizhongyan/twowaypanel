"""
Datasets module

Created on Tue Sep 9 2025

Author: Zizhong Yan @ copyright
"""

import pandas as pd
import pkg_resources 

def angristevans98():
    """
    Load the data and return a dataset class instance.
    """
    path = pkg_resources.resource_filename('twowaypanel', 'database/')
    data = pd.read_stata(path+'/angristevans98.dta')  
    return data