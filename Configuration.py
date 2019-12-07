# -*- coding: utf-8 -*-
# File 'Configuration.py' completes the basic setting for the whole project
import pandas as pd
import numpy as np
import datetime as dt
import os
import scipy.stats as stats
import scipy.optimize as optimize

class conf:
    def __init__(self,year=252):
        self.year = 252
        self.s0 = 10000
        self.path = '/home/alanmei/Alan/Columbia University/Financial Risk Management/'
        self.start_date = dt.datetime(1997,9,5)

con = conf()