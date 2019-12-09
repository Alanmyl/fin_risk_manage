# -*- coding: utf-8 -*-
# File 'Configuration.py' completes the basic setting for the whole project
import pandas as pd
import numpy as np
import os
import datetime as dt
import scipy.stats as stats
from typing import Union


class conf:
    ''' class to define the basic public variables in the whole project.

    Attributes:
        year(int): presumed number of trade days in a year.
        s0: initial dollars of the portfolio.
        path: the working directory of the whole risk management codes
    '''

    def __init__(self, year=252):
        self.year = 252
        self.s0 = 10000
        self.path = '/home/alanmei/Alan/Columbia University/Financial Risk Management/'
        # self.start_date = dt.datetime(1997, 9, 5)


con = conf()
