# -*- coding: utf-8 -*-
from Plots import *
from Risk import *
from utils import *
from Estimate import *
options = pre_data(con.path+'data/options/MMM_option.csv')
stock = pre_data(con.path + 'data/stocks/MMM.csv')
risk_free = pre_data(con.path + 'data/TNX.csv')['Close']
start = dt.datetime.now()
risk1 = risk_opt_stck('VaR', 0.95, 'put', 0.08, est_gbm_win(
    5, stock), risk_free, options['volat'])
print(dt.datetime.now() - start)
risk2 = risk_opt_stck('VaR', 0.95, 'put', 0.0, est_gbm_win(
    5, stock), risk_free, options['volat'])
"""

risk3 = risk_gbm('VaR', 0.95, est_gbm_win(
    5, stock))"""
test = hedge_opt_stck('VaR', 0.95, 'put', 0.8, est_gbm_win(
    5, stock), risk_free, options['volat'], 0.0)
