# -*- coding: utf-8 -*-
from Plots import *
from Risk import *
from utils import *
from Estimate import *
files = [pre_data(con.path+'data/'+path) for path in os.listdir(con.path+'data/')]
nums = [3 for x in files]
paras = est_Ngbm_win(win_len=5,nums=nums,files=files)
#estimate('RollWinNGBM',win_len=5,nums=nums,files=files)
#risk = risk_Ngbm_mc_short('ES',0.975,paras,size=1000000)
para = est_gbm_win(5,files[0])
#test = risk_opt_stck('VaR',0.975,'put',0.01,para.iloc[:1])
print(dt.datetime.now())
test = hedge_opt_stck('ES',0.8,0.975,'put',para.iloc[:5],init=0.01)
print(dt.datetime.now())
print(test)