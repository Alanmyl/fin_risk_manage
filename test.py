# -*- coding: utf-8 -*-
from Plots import *
from Risk import *
from utils import *
from Estimate import *
files = [pre_data(con.path+'data/'+path) for path in os.listdir(con.path+'data/')]
nums = [3 for x in files]
paras = est_Ngbm_win(win_len=5,nums=nums,files=files)
estimate('RollWinNGBM',win_len=5,nums=nums,files=files)
print(dt.datetime.now())
#risk = risk_Ngbm_mc_short('ES',0.975,paras,size=1000000)
print(dt.datetime.now())