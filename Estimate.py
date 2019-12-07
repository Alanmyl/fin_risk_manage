# -*- coding: utf-8 -*-
# File 'Estimate.py' contains functions to estimate parameters with different methods.
from Configuration import *
from utils import *
def est_gbm_win(win_len,file,period='daily'):
    df = logrtn(file).to_frame(name='logrtn')
    df['sig'] = df.logrtn.rolling(window=win_len*con.year).std()
    df['mu'] = df.logrtn.rolling(window=win_len*con.year).mean()+df.logrtn.rolling(window=win_len*con.year).var()/2
    df.index = file.index[1:]
    if period == 'annual':
        df.loc[:,'sig'] = df.sig*con.year**0.5
        df.loc[:,'mu'] = df.mu*con.year
    return df.dropna()

def est_gbm_exp(lam,file,period='daily'):
    df = logrtn(file).to_frame(name='logrtn')
    df['sig'] = df['logrtn'].ewm(alpha=1-lam).std()
    df['mu'] = df['logrtn'].ewm(alpha=1-lam).mean()+df['logrtn'].ewm(alpha=1-lam).var()/2
    df.index=file.index[1:]
    if period == 'annual':
        df.loc[:,'sig'] = df.sig*con.year**0.5
        df.loc[:,'mu'] = df.mu*con.year
    return df.dropna()

def est_2gbm_win(win_len,num1,num2,file1,file2,period='daily'):
    df = pd.concat([est_gbm_win(win_len,file1),est_gbm_win(win_len,file2)],axis=1,join='inner')
    df.columns = ['logrtn1','sig1','mu1','logrtn2','sig2','mu2']
    df['rho'] = ((df.logrtn1-df.logrtn1.mean())*(df.logrtn2-df.logrtn2.mean())/df.sig1/df.sig2).rolling(window=5*con.year).mean()
    df['w'] = build_port2(num1,num2,file1,file2)['w'].loc[df.index]
    if period == 'annual':
        df.loc[:,'sig'] = df.sig*con.year**0.5
        df.loc[:,'mu'] = df.mu*con.year
    return df.dropna()

def est_Ngbm_win(win_len,nums,files,period='daily'):
    dfs = pd.concat([est_gbm_win(win_len,file) for file in files],axis=1,join='inner')
    dfs.columns = [name+str(i+1) for i in range(len(files)) for name in ['logrtn','sig','mu']]
    dfs = pd.concat([dfs,build_portN(nums,files)],axis=1)
    if period == 'annual':
        dfs.loc[:,'sig'] = dfs.sig*con.year**0.5
        dfs.loc[:,'mu'] = dfs.mu*con.year
    return dfs.dropna()

def est_Ngbm_exp(lam,nums,files,period='daily'):
    dfs = pd.concat([est_gbm_exp(lam,file) for file in files],axis=1,join='inner')
    dfs.columns = [name+str(i+1) for i in range(len(files)) for name in ['logrtn','sig','mu']]
    dfs = pd.concat([dfs,build_portN(nums,files)],axis=1)
    if period == 'annual':
        dfs.loc[:,'sig'] = dfs.sig*con.year**0.5
        dfs.loc[:,'mu'] = dfs.mu*con.year
    return dfs.dropna()

