# -*- coding: utf-8 -*-
# File 'utils.py' contains functions to finish the basic computation
from Configuration import *
def pre_data(path,header=0):
    file = pd.read_csv(path,header=header,index_col=0).dropna()
    file.index = file.index.map(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))
    file = file.rename(columns = {'Adj Close':'adj_close'})
    return file.sort_index()

def logrtn(file):
    return pd.Series(np.log(file.adj_close.iloc[1:].values/file.adj_close.iloc[:-1].values),index=file.index[1:])

def build_port2(num1,num2,file1,file2):
    res = (file1.adj_close*num1+file2.adj_close*num2).to_frame(name='adj_close')
    res['w'] = file1.adj_close*num1/res.adj_close
    return res.dropna()

def build_portN(nums,files):
    summ = np.sum([files[i].adj_close*nums[i] for i in range(len(files))],axis=0)
    res = pd.DataFrame()
    for i in range(len(files)):
        res['w'+str(i+1)] = files[i].adj_close*nums[i]/summ
    return res

def bs_put(r,sig,T,K,s0):
    r=r/con.year
    d1 = (np.log(s0/K)+(r+sig**2/2)*T)/sig/T**0.5
    d2 = d1-sig*T**0.5
    return stats.norm.cdf(-d2)*K*np.exp(-r*T)-stats.norm.cdf(-d1)*s0

def bs_call(r,sig,T,K,s0):
    r=r/con.year
    d1 = (np.log(s0/K)+(r+sig**2/2)*T)/sig/T**0.5
    d2 = d1-sig*T**0.5
    return stats.norm.cdf(d1)*s0-stats.norm.cdf(d2)*K*np.exp(-r*T)