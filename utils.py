# -*- coding: utf-8 -*-
# File 'utils.py' contains functions to finish the basic computation
from Configuration import *


def pre_data(path: str, header: int = 0, colname: str = 'Adj Close') -> pd.DataFrame:
    '''function to read financial data with dates in the first column with type `YYYY-mm-dd` or `mm/dd/YYYY`.

    Args:
        path: the absolute or the relative path of the file to read.
        header: the line which will be taken as column names.
        colname: name of the column where the price is. 

    Returns:
        A dataframe with ascending dates indices without `NA`.
    '''
    data = pd.read_csv(path, header=header, index_col=0).dropna()
    try:
        data.index = data.index.map(
            lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    except ValueError:
        data.index = data.index.map(
            lambda x: dt.datetime.strptime(x, '%m/%d/%Y'))
    data = data.rename(columns={colname: 'adj_close'})
    return data.sort_index()


def logrtn(data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    '''function to calculate log returns from the asset price.

    Args:
        data: a dataframe containing column name `adj_close` as price or a series of price.

    Returns:
        A Series of log returns with almost the same index as `data`.
    '''
    if type(data) == pd.DataFrame:
        return pd.Series(np.log(data.adj_close.iloc[1:].values / data.adj_close.iloc[:-1].values), index=data.index[1:])
    else:
        return pd.Series(np.log(data.iloc[1:].values / data.iloc[:-1].values), index=data.index[1:])


def build_port2(num1, num2, file1, file2):
    res = (file1.adj_close*num1+file2.adj_close *
           num2).to_frame(name='adj_close')
    res['w'] = file1.adj_close*num1/res.adj_close
    return res.dropna()


def build_portN(nums, files):
    summ = np.sum([files[i].adj_close*nums[i]
                   for i in range(len(files))], axis=0)
    res = pd.DataFrame()
    for i in range(len(files)):
        res['w'+str(i+1)] = files[i].adj_close*nums[i]/summ
    return res


def bs_option(opt_type, r, sig, T, K, s0):
    d1 = (np.log(s0/K)+(r+sig**2/2)*T)/sig/T**0.5
    d2 = d1-sig*T**0.5
    if opt_type == 'put':
        return stats.norm.cdf(-d2)*K*np.exp(-r*T)-stats.norm.cdf(-d1)*s0
    else:
        return stats.norm.cdf(d1)*s0-stats.norm.cdf(d2)*K*np.exp(-r*T)
