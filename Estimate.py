# -*- coding: utf-8 -*-
# File 'Estimate.py' contains functions to estimate parameters with different methods.
from Configuration import *
from utils import build_port2, logrtn, build_portN


def est_gbm_win(win_len: Union[int, float], prices: Union[pd.Series, pd.DataFrame], period: str = 'daily') -> pd.DataFrame:
    '''Estimate the parameters required in a single stock GBM model by averaging historical data in a rolling fixed-length window.

    Args:
        win_len: number of years of the window length for smoothing.
        prices: the adjusted price of the financial assets.
        period: the default is the estimates of daily data, but user can also specify the period.

    Returns:
        A DataFrame including necessary parameter columns with datetime as index and columns `logrtn`, `mu` and `sig`,
        where 'mu' and 'sig' are estimated values of GBM model.
    '''
    assert period in ['daily', 'annual'], 'Wrong argument of period '+period
    df = logrtn(prices).to_frame(name='logrtn')
    df['sig'] = df.logrtn.rolling(window=win_len*con.year).std()
    df['mu'] = df.logrtn.rolling(
        window=win_len*con.year).mean()+df.logrtn.rolling(window=win_len*con.year).var()/2
    df.index = prices.index[1:]
    if period == 'annual':
        df.loc[:, 'sig'] = df.sig*con.year**0.5
        df.loc[:, 'mu'] = df.mu*con.year
    return df.dropna()


def est_gbm_exp(lam: Union[int, float], prices: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
    '''Estimate the parameters required in a single stock GBM model by giving exponential weights to the historical data.

    Args:
        lam: the weight of past data, larger lambda leads to slower decay.
        prices: the adjusted price of the financial assets.
        period: the default is the estimates of daily data, but user can also specify the period.

    Returns:
        A DataFrame including necessary parameter columns with datetime as index and columns `logrtn`, `mu` and `sig`,
        where 'mu' and 'sig' are estimated values of GBM model.
    '''
    assert period in ['daily', 'annual'], 'Wrong argument ' + period
    assert (lam >= 0) and (lam <= 1), 'Wrong argument '+lam
    df = logrtn(prices).to_frame(name='logrtn')
    df['sig'] = df['logrtn'].ewm(alpha=1-lam).std()
    df['mu'] = df['logrtn'].ewm(alpha=1-lam).mean() + \
        df['logrtn'].ewm(alpha=1-lam).var()/2
    df.index = prices.index[1:]
    if period == 'annual':
        df.loc[:, 'sig'] = df.sig*con.year**0.5
        df.loc[:, 'mu'] = df.mu*con.year
    return df.dropna()


def est_2gbm_win(win_len: Union[int, float], num1, num2, file1: pd.Series, file2: pd.Series,
                 period: str = 'daily', rho: bool = True) -> pd.DataFrame:
    '''Estimate `mu`s and `sig`s required in two stocks GBM model by averaging historical data in a 
    rolling fixed-length window.

    Args:
        win_len: number of years of the window length for smoothing.
        num1, num2: see function `build_port2`.
        file1, file2: the adjusted price of the financial assets.
        period: the default is the estimates of daily data, but user can also specify the period.
        rho: whether to generate $\rho$ for the assumption that two stocks follow GBM and their
         correlation of Brownian motion is $\rho$. Required by `risk_gbm_norm`.

    Returns:
        A DataFrame with datetime as index and columns `logrtn`, `mu` and `sig`, where 'mu' and 'sig' are estimated values of GBM model.
    '''
    assert period in ['daily', 'annual'], 'Wrong argument '+period
    df = pd.concat([est_gbm_win(win_len, file1), est_gbm_win(
        win_len, file2)], axis=1, join='inner')
    df.columns = ['logrtn1', 'sig1', 'mu1', 'logrtn2', 'sig2', 'mu2']
    if rho == True:
        df['rho'] = ((df.logrtn1-df.logrtn1.mean())*(df.logrtn2-df.logrtn2.mean()
                                                     )/df.sig1/df.sig2).rolling(window=5*con.year).mean()
    df['w'] = build_port2(num1, num2, file1, file2)['w'].loc[df.index]
    if period == 'annual':
        df.loc[:, 'sig'] = df.sig*con.year**0.5
        df.loc[:, 'mu'] = df.mu*con.year
    return df.dropna()


def est_Ngbm_win(win_len: Union[int, float], nums, files, period='daily') -> pd.DataFrame:
    '''function to estimate `mu`s and `sig`s in N stocks of GBM model by averaging historical data in a 
    rolling fixed-length window.

    Args:
        win_len: number of years of the window length for smoothing.
        nums: see function `build_portN`.
        files: see function `build_portN`.
        period: the default is the estimates of daily data, but user can also specify the period as annual.

    Returns:
        A DataFrame with datetime as index and columns `w1`, `logrtn1`, `mu1`, `sig1` and so on, where 'mu1' and 'sig1' are estimated values of GBM model.
    '''
    assert len(files) == len(
        files), "Unconsistent lengths of arg files and nums"
    dfs = pd.concat([est_gbm_win(win_len, x)
                     for x in files], axis=1, join='inner')
    dfs.columns = [name+str(i+1) for i in range(len(files))
                   for name in ['logrtn', 'sig', 'mu']]
    dfs = pd.concat([dfs, build_portN(nums, files)], axis=1)
    if period == 'annual':
        dfs.loc[:, 'sig'] = dfs.sig*con.year**0.5
        dfs.loc[:, 'mu'] = dfs.mu*con.year
    return dfs.dropna()


def est_Ngbm_exp(lam: Union[int, float], nums, files, period='daily') -> pd.DataFrame:
    '''function to estimate `mu`s and `sig`s in N stocks of GBM model by giving exponential weights to the historical data.

    Args:
        lam: the weight of past data, larger lambda leads to slower decay.
        nums: see function `build_portN`.
        files: see function `build_portN`.
        period: the default is the estimates of daily data, but user can also specify the period as annual.

    Returns:
        A DataFrame with datetime as index and columns `w1`, `logrtn1`, `mu1`, `sig1` and so on, where 'mu1' and 'sig1' are estimated values of GBM model.
    '''
    dfs = pd.concat([est_gbm_exp(lam, x)
                     for x in files], axis=1, join='inner')
    dfs.columns = [name+str(i+1) for i in range(len(files))
                   for name in ['logrtn', 'sig', 'mu']]
    dfs = pd.concat([dfs, build_portN(nums, files)], axis=1)
    if period == 'annual':
        dfs.loc[:, 'sig'] = dfs.sig*con.year**0.5
        dfs.loc[:, 'mu'] = dfs.mu*con.year
    return dfs.dropna()
