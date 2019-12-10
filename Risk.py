# -*- coding: utf-8 -*-
# File 'Risk.py' contains functions to compute risk under different measure and assumptions
from utils import *
from Configuration import *
import scipy.optimize as optimize


def risk_gbm(measure: str, p: Union[int, float], para: pd.DataFrame, T: int = 5) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall under GBM assumption
    with close form formula when we long stock.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu` and `sig`.
        T: future T days to compute risk.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    if measure == 'VaR':
        res = con.s0*(1-np.exp(para.sig*T**0.5*stats.norm.ppf(1-p)+T*para.mu))
    else:
        res = con.s0*(1-np.exp(para.mu*T)/(1-p) *
                      stats.norm.cdf(stats.norm.ppf(1-p) - para.sig*T**0.5))
    return pd.Series(res, index=para.index)


def risk_gbm_mc(measure: str, p: Union[float, int], para: pd.DataFrame, T: int = 5, size: int = 1000000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall under GBM assumption
    with Monte Carlo method when we long stock.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu` and `sig`.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    b = np.random.normal(size=size)
    para['measure'] = measure

    def val(mu: float, sig: float, measure: str) -> float:
        x = np.exp((mu-0.5*sig**2)*T+sig*b*T**0.5)
        if measure == 'VaR':
            return np.nanquantile(x, 1-p)
        else:
            return np.mean(x[x < np.nanquantile(x, 1-p)])
    return con.s0*(1-pd.Series(list(map(val, para['mu'], para['sig'], para['measure'])), index=para.index))


def risk_hist(measure: str, p: Union[float, int], hist_len: Union[float, int], para: pd.Series, T=5) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall with historical data when we long stock.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu` and `sig`.
        T: future T days to compute risk.
        hist_len: the number of historical days that are taken as sample.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    if measure == 'VaR':
        return - para.rolling(window=T, min_periods=T).sum().rolling(window=int(hist_len*con.year), min_periods=int(hist_len)*con.year).quantile(1-p)*con.s0
    else:
        return - para.rolling(window=int(hist_len*con.year), min_periods=int(hist_len*con.year)).apply(lambda x: x[x < x.quantile(1-p)].mean(), raw=False)*con.s0


def risk_hist_short(measure, p: Union[float, int], hist_len, para, T=5) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall with historical data when we short stock.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu` and `sig`.
        T: future T days to compute risk.
        hist_len: the number of historical days that are taken as sample.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    if measure == 'VaR':
        return - para.rolling(window=T, min_periods=T).sum().rolling(window=int(hist_len*con.year), min_periods=int(hist_len*con.year)).quantile(p)*con.s0
    else:
        return - para.rolling(window=int(hist_len*con.year), min_periods=int(hist_len*con.year)).apply(lambda x: x[x > x.quantile(p)].mean(), raw=False)*con.s0


def risk_2gbm_norm(measure, p: Union[float, int], para, T=5) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall assuming to long two stocks follow GBM
    with correlation $\rho$. Then we assume their joint distribution is Gaussian distribution.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu1`, `mu2`, `sig1`, `sig2` and `rho`.
        T: future T days to compute risk.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    ev = para.w*np.exp(para.mu1*T)+(1 - para.w)*np.exp(para.mu2*T)
    sig2 = para.w**2*np.exp((2 * para.mu1 + para.sig1**2)*T)+(1 - para.w)**2*np.exp((2 * para.mu2 + para.sig2**2)*T) + \
        2*(1 - para.w) * para.w * \
        np.exp((para.mu1 + para.mu2 + para.rho * para.sig1 * para.sig2)*T)-ev**2
    if measure == 'VaR':
        return pd.Series(data=con.s0*(1-(ev+stats.norm.ppf(1-p)*sig2**0.5)), index=para.index)
    else:
        def tmp(ev, sig2):
            return con.s0*(1-stats.norm.expect(lambda x: x, loc=ev, scale=sig2**0.5, lb=-np.inf, ub=(ev+stats.norm.ppf(1-p)*sig2**0.5))/(1-p))
        tmp = np.vectorize(tmp)
        return pd.Series(tmp(ev, sig2), index=para.index)


def risk_2gbm_mc(measure, p: Union[float, int], para, T=5, size=1000000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall assuming to long two stocks follow GBM. 
    Then we get their joint distribution through simulation.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu1`, `mu2`, `sig1`, `sig2` and `rho`.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    b1 = np.random.normal(size=size)
    b2 = np.random.normal(size=size)
    para['measure'] = measure

    def val(w: float, mu1: float, sig1: float, mu2: float, sig2: float, measure: str) -> float:
        '''compute risk for each day.'''
        x = w*np.exp((mu1-0.5*sig1**2)*T+sig1*b1*T**0.5)+(1-w) * \
            np.exp((mu2-0.5*sig2**2)*T+sig2*b2*T**0.5)
        if measure == 'VaR':
            return np.nanquantile(x, 1-p)
        else:
            return np.mean(x[x < np.nanquantile(x, 1-p)])
    return con.s0*(1-pd.Series(list(map(val, para['w'], para['mu1'], para['sig1'], para['mu2'], para['sig2'], para['measure'])), index=para.index))


def risk_Ngbm_mc(measure, p: Union[float, int], paras, T=5, size=1000000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall assuming to long N stocks follow GBM. 
    Then we get their joint distribution through simulation.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu1`, `mu2`, `sig1`, `sig2` and so on.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    stocknum = len(paras.filter(regex='mu').columns)
    bs = np.random.normal(scale=T**0.5, size=(stocknum, size))

    def val(measure: str, w, mus, sigs) -> float:
        '''compute risk estimated of each day.'''
        x = 0
        for i in range(stocknum):
            x += np.exp((mus[i]-0.5*sigs[i]**2)*T+sigs[i]*bs[i])*w[i]
        if measure == 'VaR':
            return np.nanquantile(x, 1-p)
        else:
            return np.mean(x[x < np.nanquantile(x, 1-p)])
    # return con.s0*(1-pd.Series(list(map(val,paras['measure'],paras.filter(regex='w').values,paras.filter(regex='mu').values,paras.filter(regex='sig').values)),index=paras.index))
    ws, mus, sigs = paras.filter(regex='w').values, paras.filter(
        regex='mu').values, paras.filter(regex='sig').values
    return con.s0*(1-pd.Series([val(measure, ws[i], mus[i], sigs[i]) for i in range(len(paras))], index=paras.index))


def risk_Ngbm_mc_short(measure, p: Union[float, int], paras, T=5, size=1000000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall assuming to short N stocks following GBM. 
    Then we get their joint distribution through simulation.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu1`, `mu2`, `sig1`, `sig2` and so on.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    stocknum = len(paras.filter(regex='mu').columns)
    bs = np.random.normal(scale=T**0.5, size=(stocknum, size))

    def val(measure: str, w, mus, sigs) -> float:
        x = 0
        for i in range(stocknum):
            x += np.exp((mus[i]-0.5*sigs[i]**2)*T+sigs[i]*bs[i])*w[i]
        if measure == 'VaR':
            return np.nanquantile(x, p)
        else:
            return np.mean(x[x > np.nanquantile(x, p)])
    ws, mus, sigs = paras.filter(regex='w').values, paras.filter(
        regex='mu').values, paras.filter(regex='sig').values
    return con.s0*(1-pd.Series([val(measure, ws[i], mus[i], sigs[i]) for i in range(len(paras))], index=paras.index))


def risk_gbm_short(measure, p: Union[float, int], para, T=5, s0=10000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall under GBM assumption
    with Monte Carlo method when we short stock.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu` and `sig`.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    if measure == 'VaR':
        return con.s0 * (np.exp(para.sig * T ** 0.5 * stats.norm.ppf(p) + T * para.mu) - 1)
    else:
        return con.s0 * (np.exp(para.mu * T) / (1 - p) * stats.norm.cdf(-stats.norm.ppf(p) + para.sig * T ** 0.5) - 1)


def risk_2gbm_mc_short(measure, p: Union[float, int], para, T=5, size=1000000) -> pd.Series:
    '''function to compute Value at Risk and Expected Shortfall assuming to short N stocks following GBM. 
    Then we get their joint distribution through simulation.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        para: estimated parameters containing columns `mu1`, `mu2`, `sig1`, `sig2` and so on.
        T: future T days to compute risk.
        size: the sample size of Monte Carlo method.

    Returns:
        A series of risk estimates.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument '+measure
    b1 = np.random.normal(size=size)
    b2 = np.random.normal(size=size)
    para['measure'] = measure

    def sampling(w, mu1, sig1, mu2, sig2, measure: str) -> float:
        x = w * np.exp((mu1 - 0.5 * sig1 ** 2) * T + sig1 * b1 * T ** 0.5) + \
            (1 - w) * np.exp((mu2 - 0.5 * sig2 ** 2) * T + sig2 * b2 * T ** 0.5)
        if measure == 'VaR':
            res = np.nanquantile(x, p)
        else:
            res = np.mean(x[x > np.nanquantile(x, p)])
        return res
    tmp = list(map(sampling, para['w'], para['mu1'],
                   para['sig1'], para['mu2'], para['sig2'], para['measure']))
    return con.s0*(pd.Series(data=tmp, index=para.index)-1)


def backtest_stock(prices: pd.Series, T: int = 5) -> pd.Series:
    '''function to compute stock price changes in days.

    Args:
        prices: a sequence of stock price with datetime index.
        T: length of period to calculate stock return.

    Returns:
        a series of stock returns.
    '''
    return prices.rolling(window=T + 1, min_periods=T + 1).apply(lambda x: x[-1] / x[0], raw=True) * con.s0


def backtest_opt_stck(stock_prices: pd.Series, opt_prices: pd.Series, ratio: Union[int, float], T: int = 5) -> pd.Series:
    '''function to compute stock price changes in days.

    Args:
        stock_prices: a sequence of stock price with datetime index.
        opt_prices: a sequence of options price with datetime index.
        T: length of period to calculate stock return.

    Returns:
        a series of portfolio returns.
    '''
    prices = stock_prices*(1-ratio)+opt_prices*ratio
    return prices.rolling(window=T + 1, min_periods=T + 1).apply(lambda x: x[-1] / x[0], raw=True) * con.s0


def compare_backtest(theory_risk: pd.Series, reality_risk: pd.Series, cmp_win: Union[float, int]) -> pd.Series:
    '''function to compute the number of days when the loss exceeds risk estimated in a specific length of period.

    Args:
        theory_risk: a series of estimated risk with datetime index.
        reality_risk: a series of real return with datetime index.
        cmp_win: the number of years to calculate the times when losses exceed estimated risk.

    Returns:
        a series of numbers indicating the times that loss exceeds estimated risk from current datetime
    '''
    diff = (theory_risk - reality_risk).dropna()
    except_nums = diff.rolling(window=con.year*cmp_win, min_periods=con.year *
                               cmp_win).apply(lambda x: (x > 0).sum(), raw=True)
    return except_nums.dropna()


def risk_opt_stck(measure, p: Union[float, int], opt_type: str, ratio: Union[float, int], stock,
                  risk_free: pd.Series, options_vol: pd.Series, seed=None, T=5, size=1000000) -> pd.Series:
    '''function to compute the risk of portfolio of one stock and one option.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        opt_type: `put` or `call`.
        ratio: the proportion of initial capital that is used to buy options.
        stock: estimated `mu` and `sig` of stock.
        risk_free: annual Treasury Note of US bond of each day.
        options_vol: the implied volatility of the ATM option with one-year maturity.
        seed: the seed for Monte Carlo.
        T: the number of days to estimate risk.
        size: the sample size of Monte Carlo method.

    Returns:
        a series of estimated risk with datetime index.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument ' + measure
    assert opt_type in ['call', 'put'], 'Wrong argument' + opt_type
    risk_free.name, options_vol.name = 'r', 'vol'
    df = pd.concat([stock, options_vol, risk_free], axis=1, join='inner')
    ops_num = con.s0*ratio / \
        bs_option(opt_type, df['r'], df['vol'], 1, con.s0, con.s0)
    ops_price = bs_option(
        opt_type, df['r'], df['vol'], 1-T/con.year, con.s0, con.s0)
    mus, sigs = df['mu'].values, df['sig'].values

    def sample(stock_mu: float, stock_sig: float, opt_num: float, opt_price) -> float:
        '''compute daily estimated risk.'''
        if seed is None:
            np.random.seed(233)
        st = np.exp((stock_mu - stock_sig**2/2)*T + stock_sig*T**0.5 *
                    np.random.normal(size=size))*con.s0
        sample = con.s0-opt_num*opt_price-st*(1-ratio)
        if measure == 'VaR':
            return np.nanquantile(sample, p)
        else:
            return np.mean(sample[sample > np.nanquantile(sample, p)])

    return pd.Series([sample(mus[i], sigs[i], ops_num[i], ops_price[i]) for i in range(len(df))], index=df.index)


def hedge_opt_stck(measure: str, p: Union[int, float], opt_type: str, proportion: Union[int, float],
                   stock: pd.DataFrame, risk_free: pd.Series, options_vol: pd.Series, init: Union[int, float],
                   seed=None, T: int = 5, size: int = 1000000) -> pd.Series:
    '''function to compute the risk of portfolio of one stock and one option.

    Args:
        measure: `VaR` or `ES`.
        p: percentile of VaR or ES.
        opt_type: `put` or `call`.
        proportion: the expected proportion of original risk without options.
        stock: estimated `mu` and `sig` of stock.
        risk_free: annual Treasury Note of US bond of each day.
        options_vol: the implied volatility of the ATM option with one-year maturity.
        seed: the seed for Monte Carlo.
        init: initial ratio to solve proper proportion.
        T: the number of days to estimate risk.
        size: the sample size of Monte Carlo method.

    Returns:
        a series of estimated risk with datetime index.
    '''
    assert measure in ['ES', 'VaR'], 'Wrong argument ' + measure
    assert opt_type in ['call', 'put'], 'Wrong argument' + opt_type
    risk_free.name, options_vol.name = 'r', 'vol'
    df = pd.concat([stock, options_vol, risk_free], axis=1, join='inner')
    ops_price1 = bs_option(
        opt_type, df['r'], df['vol'], 1, con.s0, con.s0)
    ops_price2 = bs_option(
        opt_type, df['r'], df['vol'], 1-T/con.year, con.s0, con.s0)
    mus, sigs = df['mu'].values, df['sig'].values

    def sample(stock_mu, stock_sig, opt_price1, opt_price2, ratio):
        if seed is not None:
            np.random.seed(233)
        st = np.exp((stock_mu - stock_sig**2/2)*T + stock_sig*T**0.5 *
                    np.random.normal(size=size))*con.s0
        sample = con.s0 - con.s0 / opt_price1 * opt_price2 - st * (1 - ratio)
        if measure == 'VaR':
            return np.nanquantile(sample, p)
        else:
            return np.mean(sample[sample > np.nanquantile(sample, p)])
    prev = [sample(mus[i], sigs[i], ops_price1[i], ops_price2[i], 0)
            for i in range(len(df))]
    # return optimize.fsolve(lambda x: risk_opt_stck(measure,p,option,x,para,seed=None,T=5,size=1000000)-prev*proportion,init,epsfcn=1e-4,xtol=1e-2)
    return pd.Series([optimize.fsolve(lambda x: sample(mus[i], sigs[i], ops_price1[i], ops_price2[i], x)-prev[i]*proportion, init, epsfcn=1e-4)[0] for i in range(len(df))], index=df.index)
