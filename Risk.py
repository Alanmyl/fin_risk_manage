# -*- coding: utf-8 -*-
# File 'Risk.py' contains functions to compute risk under different measure and assumptions
from utils import *
from Configuration import *
import scipy.optimize as optimize


def risk_gbm(measure, p, para, T=5):
    if measure == 'VaR':
        res = con.s0*(1-np.exp(para.sig*T**0.5*stats.norm.ppf(1-p)+T*para.mu))
    elif measure == 'ES':
        res = con.s0*(1-np.exp(para.mu*T)/(1-p) *
                      stats.norm.cdf(stats.norm.ppf(1-p) - para.sig*T**0.5))
    else:
        return None
    return pd.Series(res, index=para.index)


def risk_gbm_mc(measure, p, para, T=5, size=1000000):
    percentile = np.sort(np.random.normal(size=size))[int(size*(1-p))]
    if measure == 'VaR':
        return con.s0*(1-np.exp(para.sig*T**0.5*percentile+T * para.mu))
    elif measure == 'ES':
        return con.s0*(1-np.exp(para.mu*T)/(1-p)*stats.norm.cdf(percentile - para.sig*T**0.5))
    else:
        return None


def risk_hist(measure, p, hist_len, para, T=5):
    if measure == 'VaR':
        return - para.rolling(window=T, min_periods=T).sum().rolling(window=hist_len*con.year, min_periods=hist_len*con.year).quantile(1-p)*con.s0
    elif measure == 'ES':
        return - para.rolling(window=hist_len*con.year, min_periods=hist_len*con.year).apply(lambda x: x[x < x.quantile(1-p)].mean(), raw=False)*con.s0
    else:
        return None


def risk_hist_mc(measure, p, hist_len, para, T=5):
    if measure == 'VaR':
        return - para.rolling(window=T, min_periods=T).sum().rolling(window=hist_len*con.year, min_periods=hist_len*con.year).quantile(p)*con.s0
    elif measure == 'ES':
        return - para.rolling(window=hist_len*con.year, min_periods=hist_len*con.year).apply(lambda x: x[x > x.quantile(p)].mean(), raw=False)*con.s0
    else:
        return None


def risk_2gbm_norm(measure, p, para, T=5):
    ev = para.w*np.exp(para.mu1*T)+(1 - para.w)*np.exp(para.mu2*T)
    sig2 = para.w**2*np.exp((2 * para.mu1 + para.sig1**2)*T)+(1 - para.w)**2*np.exp((2 * para.mu2 + para.sig2**2)*T) + \
        2*(1 - para.w) * para.w * \
        np.exp((para.mu1 + para.mu2 + para.rho * para.sig1 * para.sig2)*T)-ev**2
    if measure == 'VaR':
        return pd.Series(data=con.s0*(1-(ev+stats.norm.ppf(1-p)*sig2**0.5)), index=para.index)
    elif measure == 'ES':
        def tmp(ev, sig2):
            return con.s0*(1-stats.norm.expect(lambda x: x, loc=ev, scale=sig2**0.5, lb=-np.inf, ub=(ev+stats.norm.ppf(1-p)*sig2**0.5))/(1-p))
        tmp = np.vectorize(tmp)
        return pd.Series(tmp(ev, sig2), index=para.index)
    else:
        return None


def risk_2gbm_mc(measure, p, para, T=5, size=1000000):
    b1 = np.random.normal(size=size)
    b2 = np.random.normal(size=size)
    para['measure'] = measure

    def val(w, mu1, sig1, mu2, sig2, measure):
        x = w*np.exp((mu1-0.5*sig1**2)*T+sig1*b1*T**0.5)+(1-w) * \
            np.exp((mu2-0.5*sig2**2)*T+sig2*b2*T**0.5)
        if measure == 'VaR':
            return np.nanquantile(x, 1-p)
        else:
            return np.mean(x[x < np.nanquantile(x, 1-p)])
    return con.s0*(1-pd.Series(list(map(val, para['w'], para['mu1'], para['sig1'], para['mu2'], para['sig2'], para['measure'])), index=index))


def risk_Ngbm_mc(measure, p, paras, T=5, size=1000000):
    stocknum = len(paras.filter(regex='mu').columns)
    bs = np.random.normal(scale=T**0.5, size=(stocknum, size))

    def val(measure, w, mus, sigs):
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


def risk_Ngbm_mc_short(measure, p, paras, T=5, size=1000000):
    stocknum = len(paras.filter(regex='mu').columns)
    bs = np.random.normal(scale=T**0.5, size=(stocknum, size))

    def val(measure, w, mus, sigs):
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


def risk_gbm_short(measure, p, para, T=5, s0=10000):
    if measure == 'VaR':
        return con.s0*(np.exp(para.sig*T**0.5*stats.norm.ppf(p)+T * para.mu)-1)
    elif measure == 'ES':
        return con.s0*(np.exp(para.mu*T)/(1-p)*stats.norm.cdf(-stats.norm.ppf(p) + para.sig*T**0.5)-1)
    else:
        return None


def risk_2gbm_mc_short(measure, p, para, T=5, size=1000000):
    b1 = np.random.normal(size=size)
    b2 = np.random.normal(size=size)
    para['measure'] = measure

    def sampling(w, mu1, sig1, mu2, sig2, measure):
        x = w*np.exp((mu1-0.5*sig1**2)*T+sig1*b1*T**0.5)+(1-w) * \
            np.exp((mu2-0.5*sig2**2)*T+sig2*b2*T**0.5)
        if measure == 'VaR':
            res = np.nanquantile(x, p)
        else:
            res = np.mean(x[x > np.nanquantile(x, p)])
        return res
    tmp = list(map(sampling, para['w'], para['mu1'],
                   para['sig1'], para['mu2'], para['sig2'], para['measure']))
    return con.s0*(pd.Series(data=tmp, index=para.index)-1)


def backtest_stock(pos, prices, T=5):
    assert pos in ['long', 'short'], 'Wrong argument ' + pos
    if pos == 'long':
        return prices.adj_close.rolling(window=T+1, min_periods=T+1).apply(
            lambda x: abs(np.min(x[1:] - x[0])), raw=True)*con.s0 / prices.adj_close
    else:
        return prices.adj_close.rolling(window=T+1, min_periods=T+1).apply(
            lambda x: abs(np.max(x[1:] - x[0])), raw=True)*con.s0 / prices.adj_close


def compare_backtest(theory_risk, reality_risk, cmp_win):
    diff = (theory_risk - reality_risk).dropna()
    except_nums = diff.rolling(window=con.year*cmp_win, min_periods=con.year *
                               cmp_win).apply(lambda x: (x > 0).sum(), raw=True)
    return except_nums.dropna()


def risk_opt_stck(measure, p, opt_type, ratio, stock, risk_free, options_vol, seed=None, T=5, size=1000000):
    risk_free.name, options_vol.name = 'r', 'vol'
    df = pd.concat([stock, options_vol, risk_free], axis=1, join='inner')
    ops_num = con.s0*ratio / \
        bs_option(opt_type, df['r'], df['vol'], 1, con.s0, con.s0)
    ops_price = bs_option(
        opt_type, df['r'], df['vol'], 1-T/con.year, con.s0, con.s0)
    mus, sigs = df['mu'].values, df['sig'].values

    def sample(stock_mu, stock_sig, opt_num, opt_price):
        if seed is not None:
            np.random.seed(233)
        st = np.exp((stock_mu - stock_sig**2/2)*T + stock_sig*T**0.5 *
                    np.random.normal(size=size))*con.s0
        sample = con.s0-opt_num*opt_price-st*(1-ratio)
        if measure == 'VaR':
            return np.nanquantile(sample, p)
        else:
            return np.mean(sample[sample > np.nanquantile(sample, p)])

    return pd.Series([sample(mus[i], sigs[i], ops_num[i], ops_price[i]) for i in range(len(df))], index=df.index)


def hedge_opt_stck(measure, p, opt_type, proportion, stock, risk_free, options_vol, init, seed=None, T=5, size=1000000):
    risk_free.name, options_vol.name = 'r', 'vol'
    df = pd.concat([stock, options_vol, risk_free], axis=1, join='inner')
    ops_price1 = bs_option(
        opt_type, df['r'], df['vol'], con.year, con.s0, con.s0)
    ops_price2 = bs_option(
        opt_type, df['r'], df['vol'], con.year-T, con.s0, con.s0)
    mus, sigs = df['mu'].values, df['sig'].values

    def sample(stock_mu, stock_sig, opt_price1, opt_price2, ratio):
        if seed is not None:
            np.random.seed(233)
        st = np.exp((stock_mu - stock_sig**2/2)*T + stock_sig*T**0.5 *
                    np.random.normal(size=size))*con.s0
        sample = con.s0-con.s0/opt_price1*opt_price2-st*(1-ratio)
        if measure == 'VaR':
            return np.nanquantile(sample, p)
        else:
            return np.mean(sample[sample > np.nanquantile(sample, p)])
    prev = [sample(mus[i], sigs[i], ops_price1[i], ops_price2[i], 0)
            for i in range(len(df))]
    # return optimize.fsolve(lambda x: risk_opt_stck(measure,p,option,x,para,seed=None,T=5,size=1000000)-prev*proportion,init,epsfcn=1e-4,xtol=1e-2)
    return pd.Series([optimize.fsolve(lambda x: sample(mus[i], sigs[i], ops_price1[i], ops_price2[i], x)-prev[i]*proportion, init, epsfcn=1e-4)[0] for i in range(len(df))], index=df.index)
