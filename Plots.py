# -*- coding: utf-8 -*-
# File 'Plots.py' provides a few ways to plot graphs
from Configuration import *
import matplotlib.pyplot as plt
def mono_fig_plot(data,figsize,title,xlabel=None,ylabel=None,legends=None,savepath=None,start_date=None,dpi=300):
    df = pd.concat(data,axis=1,join='inner')
    if start_date is not None:
        df = df.loc[df.index>=start_date]
    if legends is None:
        fig = df.plot(figsize=figsize,title=title,legend=False)
    elif legends == True:
        fig = df.plot(figsize=figsize,title=title)
    else:
        df.columns = legends
        fig = df.plot(figsize=figsize,title=title)
    fig.set(xlabel=xlabel,ylabel=ylabel)
    if savepath is not None:
        fig.get_figure().savefig(savepath,bbox_inches='tight',dpi=dpi)

def multi_fig_plot(data,titles,xlabels=None,ylabels=None,width=2,savepath=None,dpi=300):
    fig = plt.figure()
    for i in range(len(data)):
        x = int(np.ceil(len(data)/width))
        axs = fig.subplots(x,i-(x-1)*width)
        data[i].plot(ax = axs)
        axs.set_xlabel(xlabels[i])
        axs.set_ylabel(ylabels[i])
        axs.set_title(titles[i])
    if savepath is not None:
        plt.savefig(savepath,bbox_inches='tight',dpi=dpi)

