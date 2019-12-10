# -*- coding: utf-8 -*-
# File 'Plots.py' provides a few ways to plot graphs
from Configuration import *
import matplotlib.pyplot as plt


def mono_fig_plot(data, figsize: tuple, title: str, xlabel: str = None, ylabel: str = None,
                  legends=None, savepath: str = None, start_date: dt.datetime = None, dpi: int = 300):
    '''function to plot multiple curves in the same figure.

    Args:
        data: a sequence of Series with datetime index.
        title: the title of the figure.
        figsize: the size of the figure, like (x,y).
        xlabel, ylabel: labels of axes of the figure.
        legends: `True` for the original column names, `None` for no legends or a sequence of 
        names to replace original column names.
        savepath: `None` for don't save or specify path to save.
        start_date: only the data after it will be drawn.
        dpi: large dpi leads to larger file size and clearer figure.

    Returns:
        nothing to return.
    '''
    df = pd.concat(data, axis=1, join='inner')
    if start_date is not None:
        df = df.loc[df.index >= start_date]
    if legends is None:
        fig = df.plot(figsize=figsize, title=title, legend=False)
    elif legends == True:
        fig = df.plot(figsize=figsize, title=title)
    else:
        df.columns = legends
        fig = df.plot(figsize=figsize, title=title)
    fig.set(xlabel=xlabel, ylabel=ylabel)
    if savepath is not None:
        fig.get_figure().savefig(savepath, bbox_inches='tight', dpi=dpi)


def multi_fig_plot(data, titles, figsize: tuple, xlabels: str = None, ylabels: str = None,
                   width=2, savepath: str = None, dpi: int = 300):
    '''function to plot curves in the different figure and output one figure.

    Args:
        data: a sequence of Series with datetime index.
        titles: the titles of the figure for each subfigure.
        figsize: the size of the figure, like (x,y).
        xlabel, ylabel: labels of axes of the figure.
        width: the number of subfigures in each line of figure.
        savepath: `None` for don't save or specify path to save.
        dpi: large dpi leads to larger file size and clearer figure.

    Returns:
        nothing to return.
    '''
    fig = plt.figure(figsize=figsize)
    for i in range(len(data)):
        x = int(np.ceil(len(data)/width))
        axs = fig.subplots(x, i-(x-1)*width)
        data[i].plot(ax=axs)
        axs.set_xlabel(xlabels[i])
        axs.set_ylabel(ylabels[i])
        axs.set_title(titles[i])
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
