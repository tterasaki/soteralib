import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def get_7wafer_axes_h(figsize):
    fig = plt.figure(figsize=figsize)
    gs_master = GridSpec(nrows=3, ncols=6, height_ratios=[1, 1, 1])

    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, 1:3])
    axes_1 = fig.add_subplot(gs_1[:, :])

    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, 3:5])
    axes_2 = fig.add_subplot(gs_2[:, :])

    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1, 0:2])
    axes_3 = fig.add_subplot(gs_3[:, :])

    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1, 2:4])
    axes_4 = fig.add_subplot(gs_4[:, :])

    gs_5 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1, 4:6])
    axes_5 = fig.add_subplot(gs_5[:, :])

    gs_6 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[2, 1:3])
    axes_6 = fig.add_subplot(gs_6[:, :])

    gs_7 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[2, 3:5])
    axes_7 = fig.add_subplot(gs_7[:, :])
    
    ax = [axes_1, axes_2, axes_3, axes_4, axes_5, axes_6, axes_7, ]
    return fig, ax

def get_7wafer_axes_v(figsize):
    fig = plt.figure(figsize=figsize)
    gs_master = GridSpec(nrows=6, ncols=3, height_ratios=[1, 1, 1, 1, 1, 1,])

    gs_1 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[1:3, 0])
    axes_1 = fig.add_subplot(gs_1[:, :])

    gs_2 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[3:5, 0])
    axes_2 = fig.add_subplot(gs_2[:, :])

    gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[0:2, 1])
    axes_3 = fig.add_subplot(gs_3[:, :])

    gs_4 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[2:4, 1])
    axes_4 = fig.add_subplot(gs_4[:, :])

    gs_5 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[4:6, 1])
    axes_5 = fig.add_subplot(gs_5[:, :])

    gs_6 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[1:3, 2])
    axes_6 = fig.add_subplot(gs_6[:, :])

    gs_7 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[3:5, 2])
    axes_7 = fig.add_subplot(gs_7[:, :])
    
    ax = [axes_1, axes_2, axes_3, axes_4, axes_5, axes_6, axes_7, ]
    return fig, ax