import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator


def clean_plot(fig, axes, tight_layout):
    if hasattr(axes, 'flat'):
        ax_list = axes.flat
    elif hasattr(axes, 'values'):
        ax_list = list(axes.values())
    elif isinstance(axes, list):
        ax_list = axes
    else:
        ax_list = [axes]

    for axi in ax_list:
        xlim = axi.get_xlim()
        ylim = axi.get_ylim()

        axi.set_xlabel(axi.get_xlabel().replace("_", " "))
        axi.set_ylabel(axi.get_ylabel().replace("_", " "))
        axi.set_title(axi.get_title().replace("_", " "))

        axi.spines['left'].set_bounds(ylim)
        axi.spines['bottom'].set_bounds(xlim)

        sns.despine(ax=axi, offset=5)
    # sns.despine(fig=fig, offset=5)

    if tight_layout:
        fig.tight_layout()

    fig.align_labels()


def get_limits_from_data(data, balanced=True):
    mins = []
    maxs = []
    for d in data:
        mins.append(np.nanmin(np.nanmin(d)))
        maxs.append(np.nanmax(np.nanmax(d)))

    if balanced:  # if min and max are positive/negative of same value (i.e. -10, 10)
        lim_abs = np.nanmax([abs(x) for x in [np.nanmin(mins), np.nanmax(maxs)]])
        limits = [-lim_abs, lim_abs]
    else:
        limits = [np.nanmin(mins), np.nanmax(maxs)]

    return limits


def get_color_theme():
    # found perceptually uniform brightness colors with: https://www.hsluv.org
    color_theme_dict = dict()
    color_theme_dict['CA1'] = '#4897D8'
    color_theme_dict['PFC'] = '#F77669'

    color_theme_dict['Narrow Interneuron'] = '#2594f6'
    color_theme_dict['Pyramidal Cell'] = '#da3b46'
    color_theme_dict['Wide Interneuron'] = '#20a0a8'

    color_theme_dict['control'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['nan'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['home'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['left'] = '#2594f6'  # 250 in degrees, 95 saturation, 60 light
    color_theme_dict['right'] = '#da3b46'  # 12 in degrees, 95 saturation, 60 light
    color_theme_dict['stay'] = '#1ea477'  # 152 in degrees, 95 saturation, 60 light
    color_theme_dict['switch'] = '#927ff9'  # 270 in degrees, 95 saturation, 60 light
    color_theme_dict['stay_update'] = color_theme_dict['stay']
    color_theme_dict['switch_update'] = color_theme_dict['switch']
    color_theme_dict['initial_stay'] = color_theme_dict['stay']
    color_theme_dict['non_update'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['error'] = '#a150db'  # 285 degrees, 75 saturation, 50 light
    color_theme_dict['all'] = '#f26b49'  # 12 in degrees, 0 saturation, 30 light

    color_theme_dict['cmap'] = sns.color_palette("rocket_r", as_cmap=True)
    color_theme_dict['cmap_r'] = sns.color_palette("rocket", as_cmap=True)
    color_theme_dict['all_cmap'] = sns.color_palette("rocket_r", as_cmap=True)
    color_theme_dict['div_cmap'] = sns.diverging_palette(240, 10, s=75, l=60, n=5, center='light', as_cmap=True)
    color_theme_dict['plain_cmap'] = sns.color_palette("Greys_r", as_cmap=True)
    color_theme_dict['home_cmap'] = sns.color_palette("Greys", as_cmap=True)
    color_theme_dict['switch_cmap'] = sns.light_palette(color_theme_dict['switch'], as_cmap=True)
    color_theme_dict['stay_cmap'] = sns.light_palette(color_theme_dict['stay'], as_cmap=True)
    color_theme_dict['initial_stay_cmap'] = color_theme_dict['stay_cmap']
    color_theme_dict['left_cmap'] = sns.light_palette(color_theme_dict['left'], as_cmap=True)
    color_theme_dict['right_cmap'] = sns.light_palette(color_theme_dict['right'], as_cmap=True)
    color_theme_dict['switch_stay_cmap_div'] = sns.diverging_palette(270, 152, s=95, l=60, as_cmap=True)
    color_theme_dict['left_right_cmap_div'] = sns.diverging_palette(250, 12, s=95, l=60, as_cmap=True)

    color_theme_dict['animals'] = sns.color_palette("husl", 7)
    color_theme_dict['general'] = sns.color_palette("husl", 10)
    color_theme_dict['trials'] = [color_theme_dict['non_update'], color_theme_dict['switch'], color_theme_dict['stay']]

    return color_theme_dict


def plot_distributions(data, axes, column_name, group, row_ids, col_ids, xlabel, title, stripplot=True, show_median=True,
                       palette=None, histstat='proportion',):
    if group:
        palette = palette or sns.color_palette(n_colors=len(data[group].unique()))
        if len(palette) > len(data[group].unique()):
            palette = palette[:len(data[group].unique())]
    else:
        palette = palette or sns.color_palette()

    if group and show_median:
        medians = data.groupby([group])[column_name].median()
        limits = [np.nanmin(data.groupby([group])[column_name].min().values),
                  np.nanmax(data.groupby([group])[column_name].max().values)]
    else:
        medians = {column_name: data[column_name].median()}
        limits = [data[column_name].min(), data[column_name].max()]

    # cum fraction plots
    axes[row_ids[0]][col_ids[0]] = sns.ecdfplot(data=data, x=column_name, hue=group, ax=axes[row_ids[0]][col_ids[0]],
                                                palette=palette)
    axes[row_ids[0]][col_ids[0]].set_title(title)
    axes[row_ids[0]][col_ids[0]].set(xlabel=xlabel, ylabel='Proportion', xlim=limits)
    axes[row_ids[0]][col_ids[0]].set_aspect(1. / axes[row_ids[0]][col_ids[0]].get_data_ratio(), adjustable='box')

    # add median annotations to the first plot
    new_line = '\n'
    median_text = [f"{g} median: {m:.2f} {new_line}" for g, m in medians.items()]
    axes[row_ids[0]][col_ids[0]].text(0.55, 0.2, ''.join(median_text)[:-1],
                                      transform=axes[row_ids[0]][col_ids[0]].transAxes, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # histograms
    axes[row_ids[1]][col_ids[1]] = sns.histplot(data=data, x=column_name, hue=group, ax=axes[row_ids[1]][col_ids[1]],
                                                element='step', palette=palette, stat=histstat, common_norm=False)
    axes[row_ids[1]][col_ids[1]].set(xlabel=xlabel, ylabel='Proportion', xlim=limits)

    # violin plots
    axes[row_ids[2]][col_ids[2]] = sns.violinplot(data=data, x=group, y=column_name, ax=axes[row_ids[2]][col_ids[2]],
                                                  palette=palette)
    plt.setp(axes[row_ids[2]][col_ids[2]].collections, alpha=.25)
    if stripplot:
        sns.stripplot(data=data, y=column_name, x=group, size=3, jitter=True, ax=axes[row_ids[2]][col_ids[2]], palette=palette)
    axes[row_ids[2]][col_ids[2]].set_title(title)


def plot_scatter_with_distributions(data, x, y, hue, kind='scatter',fig=None, title=None,
                                    plt_kwargs={'alpha': 0.5, 'palette': sns.color_palette('mako')}):
    """
    based on this example: https://gist.github.com/LegrandNico/2b201863dc7ae28d568573c66047dd86
    """

    # setup data
    xlabel, ylabel = x, y
    data['diff'] = data[xlabel] - data[ylabel]
    if hue is not None:
        xs, ys, labels = [], [], []
        for h in data[hue].unique():
            xs.append(data[data[hue] == h][x].to_numpy())
            ys.append(data[data[hue] == h][y].to_numpy())
            labels.append(h)

    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6),
                                 gridspec_kw={'wspace': 0, 'hspace': -0, 'width_ratios': [0.8, 0.2],
                                              'height_ratios': [0.2, 0.8]})
    else:
        axes = fig.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0, 'hspace': -0, 'width_ratios': [0.8, 0.2],
                                              'height_ratios': [0.2, 0.8]})
    ax_joint = axes[1][0]
    ax_marg_x = axes[0][0]
    ax_marg_y = axes[1][1]
    ax_diag = axes[0][1]

    dist_from_corner = 0.6425  # distance for histogram to be plot from the corner of main plot
    hist_size = .7  # 0.5 this does not actually end up being the case bc of how I change the main plot size

    # Set axes limit
    all_values = [np.hstack(xs), np.hstack(ys)]
    lim_min = np.min(all_values) - (np.max(all_values) - np.min(all_values)) * .1
    lim_max = np.max(all_values) + (np.max(all_values) - np.min(all_values)) * .2
    lim = lim_min, lim_max
    if kind == 'scatter':
        diag_plot_extent = np.max(data[['diff', hue]].groupby([hue])
                                  .apply(lambda x: np.max(np.histogram(x, bins=20, density=True)[0])).values) * 1.5
    elif kind == 'kde':
        diag_plot_extent = 2.5

    # Set hist axes size
    hist_range = hist_size * np.sqrt(2) * (lim_max - lim_min)  # Length of X axis
    plot_extents = (-hist_range / 2, hist_range / 2, 0, diag_plot_extent)  # This creates axis limits
    bin_range = [-hist_range / 2, hist_range / 2]

    # setup diagonal histogram
    transform = Affine2D().scale(diag_plot_extent/(1.5*hist_range*np.sqrt(2)), 1/np.sqrt(2)).rotate_deg(-45)
    helper = floating_axes.GridHelperCurveLinear(transform, extremes=plot_extents, grid_locator1=MaxNLocator(4))
    inset = floating_axes.FloatingAxes(fig, [dist_from_corner, dist_from_corner, hist_size, hist_size], grid_helper=helper)
    bar_ax = inset.get_aux_axes(transform)
    fig.add_axes(inset)

    if kind == 'scatter':
        sns.scatterplot(data, x=xlabel, y=ylabel, hue=hue, ax=ax_joint, **plt_kwargs)
        sns.histplot(data, x=xlabel, hue=hue, palette=plt_kwargs['palette'], ax=ax_marg_x, legend=False, bins=20,
                     stat='density', common_norm=False, element='step')
        sns.histplot(data, y=ylabel, hue=hue, palette=plt_kwargs['palette'], ax=ax_marg_y, legend=False, bins=20,
                     stat='density', common_norm=False, element='step')
        sns.histplot(data, x='diff', hue=hue, palette=plt_kwargs['palette'], ax=bar_ax, legend=False, bins=20,
                     binrange=bin_range, stat='density', common_norm=False, element='step')
    elif kind == 'kde':
        try:
            sns.kdeplot(data, x=xlabel, y=ylabel, hue=hue, ax=ax_joint, **plt_kwargs)
            sns.kdeplot(data, x=xlabel, hue=hue, palette=plt_kwargs['palette'], ax=ax_marg_x, legend=False)
            sns.kdeplot(data, y=ylabel, hue=hue, palette=plt_kwargs['palette'], ax=ax_marg_y, legend=False)
            sns.kdeplot(data, x='diff', hue=hue, palette=plt_kwargs['palette'], ax=bar_ax, legend=False)
        except IndexError:
            warnings.warn('Not enough data to plot kde distributions')

    # setup limits and scaling
    ax_joint.plot(lim, lim, c=".7", dashes=(4, 2), zorder=0)
    ax_joint.set(xlim=lim, ylim=lim)
    ax_joint.set(xlabel=xlabel, ylabel=ylabel)

    bar_ax.plot((0, 0), (0, diag_plot_extent / 2), c="0", dashes=(4, 2), alpha=0.5)

    # turn off axes and tick lines
    for dir in ['left', 'right', 'top']:
        inset.axis[dir].set_visible(False)
    inset.axis["bottom"].major_ticklabels.set_rotation(45)

    # turn off spines and tick marks
    ax_diag.spines['left'].set_visible(False)
    ax_diag.spines['bottom'].set_visible(False)
    ax_marg_x.tick_params(axis='x', labelbottom=False)
    ax_marg_y.tick_params(axis='y', labelbottom=False)
    ax_marg_x.sharex(ax_joint)
    ax_marg_y.sharey(ax_joint)
    ax_marg_y.set_ylabel('')

    plt.setp(ax_diag.get_xticklabels(), visible=False)
    plt.setp(ax_diag.get_yticklabels(), visible=False)
    plt.setp(ax_diag.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_diag.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    fig.suptitle(title)

    return fig