import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from matplotlib.transforms import Affine2D, offset_copy
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator
from matplotlib.collections import LineCollection, PolyCollection


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
        if hasattr(axi, '_colorbar_info'):
            pass
        else:
            xlim = axi.get_xlim()
            ylim = axi.get_ylim()
            #
            # axi.set_xlabel(axi.get_xlabel().replace("_", " "))
            # axi.set_ylabel(axi.get_ylabel().replace("_", " "))
            # axi.set_title(axi.get_title().replace("_", " "))

            axi.spines['left'].set_bounds(ylim)
            axi.spines['bottom'].set_bounds(xlim)

            sns.despine(ax=axi, offset=5)
    # sns.despine(fig=fig, offset=5)

    if tight_layout:
        fig.tight_layout()

    fig.align_labels()


def clean_box_plot(ax, labelcolors=None, fill=True):
    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    colors = [patch.get_facecolor() for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    colors = colors * int(len(box_patches) / len(colors))
    lines_per_boxplot = len(ax.lines) // len(box_patches)
    for i, (box, color) in enumerate(zip(box_patches, colors)):
        box.set_edgecolor(color)
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            if line.get_color() != 'white':  # leave the median white
                line.set_color(color)

    if labelcolors:
        colors = labelcolors  # get new list of colors if different labelcolors provided

    for ticklabel, color in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(color)

    if not fill:
        [patch.set_facecolor(None) for patch in ax.patches if type(patch) == mpl.patches.PathPatch]

    return ax


def clean_violin_plot(ax, colors, line_start=0):
    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        violin.set(facecolor=(*mpl.colors.to_rgb(colors[ind]), 0.5))
        violin.set(edgecolor=(*mpl.colors.to_rgb(colors[ind]), 1))

    for ind, l in enumerate(ax.lines[line_start:]):
        color_ind = int(np.floor(ind / len(colors)))
        if color_ind < len(colors):
            if (ind - 1) % len(colors) == 0:
                l.set(linestyle='-', linewidth=1, color=colors[color_ind], alpha=0.8)
            else:
                l.set(linestyle='--', linewidth=0.5, color=colors[color_ind])

    return ax


def add_task_phase_lines(ax, cue_locations=dict(), label_dict=None, text_height=0.95, text_brackets=False,
                         vline_kwargs=None):
    name_remapping = label_dict or {'initial cue': 'sample', 'delay cue': 'delay', 'update cue': 'update',
                      'delay2 cue': 'delay'}
    vline_kwargs = vline_kwargs or dict(linestyle='solid', color='#ececec', linewidth=0.75)

    cue_details = dict()
    for i, cue_loc in enumerate([*list(cue_locations.values()), 1][1:]):
        cue_name = list(cue_locations.keys())[i]
        cue_details[cue_name] = dict(middle=(cue_loc + list(cue_locations.values())[i]) / 2,
                                     start=list(cue_locations.values())[i], end=cue_loc,
                                     label=name_remapping[cue_name])
    for i, (cue_name, cue_loc) in enumerate(cue_locations.items()):
        ax.axvline(cue_loc, **vline_kwargs)

    if text_brackets:
        for cue_name, cue_loc in cue_details.items():
            ax.text(cue_details[cue_name]['middle'], text_height, cue_details[cue_name]['label'], ha='center',
                           va='bottom', transform=ax.get_xaxis_transform(), fontsize=7)
            ax.annotate('', xy=(cue_details[cue_name]['start'], text_height - 0.025),
                               xycoords=ax.get_xaxis_transform(),
                               xytext=(cue_details[cue_name]['end'], text_height - 0.025),
                               textcoords=ax.get_xaxis_transform(),
                               arrowprops=dict(arrowstyle='|-|, widthA=0.15, widthB=0.15', shrinkA=1, shrinkB=1, lw=1),
                               ha='left', rotation=30)
    return ax

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

    color_theme_dict['left'] = '#2594f6'  # 250 in degrees, 95 saturation, 60 light
    color_theme_dict['right'] = '#da3b46'  # 12 in degrees, 95 saturation, 60 light
    color_theme_dict['left_cmap'] = sns.light_palette(color_theme_dict['left'], as_cmap=True)
    color_theme_dict['right_cmap'] = sns.light_palette(color_theme_dict['right'], as_cmap=True)
    color_theme_dict['left_right_cmap_div'] = sns.diverging_palette(250, 12, s=95, l=60, as_cmap=True)

    color_theme_dict['phase_dividers'] = '#ececec'

    for key in ['control', 'nan', 'home', 'central', 'non_update', 'non update', 'all', 'delay only', 'correct', 'local']:
        color_theme_dict[key] = '#303030'  # black - 0 in degrees, 0 saturation, 20 light
        color_theme_dict[f'{key}_light'] = '#c0c0c0'  # black - 0 in degrees, 0 saturation, 20 light
        color_theme_dict[f'{key}_medium'] = '#898989'  # black - 0 in degrees, 0 saturation, 20 light
        color_theme_dict[f'{key}_cmap'] = sns.color_palette('blend:#ffffff,#000000', as_cmap=True)
        color_theme_dict[f'{key}_cmap_r'] = sns.color_palette('blend:#000000,#ffffff', as_cmap=True)
        color_theme_dict[f'{key}_quartiles'] = sns.color_palette('blend:#c0c0c0,#303030', 4)  # increasing grey scales
        color_theme_dict[f'{key}_cmap_light_to_dark'] = sns.color_palette('blend:#c0c0c0,#303030', as_cmap=True)  # (40 to 80 light)
    for key in ['original']:
        color_theme_dict[key] = '#2594f6'  # light blue - 250 in degrees, 95 saturation, 60 light
    for key in ['switch_trials', 'switch']:
        color_theme_dict[key] = '#785cf7'  # purple - 270 in degrees, 95 saturation, 50 light
        color_theme_dict[f'{key}_light'] = '#c7c0fc'  # purple - 270 in degrees, 95 saturation, 80 light
    for key in ['stay_trials', 'stay']:
        color_theme_dict[key] = '#178761'  # green - 152 in degrees, 95 saturation, 50 light (was 60)
        color_theme_dict[f'{key}_light'] = '#66dca9'  # green - 152 in degrees, 95 saturation, 80 light
    for key in ['initial', 'initial_stay', 'stay_update']:
        color_theme_dict[key] = '#2459bd'  # blue - 258 degrees, 85 saturation, 40 light
        color_theme_dict[f'{key}_light'] = '#b7c4fa'
        color_theme_dict[f'{key}_cmap'] = sns.color_palette('blend:#ffffff,#2459bd',
                                                            as_cmap=True)  # start at dark blue (30 light)
        color_theme_dict[f'{key}_quartiles'] = sns.color_palette('blend:#b7c4fa,#2459bd', 4)  # (30 to 80 light)
        color_theme_dict[f'{key}_cmap_light_to_dark'] = sns.color_palette("blend:#b7c4fa,#2459bd", as_cmap=True)  # (40 to 80 light)
    for key in ['new', 'switch_update']:
        color_theme_dict[key] = '#b01e70'  # pink - 345 degrees, 90 saturation, 40 light (was 30, testing out)
        color_theme_dict[f'{key}_light'] = '#fab2cf'  # 80 light
        color_theme_dict[f'{key}_cmap'] = sns.color_palette('blend:#ffffff,#b01e70',
                                                            as_cmap=True)  # start at dark pink (30 light)
        color_theme_dict[f'{key}_quartiles'] = sns.color_palette('blend:#fab2cf,#b01e70', 4)  # (40 to 80 light)
        color_theme_dict[f'{key}_cmap_light_to_dark'] = sns.color_palette('blend:#fab2cf,#b01e70', as_cmap=True)  # (40 to 80 light)
    for key in ['error', 'incorrect']:
        color_theme_dict[key] = '#dba527'  # 57 degrees, 95 sat, 71 light  #'#bc1026'  # 10 degrees, 95 saturation, 40 light
    for key in ['choice', 'choice_commitment']:
        color_theme_dict[key] = sns.light_palette("#9119cf", 5)  # 285 degrees, 95 saturation, 40 light
        color_theme_dict[f'{key}_cmap'] = sns.color_palette("blend:#d7c0e4,#6e119f", as_cmap=True)  # start at dark purple (30 light)

    color_theme_dict['cmap'] = sns.color_palette("rocket_r", as_cmap=True)
    color_theme_dict['cmap_r'] = sns.color_palette("rocket", as_cmap=True)
    color_theme_dict['div_cmap'] = sns.diverging_palette(240, 10, s=75, l=60, n=5, center='light', as_cmap=True)
    color_theme_dict['switch_stay_cmap_div'] = sns.diverging_palette(345, 258, s=90, l=50, as_cmap=True)

    color_theme_dict['animals'] = sns.color_palette("husl", 7)
    color_theme_dict['general'] = sns.color_palette("husl", 10)
    color_theme_dict['trials'] = [color_theme_dict['non_update'], color_theme_dict['switch_trials'],
                                  color_theme_dict['stay_trials']]
    color_theme_dict['delays'] = sns.color_palette('blend:#c0c0c0,#303030', 5)  # increasing grey scales

    return color_theme_dict


def plot_distributions(data, axes, column_name, group, row_ids, col_ids, xlabel, title, stripplot=True,
                       show_median=True,
                       palette=None, histstat='proportion', ):
    if group:
        palette = palette or sns.color_palette(n_colors=len(data[group].stack().unique()))
        if len(palette) > len(data[group].stack().unique()):
            palette = palette[:len(data[group].stack().unique())]
    else:
        palette = palette or sns.color_palette()

    if group and show_median:
        medians = data.groupby(group)[column_name].median()#group was in [], DC change back
        limits = [np.nanmin(data.groupby(group)[column_name].min().values),#group was in [], DC change back
                  np.nanmax(data.groupby(group)[column_name].max().values)]#group was in [], DC change back
    else:
        medians = {column_name: data[column_name].median()}
        limits = [data[column_name].min(), data[column_name].max()]

    # cum fraction plots
    axes[row_ids[0]][col_ids[0]] = sns.ecdfplot(data=data, x=column_name, hue=data['region'], ax=axes[row_ids[0]][col_ids[0]],
                                                palette=palette)#hue was group, DC change back
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
    axes[row_ids[1]][col_ids[1]] = sns.histplot(data=data, x=column_name, hue=data['region'], ax=axes[row_ids[1]][col_ids[1]],
                                                element='step', palette=palette, stat=histstat, common_norm=False)#hue was group, DC change back
    axes[row_ids[1]][col_ids[1]].set(xlabel=xlabel, ylabel='Proportion', xlim=limits)

    # violin plots
    axes[row_ids[2]][col_ids[2]] = sns.violinplot(data=data, x='region', y=column_name, ax=axes[row_ids[2]][col_ids[2]],
                                                  palette=palette)#x was group, DC change back
    plt.setp(axes[row_ids[2]][col_ids[2]].collections, alpha=.25)
    if stripplot:
        sns.stripplot(data=data, y=column_name, x='region', size=3, jitter=True, ax=axes[row_ids[2]][col_ids[2]],
                      palette=palette)#x was group, DC change back
    axes[row_ids[2]][col_ids[2]].set_title(title)


def plot_scatter_with_distributions(data, x, y, hue, kind='scatter', fig=None, title=None,
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
    transform = Affine2D().scale(diag_plot_extent / (1.5 * hist_range * np.sqrt(2)), 1 / np.sqrt(2)).rotate_deg(-45)
    helper = floating_axes.GridHelperCurveLinear(transform, extremes=plot_extents, grid_locator1=MaxNLocator(4))
    inset = floating_axes.FloatingAxes(fig, [dist_from_corner, dist_from_corner, hist_size, hist_size],
                                       grid_helper=helper)
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


def rainbow_text(x, y, strings, colors, orientation='stacked',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    new_line = '\n'
    if ax is None:
        ax = plt.gca()
    t = ax.transAxes
    fig = ax.figure
    canvas = fig.canvas

    assert orientation in ['horizontal', 'vertical', 'stacked']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, f'{s} {new_line}', color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # Convert window extent from pixels to inches
        # to avoid issues displaying at different dpi
        ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)

        if orientation == 'horizontal':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=ex.width, y=0)
        elif orientation == 'vertical':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=0, y=ex.height)
        elif orientation == 'stacked':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=0, y=-ex.height * 0.5)  # adjust to make gap smaller


def colorline(x, y, z=None, cmap=sns.color_palette('Greys'), norm=plt.Normalize(0.0, 1.0), linewidth=1,
              linestyle='solid', alpha=1.0, label='', ax=plt.gca()):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    # create list of line segments in correct format for line collection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                        label=label)

    ax.add_collection(lc)

    return ax