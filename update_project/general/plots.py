import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def clean_plot(fig, axes):
    if hasattr(axes, 'flat'):
        ax_list = axes.flat
    elif hasattr(axes,'values'):
        ax_list = list(axes.values())
    elif isinstance(axes,list):
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

    sns.despine(fig=fig, offset=5)

    fig.tight_layout()


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

    color_theme_dict['control'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['left'] = '#2594f6'  # 250 in degrees, 95 saturation, 60 light
    color_theme_dict['right'] = '#da3b46'  # 12 in degrees, 95 saturation, 60 light
    color_theme_dict['stay'] = '#1ea477'  # 152 in degrees, 95 saturation, 60 light
    color_theme_dict['switch'] = '#927ff9'  # 270 in degrees, 95 saturation, 60 light
    color_theme_dict['stay_update'] = color_theme_dict['stay']
    color_theme_dict['switch_update'] = color_theme_dict['switch']
    color_theme_dict['initial_stay'] = color_theme_dict['stay']
    color_theme_dict['non_update'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['error'] = '#a150db'  # 285 degrees, 75 saturation, 50 light

    color_theme_dict['cmap'] = sns.color_palette("rocket_r", as_cmap=True)
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
                       palette=None, histstat='proportion', common_norm=False):
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

