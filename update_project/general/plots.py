import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate

color_wheel = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# red = color_wheel[2]
# green = color_wheel[3]
# color_wheel[2] = green
# color_wheel[3] = red  # flip colors bc of how I have correct/incorrect trials coded


def clean_plot(fig, axes):
    if hasattr(axes, 'flat'):
        ax_list = axes.flat
    else:
        ax_list = list(axes.values())

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
    color_theme_dict['cmap'] = sns.color_palette("rocket_r", as_cmap=True)  # rocket is cool too?
    color_theme_dict['control'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['left'] = '#2594f6'  # 250 in degrees, 95 saturation, 60 light
    color_theme_dict['right'] = '#da3b46'  # 12 in degrees, 95 saturation, 60 light
    color_theme_dict['stay'] = '#1ea477'  # 152 in degrees, 95 saturation, 60 light
    color_theme_dict['switch'] = '#927ff9'  # 270 in degrees, 95 saturation, 60 light
    color_theme_dict['stay_update'] = color_theme_dict['stay']
    color_theme_dict['switch_update'] = color_theme_dict['switch']
    color_theme_dict['non_update'] = '#474747'  # 12 in degrees, 0 saturation, 30 light
    color_theme_dict['error'] = '#a150db'  # 285 degrees, 75 saturation, 50 light
    color_theme_dict['switch_cmap'] = sns.light_palette(color_theme_dict['switch'], as_cmap=True)
    color_theme_dict['stay_cmap'] = sns.light_palette(color_theme_dict['stay'], as_cmap=True)
    color_theme_dict['switch_stay_cmap_div'] = sns.diverging_palette(270, 152, s=95, l=60, as_cmap=True)
    color_theme_dict['left_right_cmap_div'] = sns.diverging_palette(250, 12, s=95, l=60, as_cmap=True)
    color_theme_dict['animals'] = sns.color_palette("husl", 7)

    return color_theme_dict

def plot_distributions(data, axes, column_name, group, row_ids, col_ids, xlabel, title, stripplot=True, show_median=True):
    if group and show_median:
        medians = data.groupby([group])[column_name].median()
        limits = [np.nanmin(data.groupby([group])[column_name].min().values),
                  np.nanmax(data.groupby([group])[column_name].max().values)]
    else:
        medians = {column_name: data[column_name].median()}
        limits = [data[column_name].min(), data[column_name].max()]

    # cum fraction plots
    axes[row_ids[0]][col_ids[0]] = sns.ecdfplot(data=data, x=column_name, hue=group, ax=axes[row_ids[0]][col_ids[0]],
                                                palette=sns.color_palette(n_colors=len(data[group].unique())))
    axes[row_ids[0]][col_ids[0]].set_title(title)
    axes[row_ids[0]][col_ids[0]].set(xlabel=xlabel, ylabel='Proportion', xlim=limits)
    axes[row_ids[0]][col_ids[0]].set_aspect(1. / axes[row_ids[0]][col_ids[0]].get_data_ratio(), adjustable='box')

    # add median annotations to the first plot
    new_line = '\n'
    median_text = [f"{g} median: {m:.2f} {new_line}" for g, m in medians.items()]
    axes[row_ids[0]][col_ids[0]].text(0.55, 0.2, ''.join(median_text),
                                      transform=axes[row_ids[0]][col_ids[0]].transAxes, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # histograms
    axes[row_ids[1]][col_ids[1]] = sns.histplot(data=data, x=column_name, hue=group, ax=axes[row_ids[1]][col_ids[1]],
                                                element='step', palette=sns.color_palette(n_colors=len(data[group].unique())))
    axes[row_ids[1]][col_ids[1]].set(xlabel=xlabel, ylabel='Proportion', xlim=limits)

    # violin plots
    axes[row_ids[2]][col_ids[2]] = sns.violinplot(data=data, x=group, y=column_name, ax=axes[row_ids[2]][col_ids[2]],)
    plt.setp(axes[row_ids[2]][col_ids[2]].collections, alpha=.25)
    if stripplot:
        sns.stripplot(data=data, y=column_name, x=group, size=3, jitter=True, ax=axes[row_ids[2]][col_ids[2]],)
    axes[row_ids[2]][col_ids[2]].set_title(title)

def show_start_aligned_psth(start_aligned, note_times, group_inds, window, labels, axes=None):
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    else:
        axes = axes if isinstance(axes, list) else [axes]

    # add annotation times and markers to each of the psth plots
    annotation_times = dict()
    for key, times in note_times.items():
        annotation_times[key] = np.array([val for val in times for _ in range(2)])
    update_bool = ~np.isnan(annotation_times['t_update'])
    lines_for_delay = annotation_times['t_choice_made'].copy()
    lines_for_delay[update_bool] = annotation_times['t_update'][update_bool]

    # add fills for event intervals
    y_to_fill = [(x * 0.5) - 0.5 for x in range(0, len(start_aligned) * 2)]
    y_to_fill_no_gaps = [y - 0.25 if ind % 2 == 0 else y + 0.25 for ind, y in enumerate(y_to_fill)]
    y_length = len(start_aligned)
    for ax in axes:
        # fill delay and choice periods
        ax.fill_betweenx(y_to_fill_no_gaps, annotation_times['t_delay'], lines_for_delay, facecolor='k', alpha=0.1)
        ax.fill_betweenx(y_to_fill_no_gaps, annotation_times['t_update'], annotation_times['t_delay2'],
                         where=update_bool, facecolor='c', alpha=0.1)
        ax.fill_betweenx(y_to_fill_no_gaps, annotation_times['t_delay2'], annotation_times['t_choice_made'],
                         where=update_bool, facecolor='k', alpha=0.1)
        ax.fill_betweenx(y_to_fill_no_gaps, annotation_times['t_choice_made'], np.zeros(len(y_to_fill))+window,
                         facecolor='y', alpha=0.1)

        # add test annotations
        show_event_annotations(ax, annotation_times, y_length)

        # add raster plots for each group
        show_psth_raster(start_aligned, ax=ax, start=0, end=window, group_inds=group_inds, labels=labels)


def show_event_annotations(ax, annotation_times, y_length):
    text_offset = -y_length/10
    update_bool = ~np.isnan(annotation_times['t_update'])

    # add event time point markers
    ax.plot(np.zeros(y_length) + 3, range(y_length), color='black', alpha=0.25)
    ax.annotate('cue on (white)', (0, 0), xycoords='data', xytext=(0, text_offset), textcoords='data', ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('movement starts', (3, 0), xycoords='data', xytext=(3, text_offset), textcoords='data', ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('delay on (grey)', (annotation_times['t_delay'][0], 0), xycoords='data',
                xytext=(annotation_times['t_delay'][0], text_offset), textcoords='data', ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('update on (cyan)', (annotation_times['t_update'][update_bool][0], 0), xycoords='data',
                xytext=(annotation_times['t_update'][update_bool][0], text_offset + text_offset * 0.5),
                textcoords='data', ha='center', arrowprops=dict(arrowstyle='->'))
    ax.annotate('choice made (yellow)', (annotation_times['t_choice_made'][0], 0), xycoords='data',
                xytext=(annotation_times['t_choice_made'][0], text_offset), textcoords='data', ha='center',
                arrowprops=dict(arrowstyle='->'))

def show_event_aligned_psth(spikes_aligned, window, group_inds, trial_inds, labels, ax_dict=None, ax_start_key=None):
    if ax_dict is None:
        nrows = 2
        ncols = 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    else:
        ax_keys = [key for key in ax_dict.keys() if key >= ax_start_key]  # get all keys after start key (alphabetical)
        ncols = int(len(ax_keys)/2)

    for ind, event_name in enumerate(spikes_aligned):
        if ax_dict is None:
            plot_col = ind - np.floor(ind / ncols)
            ax_top = axes[0][plot_col]
            ax_bottom = axes[1][plot_col]
        else:
            ax_top = ax_dict[ax_keys[ind]]
            ax_bottom = ax_dict[ax_keys[ind+ncols]]

        group_inds_subset = group_inds[trial_inds[event_name]]
        show_psth_smoothed(spikes_aligned[event_name], ax=ax_top, start=-window, end=window, sigma_in_secs=0.1,
                           group_inds=group_inds_subset)
        show_psth_raster(spikes_aligned[event_name], ax=ax_bottom, start=-window, end=window, group_inds=group_inds_subset,
                         labels=labels)
        ax_top.set_title(f'{event_name}', fontsize=14)


def show_psth_smoothed(
        data,
        ax,
        start: float,
        end: float,
        group_inds=None,
        sigma_in_secs: float = 0.05,
        ntt: int = 1000,
):
    if not len(data):  # TODO: when does this occur?
        return
    all_data = np.hstack(data)
    if not len(all_data):  # no spikes
        return
    tt = np.linspace(start, end, ntt)
    smoothed = np.array(
        [compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data]
    )

    if group_inds is None:
        group_inds = np.zeros((len(smoothed)), dtype=np.int)
    group_stats = []
    for group in np.unique(group_inds):
        this_mean = np.mean(smoothed[group_inds == group], axis=0)
        err = scipy.stats.sem(smoothed[group_inds == group], axis=0)
        group_stats.append(
            dict(
                mean=this_mean,
                lower=this_mean - 2 * err,
                upper=this_mean + 2 * err,
                group=group,
            )
        )
    for stats in group_stats:
        color = color_wheel[stats["group"] % len(color_wheel)]
        ax.plot(tt, stats["mean"], color=color)
        ax.fill_between(tt, stats["lower"], stats["upper"], alpha=0.2, color=color)

def show_psth_raster(
        data,
        start=-0.5,
        end=2.0,
        group_inds=None,
        labels=None,
        ax=None,
        show_legend=True,
        align_line_color=(0.7, 0.7, 0.7),
        fontsize=12,
) -> plt.Axes:
    """
    Parameters
    ----------
    data: array-like
    start: float
        Start time for calculation before or after (negative or positive) the reference point (aligned to).
    end: float
        End time for calculation before or after (negative or positive) the reference point (aligned to).
    group_inds: array-like, optional
    labels: array-like, optional
    ax: plt.Axes, optional
    show_legend: bool, optional
    align_line_color: array-like, optional
        [R, G, B] (0-1)
        Default = [0.7, 0.7, 0.7]
    progress_bar: FloatProgress, optional
    fontsize: int, optional
    Returns
    -------
    plt.Axes
    """
    if not len(data):
        return ax
    ax = plot_grouped_events(
        data,
        [start, end],
        group_inds,
        color_wheel,
        ax,
        labels,
        show_legend=show_legend,
        fontsize=fontsize
    )
    ax.axvline(color=align_line_color)
    return ax


def plot_grouped_events(
        data,
        window,
        group_inds=None,
        colors=color_wheel,
        ax=None,
        labels=None,
        show_legend=True,
        offset=0,
        figsize=(8, 6),
        fontsize=12,
):
    """
    Parameters
    ----------
    data: array-like
    window: array-like [float, float]
        Time in seconds
    group_inds: array-like dtype=int, optional
    colors: array-like, optional
    ax: plt.Axes, optional
    labels: array-like dtype=str, optional
    show_legend: bool, optional
    offset: number, optional
    figsize: tuple, optional
    fontsize: int, optional
    Returns
    -------
    """

    data = np.asarray(data, dtype="object")
    legend_kwargs = dict()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        if hasattr(fig, "canvas"):
            fig.canvas.header_visible = False
        else:
            legend_kwargs.update(bbox_to_anchor=(1.01, 1))
    if group_inds is not None:
        ugroup_inds = np.unique(group_inds)
        handles = []
        this_iter = enumerate(ugroup_inds)
        for i, ui in this_iter:
            color = colors[ugroup_inds[i] % len(colors)]
            lineoffsets = np.where(group_inds == ui)[0] + offset
            event_collection = ax.eventplot(
                data[group_inds == ui],
                orientation="horizontal",
                lineoffsets=lineoffsets,
                color=color,
            )
            handles.append(event_collection[0])
        if show_legend:
            ax.legend(
                handles=handles[::-1],
                labels=list(labels[ugroup_inds][::-1]),
                loc="upper left",
                bbox_to_anchor=(0.6, 1), #(1.01, 1),
                **legend_kwargs,
            )
    else:
        ax.eventplot(
            data,
            orientation="horizontal",
            color="k",
            lineoffsets=np.arange(len(data)) + offset,
        )

    ax.set_xlim(window)
    ax.set_xlabel("time (s)", fontsize=fontsize)
    ax.set_ylim(np.array([-0.5, len(data) - 0.5]) + offset)
    if len(data) <= 30:
        ax.set_yticks(range(offset, len(data) + offset))
        ax.set_yticklabels(range(offset, len(data) + offset))

    return ax