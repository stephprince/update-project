from scipy.interpolate import griddata, interp1d


def interp1d_time_intervals(data, start_locs, stop_locs, new_times, time_offset, trials_to_flip):
    interpolated_position = []
    for start, stop, offset, flip_bool in zip(start_locs, stop_locs, time_offset, trials_to_flip):
        times = np.array(data.iloc[start:stop].index) - offset
        values = data.iloc[start:stop].values

        if flip_bool:
            values = -values

        fxn = interp1d(times, values, kind='linear')
        interpolated_position.append(fxn(new_times))

    return interpolated_position


def griddata_time_intervals(data, start_locs, stop_locs, time_offset, nbins, trials_to_flip):
    grid_prob = []
    for start, stop, offset, flip_bool in zip(start_locs, stop_locs, time_offset, trials_to_flip):
        proby = data.iloc[start:stop].stack().reset_index().values
        proby[:, 0] = proby[:, 0] - offset

        if flip_bool:
            proby[:, 1] = -proby[:, 1]  # flip position values so that prob density mapping is flipped

        x1 = np.linspace(min(proby[:, 0]), max(proby[:, 0]), nbins)  # time bins
        y1 = np.linspace(min(proby[:, 1]), max(proby[:, 1]), len(data.columns))  # position bins
        grid_x, grid_y = np.meshgrid(x1, y1)
        grid_prob_y = griddata(proby[:, 0:2], proby[:, 2], (grid_x, grid_y), method='linear', fill_value=np.nan)
        grid_prob.append(grid_prob_y)

    return grid_prob


def griddata_2d_time_intervals(data, binsxy, times, start_locs, stop_locs, time_offset, nbins, trials_to_flip):
    grid_prob = []
    flip_x = trials_to_flip['x'].values
    flip_y = trials_to_flip['y'].values
    for start, stop, offset, flip_bool_x, flip_bool_y in zip(start_locs, stop_locs, time_offset, flip_x, flip_y):
        times_around_update = times[start:stop] - offset
        data_subset = data[start:stop, :, :]

        prob_data = []
        for ind, t in enumerate(times_around_update):
            df = pd.DataFrame(data_subset[ind, :, :], index=binsxy[0], columns=binsxy[1])
            data_around_update = df.stack().reset_index().values
            data_from_timepoint = np.hstack((np.tile(t, len(data_around_update))[:, None], data_around_update))

            if flip_bool_x:
                data_from_timepoint[:, 1] = -data_from_timepoint[:,
                                             1]  # flip position values so that prob density mapping is flipped

            if flip_bool_y:
                data_from_timepoint[:, 2] = -data_from_timepoint[:, 2]

            prob_data.append(data_from_timepoint)

        all_data = np.vstack(prob_data)
        t1 = np.linspace(min(times_around_update), max(times_around_update), nbins)  # time bins
        x1 = np.linspace(min(binsxy[0]), max(binsxy[0]), np.shape(data)[1])  # x_position bins
        y1 = np.linspace(min(binsxy[1]), max(binsxy[1]), np.shape(data)[2])  # y_position bins
        grid_x, grid_y, grid_t = np.meshgrid(x1, y1, t1)

        grid_prob_trial = griddata(all_data[:, 0:3], all_data[:, 3], (grid_t, grid_x, grid_y), method='linear',
                                   fill_value=np.nan)
        grid_prob.append(grid_prob_trial)

    return grid_prob