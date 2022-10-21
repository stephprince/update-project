import numpy as np

from track_linearization import make_track_graph, get_linearized_position


class VirtualTrack:
    def __init__(self, coords, nodes, edges, cue_start_locations, cue_end_locations, choice_boundaries, home_boundaries,
                 mappings=None, linearization=False, delay_locations=None, trial_types=None):
        self.coords = coords
        self.nodes = nodes
        self.edges = edges
        self.cue_start_locations = cue_start_locations
        self.cue_end_locations = cue_end_locations
        self.choice_boundaries = choice_boundaries
        self.home_boundaries = home_boundaries
        self.mappings = mappings
        self.linearization = linearization
        self.delay_locations = delay_locations
        self.trial_types=trial_types

    def get_track_boundaries(self):
        xs, ys = zip(*self.coords)

        return xs, ys

    def linearize_track_position(self, position):
        track_graph = make_track_graph(self.nodes, self.edges)
        position_df = get_linearized_position(position=position, track_graph=track_graph)
        linear_position = position_df['linear_position'].values

        return linear_position

    def get_choice_locations(self):
        return self.choice_locations

    def get_limits(self, dim='y_position'):
        limits = dict()
        xs, ys = self.get_track_boundaries()
        limits['x_position'] = [np.min(xs), np.max(xs)]
        limits['y_position'] = [np.min(ys), np.max(ys)]

        return limits[dim]


class UpdateTrack(VirtualTrack):
    coords = [[2, 1], [2, 245], [25, 275], [25, 285], [14, 285], [0.5, 265], [-0.5, 265], [-14, 285],
              [-25, 285], [-25, 275], [-2, 245], [-2, 1], [2, 1]]
    nodes = [(0, 0),  # start of home arm
             (0, 255),  # end of home arm
             (-30, 285),  # left arm
             (30, 285)]  # right arm = nodes
    edges = [(0, 1),
             (1, 2),
             (1, 3)]
    cue_end_locations = dict(x_position={'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1, change for bins
                             view_angle={'home left max': -np.pi / 4, 'home right max': np.pi / 4},
                             y_position={'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35,
                                         'delay2 cue': 250.35, 'choice cue': 255},
                             dynamic_choice={'p_left': -0.4, 'p_right': 0.4})
    cue_start_locations = dict(y_position={'initial cue': 5, 'delay cue': 120.325, 'update cue': 145.125,
                                           'delay2 cue': 215.125})
    choice_boundaries = dict(x_position={'left': (-33, -1), 'right': (1, 33)},
                             y_position={'left': (255, 298), 'right': (298, 341)},
                             view_angle={'left': (np.pi,  2 * np.pi / 9), 'right': (-2 * np.pi/9, -np.pi)},  # 40 deg
                             choice={'left': (-2, 0), 'right': (0, 2)},
                             turn_type={'left': (-2, 0), 'right': (0, 2)},
                             dynamic_choice={'left': (-1, -0.4), 'right': (0.4, 1)})
    mappings = dict(update_type={'1': 'non_update', '2': 'switch_update', '3': 'stay_update'},
                    turn_type={'1': 'left', '2': 'right'})
    home_boundaries = dict(x_position=(-1, 1),
                           y_position=(5, 255),
                           view_angle=(2 * np.pi / 9, -2 * np.pi/9),
                           dynamic_choice=(-0.5, 0.5),
                           cue_bias=(-0.4, 0.4))
    # TODO - add dictionary with delay phase onset locations/ranges
    delay_locations = dict(delay1=(214,216), # latest delay
                         delay2=(179,181), # later delay
                         delay3=(144,146), # middle delay
                         delay4=(119,121)) # earlier delay
    trial_types = ['linear', 'ymaze_short', 'ymaze_long', 'delay1', 'delay2', 'delay3', 'delay4', 'stay_update',
                   'switch_update']

    def __init__(self, coords=coords, nodes=nodes, edges=edges, cue_start_locations=cue_start_locations,
                 cue_end_locations=cue_end_locations, choice_boundaries=choice_boundaries,
                 home_boundaries=home_boundaries, mappings=mappings, linearization=False, delay_locations=delay_locations, trial_types=trial_types):
        super().__init__(coords, nodes, edges, cue_start_locations, cue_end_locations, choice_boundaries,
                         home_boundaries, mappings, linearization, delay_locations, trial_types)
