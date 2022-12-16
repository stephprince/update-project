import numpy as np

from track_linearization import make_track_graph, get_linearized_position


class VirtualTrack:
    def __init__(self, coords, nodes, edges, cue_start_locations, cue_end_locations, choice_boundaries, home_boundaries,
                 mappings=None, linearization=False):
        self.coords = coords
        self.nodes = nodes
        self.edges = edges
        self.cue_start_locations = cue_start_locations
        self.cue_end_locations = cue_end_locations
        self.choice_boundaries = choice_boundaries
        self.home_boundaries = home_boundaries
        self.mappings = mappings
        self.linearization = linearization
        self.edge_spacing = []

        # update cue end locations if linearized
        # if linearization:
        #     self.cue_end_locations['y_position'].update(dict(left_arm=298, right_arm=341))

    def get_track_boundaries(self):
        xs, ys = zip(*self.coords)

        return xs, ys

    def get_cue_locations(self):
        return self.cue_end_locations

    def linearize_track_position(self, position):
        edge_spacing = 20
        track_graph = make_track_graph(self.nodes, self.edges)
        position_df = get_linearized_position(position=position, track_graph=track_graph, edge_spacing=edge_spacing)
        linear_position = position_df['linear_position'].values

        home_arm_max = np.round(position_df.query('track_segment_id == 0')['linear_position'].max())
        left_arm_max = np.round(position_df.query('track_segment_id == 1')['linear_position'].max())
        left_arm_min = np.round(position_df.query('track_segment_id == 1')['linear_position'].min())
        right_arm_max = np.round(position_df.query('track_segment_id == 2')['linear_position'].max())
        right_arm_min = np.round(position_df.query('track_segment_id == 2')['linear_position'].min())
        if right_arm_max < 379:
            right_arm_max = 380  # catch for one session where value never reaches actual maximum (likely bc lat-travel)
        self.cue_end_locations['y_position'].update(dict(left_arm=left_arm_max, right_arm=right_arm_max))

        # set choice boundaries so left/right sides are equal
        self.choice_boundaries['y_position'].update(dict(left=(left_arm_min - edge_spacing / 4,
                                                               left_arm_max + edge_spacing / 4),
                                                         right=(right_arm_min - edge_spacing / 2,
                                                                right_arm_max)))
        self.edge_spacing = [(home_arm_max, left_arm_min), (left_arm_max, right_arm_min)]

        return linear_position

    def get_choice_locations(self):
        return self.choice_locations

    def get_limits(self, dim='y_position'):
        xs, ys = self.get_track_boundaries()
        if self.linearization:
            ys = ys + (np.max(list(self.cue_end_locations['y_position'].values())),)

        limits = dict()
        limits['x_position'] = [np.min(xs), np.max(xs)]
        limits['y_position'] = [5, np.max(ys)]  # mouse can never be behind 5 in the update track
        limits['choice'] = (-0.5, 0.5)

        return limits[dim]


class UpdateTrack(VirtualTrack):
    coords = [[2, 1], [2, 245], [25, 275], [33, 285], [14, 285], [0.5, 265], [-0.5, 265], [-14, 285],
              [-33, 285], [-25, 275], [-2, 245], [-2, 1], [2, 1]]
    nodes = [(0, 0),  # start of home arm
             (0, 255),  # end of home arm, specify so that the lengths of arms are 42
             (-30, 285),  # left arm  # TODO - determine why this is 30 not 33
             (30, 285)]  # right arm = nodes
    edges = [(0, 1),
             (1, 2),
             (1, 3)]
    cue_end_locations = dict(x_position={'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1, change for bins
                             view_angle={'home left max': -np.pi / 4, 'home right max': np.pi / 4},
                             y_position={'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35,
                                         'delay2 cue': 250.35, 'choice cue': 255},  # TODO - update based on linearized pos bc that shifts the cue locations
                             dynamic_choice={'p_left': -0.4, 'p_right': 0.4})
    cue_start_locations = dict(y_position={'initial cue': 5, 'delay cue': 120.325, 'update cue': 145.125,
                                           'delay2 cue': 215.125})
    choice_boundaries = dict(x_position={'left': (-33, -1), 'right': (1, 33)},
                             y_position={'left': (255, 298), 'right': (298, 341)},  # TODO - change this part
                             view_angle={'left': (np.pi,  2 * np.pi / 9), 'right': (-2 * np.pi/9, -np.pi)},  # 40 deg
                             choice_binarized={'left': (-2, 0), 'right': (0, 2)},
                             choice={'left': (-0.5, -0.4), 'right': (0.4, 0.5)},
                             turn_type={'left': (-2, 0), 'right': (0, 2)},
                             dynamic_choice={'left': (-0.5, -0.4), 'right': (0.4, 0.5)})
    mappings = dict(update_type={'1': 'non_update', '2': 'switch_update', '3': 'stay_update'},
                    turn_type={'1': 'left', '2': 'right'})
    home_boundaries = dict(x_position=(-1, 1),
                           y_position=(5, 255),
                           view_angle=(2 * np.pi / 9, -2 * np.pi/9),
                           dynamic_choice=(-0.5, 0.5),
                           cue_bias=(-0.4, 0.4))

    def __init__(self, coords=coords, nodes=nodes, edges=edges, cue_start_locations=cue_start_locations,
                 cue_end_locations=cue_end_locations, choice_boundaries=choice_boundaries,
                 home_boundaries=home_boundaries, mappings=mappings, linearization=False):
        super().__init__(coords, nodes, edges, cue_start_locations, cue_end_locations, choice_boundaries,
                         home_boundaries, mappings, linearization)
