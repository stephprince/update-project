import numpy as np

from track_linearization import make_track_graph, get_linearized_position


class VirtualTrack:
    def __init__(self, coords, nodes, edges, cue_locations, linearization=False):
        self.coords = coords
        self.nodes = nodes
        self.edges = edges
        self.cue_locations = cue_locations
        self.linearization = linearization

    def get_track_boundaries(self):
        xs, ys = zip(*self.coords)

        return xs, ys

    def get_cue_locations(self):
        return self.cue_locations

    def linearize_track_position(self, position):
        track_graph = make_track_graph(self.nodes, self.edges)
        position_df = get_linearized_position(position=position, track_graph=track_graph)
        linear_position = position_df['linear_position'].values

        return linear_position


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
    cue_locations = dict(x_position={'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1, change for bins
                         view_angle={'home left max': -np.pi / 4, 'home right max': np.pi / 4},
                         y_position={'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35,
                                     'delay2 cue': 250.35, 'choice cue': 255})

    def __init__(self, coords=coords, nodes=nodes, edges=edges, cue_locations=cue_locations, linearization=False):
        super().__init__(coords, nodes, edges, cue_locations, linearization)

        # setup cue locations
        if self.linearization:
            self.cue_locations['y_position'].update({'left arm': 298, 'right arm': 340})
        else:
            self.cue_locations['y_position'].update({'arms': 285})

