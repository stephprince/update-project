import pickle


def get_track_boundaries():
    # establish track boundaries
    coords = [[2, 1], [2, 245], [25, 275], [25, 285], [14, 285], [0.5, 265], [-0.5, 265], [-14, 285], [-25, 285],
              [-25, 275], [-2, 245], [-2, 1], [2, 1]]
    xs, ys = zip(*coords)

    return xs, ys


def get_cue_locations():
    locations = dict(x={'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1 but add for bins
                     y={'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35, 'delay2 cue': 250.35,
                        'choice cue': 285})
    return locations


def load_pickled_data(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break