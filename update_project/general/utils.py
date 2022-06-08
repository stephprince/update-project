def create_track_boundaries():
    # establish track boundaries
    coords = [[2, 1], [2, 245], [25, 275], [25, 285], [14, 285], [0.5, 265], [-0.5, 265], [-14, 285], [-25, 285],
              [-25, 275], [-2, 245], [-2, 1], [2, 1]]
    xs, ys = zip(*coords)

    return xs, ys
