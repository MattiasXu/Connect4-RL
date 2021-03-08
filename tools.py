import numpy as np

# https://stackoverflow.com/a/36960495/ @Divikar
def onehot(a):
    ncols = 3
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)