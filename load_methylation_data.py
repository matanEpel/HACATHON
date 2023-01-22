if __name__ == '__main__':
    import numpy as np
    NUM_ROWS = 28217448
    table = np.fromfile('merged.lpairs', dtype=np.uint16, count=4 * NUM_ROWS).reshape((-1, 4))
    np.save("amount_in_pair.npy", table)
