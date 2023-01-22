import gzip
from itertools import groupby
import numpy as np


def read_locations(file_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    count = 0
    locations = []
    with gzip.open(file_name, 'rt') as f:
        for line in f:
            count += 1
            print(count)
            locations.append(int(line.split("\t")[1]))
    return np.array(locations)


if __name__ == '__main__':
    locs = np.load("locations.npy")
    print(locs.shape)

    # locations = read_locations("hg19.CpG.bed.gz")
    # np.save("locations.npy", locations)
    # np.save("dists.npy", dists)
