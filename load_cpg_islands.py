import gzip
from itertools import groupby
import numpy as np

def read_islands(file_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = gzip.open(file_name)
    islands = []
    faiter = (x[1] for x in groupby(f, lambda line: line.decode()))
    cpg_location_dict = {}
    count = 0
    for _ in faiter:
        count += 1
        try:
            seq = next(next(faiter)).decode().strip().split("\t")
            islands.append([int(seq[1]), int(seq[2])])
        except:
            break
    return islands

if __name__ == '__main__':
    print(read_islands("hg19.CpG-islands.bed.gz"))