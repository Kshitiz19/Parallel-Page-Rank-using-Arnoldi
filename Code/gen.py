# Usage: python gen.py "sizeOfDataset" > "output-filename"
# size of dataset is a integer n that results into a n x n sparse adjacency matrix

import random
import sys
from tqdm import tqdm
size = int(sys.argv[1])
one_row = 0
print size
for i in tqdm(range(10)):
    for j in range(size):
        if j == size - 1:
            print 1,
        else:
            print 0,
    print ""
for i in tqdm(range(10, size - 1)):
    for j in range(size):
        if i == j:
            print 0,
        else:
            a = random.randint(1, 100)
            if a > 90:
                print "1",
            else:
                print "0",
    print ""
for j in tqdm(range(size)):
    if j == size - 1:
        print 1,
    else:
        print 0,
