#To generate a dataset in form of adjacency matrix run this file with an argument n to gen n by n adj Matrix

import random
import sys
size = int(sys.argv[1])
one_row = 0
print size
for i in range(10):
    for j in range(size):
        if j == size - 1:
            print 1,
        else:
            print 0,
    print ""
for i in range(10, size - 1):
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
for j in range(size):
    if j == size - 1:
        print 1,
    else:
        print 0,
