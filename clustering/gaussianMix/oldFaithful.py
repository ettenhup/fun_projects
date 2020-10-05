import os, sys
import numpy as np
iAm = os.path.realpath(__file__)
myPath = os.path.dirname(iAm)
sys.path.append(myPath)
from gaussianMix import *


names = []
Xread = []
with open(myPath+'/../old_faithful.txt','r') as f:
    for line in f:
        names.append(line.split()[0])
        Xread.append([float(a) for a in line.split()[1:]])
X = np.array(Xread)
#print(X)
X = normalizeColumns(X)

K=2

findClusters(X,K,plot='i')



