import sys
import random


separator="~"

def rotateBack(s):
    if separator in s:
        pos=s.index(separator)
        return s[pos+1:len(s)]+s[0:pos+1] 
    else:
        return s
with open(sys.argv[1],'r') as f:
    orig = ""
    for line in f:
        orig+=line
orig += separator
sizeUncompressed = len(orig) # size in bytes = number of chars

rotations=[]
for i in range(len(orig)):
    rotation = orig[i+1:len(orig)]+orig[0:i+1]
    rotations.append(orig[i+1:len(orig)]+orig[0:i+1])
bw = ""
for rotation in sorted(rotations):
    bw+=rotation[-1]

sizeCompressed = 0
si = 0
steps = 0
while si<len(bw):
    steps += 1
    ss = bw[si]
    n = 1
    while(True):
        sj = si + n
        if(sj>=len(bw)-1):
            break
        st = bw[sj]
        if(ss!=st):
            break
        n+=1
    sizeCompressed += 4+1 # 4 bytes for an integer + 1 byte for a char
    si += n

nsymbols = 100
print("\nRebuild locally with "+str(nsymbols)+" symbols starting at pos:",end="")
numOccurrences = dict()
for symbol in sorted(set(bw)):
    n = 0
    for b in bw:
        if(b == symbol):
            n+=1
    numOccurrences[symbol] = n

while(True): # make sure we do not include the separator, otherwise a local rebuild does not make sense
    startpos = random.randint(0,len(orig)-1) 
    pos=startpos
    build = bw[pos]
    nelms=len(orig)-1 #one is the current position
    nelms=min(100,nelms) # number of elements to reconstruct
    for i in range(nelms):
        sym = bw[pos]
        offset = 0
        for s in bw[0:pos]:
            if s == sym:
                offset+=1
        pos = offset
        for symbol, nocc in numOccurrences.items():
            if symbol < sym:
                pos+=nocc
            else:
                break
        build = bw[pos] + build
    if separator not in build:
        break
print(""+str(startpos))
print(rotateBack(build))

print("Uncompressed: "+str(sizeUncompressed)+" compressed: "+str(sizeCompressed))
