import matplotlib.pyplot as plt
import numpy as np
import random
nInp = 40
nHid = 50
inp = [3, 3, 3, 4, 2, 9]
out = [[2, 5], [5, 7], [7], [1,2,3], [19, 18, 17], [19,5,7]]
#inp = [3]
#out = [[2, 5, 7]]
random.seed(10)
lr = 0.08
niter = 2000
with_bias = True
optL1 = False
lReg = 0.01
def actp(x):
    #p = 10
    #return p**x/(1+p**x)
    return 1 if x>0 else 0
def act(x):
    #p = 10
    #return np.log(1+p**x)/np.log(p)-1
    return x if x>0 else 0

print("Optimizing "+str((nInp + with_bias) * nHid + (nHid + with_bias) * (nInp))+" parameters")

def fill():
    return random.random() if bool(random.getrandbits(1)) else -random.random()

def fi(i,x):
    return np.exp(x[i])/ sum(np.exp(xj) for xj in x)

def f(x):
    return np.array([ fi(i,x) for i in range(len(x)) ])

def dfidxj(i,j,x):
    return fi(i,x) * ( (1 if i==j else 0) - fi(j,x))


#preprocess input
lastw = None
wdict = dict()
tdict = dict()
for word, context in sorted(zip(inp,out)):
    print(word, context)
    if word != lastw:
        if lastw != None:
            wdict[lastw] = tdict
            tdict = dict()
        lastw  = word
    for cword in context:
        if cword in tdict:
            tdict[cword] += 1
        else:
            tdict[cword] = 1
wdict[word] = tdict

#alternatives for defining the expected output
#x = np.array([[ 1 if i == ii else 0 for i in range(nInp)] for ii in inp])
#e = np.array([[ 1/len(ou) if i in ou else 0 for i in range(nInp) ] for ou in out])
#e = np.array([[ 1 if i in ou else 0 for i in range(nInp) ] for ou in out])
wlist = []
x = []
e = []
for word,contextdict in wdict.items():
    wlist.append(word)
    x.append([ 1 if i == word or i == nInp else 0 for i in range(nInp + with_bias)])
    e.append([contextdict[i]/sum(contextdict.values()) if i in contextdict else 0.0 for i in range(nInp)])

W = np.array([[ fill() for i in range(nInp + with_bias)] for j in range(nHid)])
C = np.array([[ fill() for i in range(nInp)] for j in range(nHid + with_bias)])
on = [ 0.0 for i in range(niter) ]
tn = [ 0.0 for i in range(niter) ]
onh = [ 0.0 for i in range(niter) ]
tnh = [ 0.0 for i in range(niter) ]

for it in range(niter-1):
    dfdW  = np.array([[0.0  for b in range(nInp + with_bias)] for a in range(nHid)])
    dfdC  = np.array([[0.0  for b in range(nInp)] for a in range(nHid + with_bias)])
    for ix, xx in enumerate(x):
        h = np.dot(W, xx)
        h = np.array([act(h[i]) if i<nHid else 1 for i in range(nHid + with_bias)])
        onh[it] += sum(np.abs(dd) for dd in h)
        tnh[it] += sum(dd**2 for dd in h)
        u = np.dot(C.transpose(), h)
        o = f(u)
        d = e[ix] - o
        on[it] += sum(np.abs(dd) for dd in d)
        tn[it] += sum(dd**2 for dd in d)
        fpija = np.array([[dfidxj(r,s,u) for r in range(nInp)] for s in range(nInp)])
        if optL1: #l1 update
            for r in range(nInp):
                if e[ix][r] > o[r]:
                    dfdW -= np.array([[ sum(fpija[r][s]*C[a][s]*actp(h[a]) for s in range(nInp)) if b in inp or b==nInp else 0.0 for b in range(nInp + with_bias)] for a in range(nHid)])
                    dfdC -= np.array([[fpija[r][s]*act(h[a]) for s in range(nInp)] for a in range(nHid + with_bias) ])
                if e[ix][r] < o[r]:
                    dfdW += np.array([[ sum(fpija[r][s]*C[a][s]*actp(h[a]) for s in range(nInp)) if b in inp or b==nInp else 0.0 for b in range(nInp + with_bias)] for a in range(nHid)])
                    dfdC += np.array([[fpija[r][s]*act(h[a]) for s in range(nInp)] for a in range(nHid + with_bias) ])
        else: #l2  update
            for r in range(nInp):
                # due to delta_sb there is no need to put this into the s loop
                dfdW += -2*(e[ix][r]-o[r]) * np.array([[ sum(fpija[r][s]*C[a][s]*actp(h[a]) for s in range(nInp)) if b in inp or b==nInp else 0.0 for b in range(nInp + with_bias)] for a in range(nHid)])
                dfdC += -2*(e[ix][r]-o[r]) * np.array([[fpija[r][b]*act(h[a]) for b in range(nInp)] for a in range(nHid + with_bias) ])
        #l1 regularization term
        if(lReg > 0.0):
            dfdW += np.array([[lReg * sum( (xx[b]*actp(h[j]) if h[j] > 0 else -xx[b]*actp(h[j])) if j == a  else 0.0 for j in range(nHid) ) for b in range(nInp + with_bias) ] for a in range(nHid) ]) 
        
    tn[it] = tn[it]**0.5
    tnh[it] = tnh[it]**0.5
    C = C - lr * dfdC
    W = W - lr * dfdW
it+=1
hsum = [ 0.0 for i in range(nHid + with_bias) ]
for ix, xx in enumerate(x):
    h = np.dot(W, xx)
    h = np.array([act(h[i]) if i<nHid else 1 for i in range(nHid + with_bias)])
    u = np.dot(C.transpose(), h)
    o = f(u)
    hsum += h
    print("oh element: "+str(wlist[ix])+" found freqs: " + " ".join( str(wlist[ix])+"({0:.4f})".format(cfreq/sum(wdict[wlist[ix]].values()))+": {0:.4f}".format(o[cword]) for cword,cfreq in wdict[wlist[ix]].items()))
    d = e[ix] - o
    on[it] += sum(np.abs(dd) for dd in d)
    tn[it] += sum(dd**2 for dd in d)
    onh[it] += sum(np.abs(dd) for dd in h)
    tnh[it] += sum(dd**2 for dd in h)
tn[it] = tn[it]**0.5
tnh[it] = tnh[it]**0.5
print("h avg: "+", ".join("{0:.5f}".format(hh/len(x)) for hh in hsum))

print("l1: "+str(on[-1])+" l2: "+str(tn[-1])+ " " +str(onh[-1]) + " " +str(tnh[-1]))
plt.plot([i for i in range(niter)],on,label="l1 opt:" +("l1" if optL1 else "l2"))
plt.plot([i for i in range(niter)],tn,label="l2 opt:" +("l1" if optL1 else "l2"))
plt.legend()
plt.figure()
plt.plot([i for i in range(niter)],onh,label="l1 opt:" +("l1" if optL1 else "l2"))
plt.plot([i for i in range(niter)],tnh,label="l2 opt:" +("l1" if optL1 else "l2"))
plt.legend()
plt.show()
