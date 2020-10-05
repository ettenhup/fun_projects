import numpy as np
import numpy.linalg as npl
from random import randrange,random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kmeans

def exp(a):
    return np.exp(a)
def log(a):
    return np.log(a) if a > 1e-13 else float("-inf")
def logsum(logX,logY):
    return logX if logY==float("-inf") else logY if logX==float("-inf") else logX+log(1+exp(logY-logX)) if logX>logY else logY+log(1+exp(logX-logY)) 

def loggauss(x,mu,si):
    d = len(x)
    logpref = -(d/2) * np.log(2*np.pi) - (1/2) * np.log(np.linalg.det(si))
    expf = -0.5 * np.dot(np.transpose(x-mu),np.dot(np.linalg.inv(si),x-mu))
    return logpref + expf

def getGamma(Pi,X,M,S):
    N = len(X)
    K = len(M)
    G = [ [ 0 for k in range(K) ] for n in range(N) ]
    for n,x in enumerate(X):
        a = float("-inf")
        for k in range(K):
            g = np.log(Pi[k]) + loggauss(x,M[k],S[k])
            a = logsum(g,a)
            G[n][k] = g
        for k in range(K):
            G[n][k] = np.exp(G[n][k]-a)
    return G
        
def getMean(G,X):
    K = len(G[0])
    D = len(X[0])
    M = [ np.array([0.0 for i in range(D)]) for k in range(K) ]
    for k in range(K):
        Nk = sum(G[n][k] for n in range(len(X)))
        for n,x in enumerate(X):
            M[k] += G[n][k] * x / Nk
    return M

def getSigma(G,X,M):
    # get the new means 
    K = len(M)
    D = len(M[0])
    Sigma = [ np.array([[ 0.0 for i in range(D)] for j in range(D)]) for k in range(K)]
    for k in range(K):
        Nk = sum(G[n][k] for n in range(len(X)))
        for n,x in enumerate(X):
            Sigma[k] += G[n][k] * (np.outer((x - M[k]),np.transpose(x - M[k]))) / Nk
    return Sigma

def getPi(G):
    K=len(G[0])
    N=len(G)
    Pi = [ 0.0 for k in range(K) ]
    for k in range(K):
        for n in range(N):
            Pi[k] += G[n][k] / N
    return Pi

def getStartingGuess(X,K,kmeansGuess=True):
    if(kmeansGuess):
        G,M = kmeans.findClusters(X,K)
        S = getSigma(G,X,M)
        Pi = getPi(G)
    else:
        Pi = [ random() for i in range(K) ]
        Pi = [ pi/sum(Pi) for pi in Pi ]
        N = len(X)
        D = len(X[0])
        
        M = np.array([X[randrange(len(X))]])
        d=computeDistanceDistribution(X,M)
        S = [[[1.0 if j==k else 0.0 for j in range(D)] for k in range(D) ] for i in range(K)]
        for i in range(K-1):
            c=X[np.random.choice(range(len(X)), 1, d)]
            M=np.vstack((M, c))
            d=computeDistanceDistribution(X, M)
    return M, S, Pi

def getLogLikelihood(X,Pi,M,S):
    ll = 0.0
    K = len(Pi)
    for n,x in enumerate(X):
        l = float("-inf")
        for k in range(K):
            l = logsum(l,np.log(Pi[k]) + loggauss(x,M[k],S[k]))
        ll += l
    return ll

def computeDistanceDistribution(data, clusterCenters):
    result=[]
    for x in data:
        d=[ npl.norm(x-c)**2 for c in clusterCenters ]
        result.append(min(d))
    return [ r / sum(result) for r in result ]

def findClusters(X,K,plot='n'):
    if plot not in [ 'n', 'i', 'f']:
        print("invalid option for argument plot: ",plot)
        exit()
        
    M,S,Pi = getStartingGuess(X,K,kmeansGuess=False)
    
    converged = False
    N = len(X)
    P = len(X[0])
    Jprev = 0.0
    niter = 0
    print( "### it\tJ\tΔJ")
    while(not converged):
        # E-step
        G = getGamma(Pi,X,M,S)
        
        # M-step
        M = getMean(G,X)
        S = getSigma(G,X,M)
        Pi = getPi(G)
        
        J = getLogLikelihood(X,Pi,M,S)
        
        if(plot == 'i'):
            plotData(X,M,G,S)
        
        Jdiff = abs(J-Jprev)
        print( "### "+str(niter)+"\t"+"{0:.2g}".format(J)+"\t"+"{0:.2g}".format(Jdiff))
        converged = Jdiff < 1.0e-13 or niter > 200
        Jprev = J
        niter += 1
        
    print ("Final associations:")
    for k in range(K):
        print(" "+str(k)+": ",end="")
        for n in range(N):
            if( G[n][k] == max(G[n])):
                print(str(n+1)+" ",end="")
        print("")
    if(plot in ['i','f']):
        plotData(X,M,G,S)
        plt.show()
    return G,M,S
        
def plotData(X,M,G,S):
    fig = plt.figure()
    N = len(X)
    P = len(X[0])
    K = len(M)
    if P>2:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    C = np.dot(X.transpose(),X)/(N-1)
    e,V = npl.eig(C)
    V = np.real(V)
    
    if( False ):
        #plot the first principal components
        XPCoA = np.dot(X,V)
        MPCoA = np.dot(M,V)
        xc = 0
        yc = 1
        zc = 2 if P>2 else -1
    else:
        #plot the dimensions which are "most colinear" with the principal components
        XPCoA = X
        MPCoA = M
        xc = np.argmax(V[0])
        yc = np.argmax(V[1])
        zc = np.argmax(V[2]) if P>2 else -1
        print("Dims: "+str(xc)+" "+str(yc)+" "+str(zc))
            
    

    
    cols = ['r','b','k','g','c','m','y']
    for k in range(K):
        c = cols[k] if k<len(cols) else random()
        x = [float(MPCoA[k][xc])]
        y = [float(MPCoA[k][yc])]
        z = [float(MPCoA[k][zc]) if P>2 else 0]
        if( P>2 ):
            ax.scatter(x,y,z,marker='x',c=c)
        else:
            ax.scatter(x,y,marker='x',c=c)
        x = []
        y = []
        z = []
        for n in range(N):
            if(G[n][k] == max(G[n])):
                x.append(float(XPCoA[n][xc]))
                y.append(float(XPCoA[n][yc]))
                z.append(float(XPCoA[n][zc]) if P>2 else 0 )
        if( P>2 ):
            ax.scatter(x,y,z,marker='o',c=c)
        else:
            ax.scatter(x,y,marker='o',c=c)
        
    
def centerColumns(X):
    N = len(X)
    P = len(X[0])
    mu = [ 0.0 for i in range(P) ]
    
    for p in range(P):
        for n in range(N):
            mu[p] += X[n][p] / N
    
    for n in range(N):
        for p in range(P):
            X[n][p] = (X[n][p] - mu[p])/mu[p]
    return X

def normalizeColumns(X):
    N = len(X)
    P = len(X[0])
    mu = [ 0.0 for i in range(P) ]
    si = [ 0.0 for i in range(P) ]
    
    for p in range(P):
        for n in range(N):
            mu[p] += X[n][p] / N
        for n in range(N):
            si[p] += (X[n][p] - mu[p])**2/(N-1)
        si[p] = (si[p])**(0.5)
    
    for n in range(N):
        for p in range(P):
            X[n][p] = (X[n][p] - mu[p])/si[p] 
    return X
