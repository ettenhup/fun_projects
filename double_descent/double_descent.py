import sys
import matplotlib.pyplot as plt
from polynomial import *

N = 10 #number of points
d = 50 #degree of the polynomial

c_sample, x_sample, y_sample = getRandomPoints(2*N,d, seed=N*d)
x_train = x_sample[:10]
x_test = x_sample[10:]
y_train = y_sample[:10]
y_test = y_sample[10:]
x_lin = np.linspace(min(x_sample), max(x_sample), 100)
y_lin = getPolynomial(x_lin, c_sample, d)

domain = [dd for dd in range(d+1)]
r_svd = list()
r_sgd5000 = list()
r_sgd = list()
r_cr = list()
r_svdtrain = list()
r_sgd5000train = list()
r_sgdtrain = list()
r_crtrain = list()

n_sgd5000l = list()
n_sgdl = list()
n_crl = list()

max_iter = 1000000
lr = 0.01
thr = 0.000001

for dd in domain: 
    c_svd = fitWithSVD(x_train, y_train, dd)   
    r_svd.append((getResidual(x_test, y_test, c_svd, dd))**2)
    r_svdtrain.append((getResidual(x_train, y_train, c_svd, dd))**2)
    
    n_sgd5000, c_sgd5000 = fitWithSGD(x_train, y_train, dd, gamma=lr, max_iter = 7500, conv_thresh=thr)
    r_sgd5000.append((getResidual(x_test, y_test, c_sgd5000, dd))**2)
    r_sgd5000train.append((getResidual(x_train, y_train, c_sgd5000, dd))**2)
    n_sgd5000l.append(n_sgd5000)
    
    n_sgd, c_sgd = fitWithSGD(x_train, y_train, dd, gamma=lr, max_iter = max_iter, conv_thresh=thr)
    r_sgd.append((getResidual(x_test, y_test, c_sgd, dd))**2)
    r_sgdtrain.append((getResidual(x_train, y_train, c_sgd, dd))**2)
    n_sgdl.append(n_sgd)
    
    n_cr, c_cr = fitWithCR(x_train, y_train, dd, conv_thresh=thr, subspace=10)
    r_cr.append((getResidual(x_test, y_test, c_cr, dd))**2)
    r_crtrain.append((getResidual(x_train, y_train, c_cr, dd))**2)
    n_crl.append(n_cr)
    
plt.title("Test set R²")
plt.plot([dd for dd in domain], r_svd, color="b", label="SVD")
plt.plot([dd for dd in domain], r_sgd5000, color="r", label="GD (max 7,500 it)")
plt.plot([dd for dd in domain], r_sgd, color="g", label="GD")
plt.plot([dd for dd in domain], r_cr, color="y", label="CR")
plt.xlabel("Degree of polynomial")
plt.ylabel("R²")
plt.legend()
plt.tight_layout()
plt.figure()
plt.title("Training set R²")
plt.plot([dd for dd in domain], r_svdtrain, color="b", label="SVD")
plt.plot([dd for dd in domain], r_sgd5000train, color="r", label="GD (max 7,500 it)")
plt.plot([dd for dd in domain], r_sgdtrain, color="g", label="GD")
plt.plot([dd for dd in domain], r_crtrain, color="y", label="CR")
plt.xlabel("Degree of polynomial")
plt.ylabel("R²")
plt.legend()
plt.tight_layout()
plt.figure()
plt.title("Iterations")
plt.plot([dd for dd in domain], n_sgd5000l, color="r", label="GD (max 7,500 it)")
plt.plot([dd for dd in domain], n_sgdl, color="g", label="GD")
plt.plot([dd for dd in domain], n_crl, color="y", label="CR")
plt.xlabel("Degree of polynomial")
plt.ylabel("#Iterations")
plt.legend()
plt.tight_layout()
plt.show()

print("SVD R2 ", sum(r_svd))
print("GD5000 R2 ", sum(r_sgd))
print("GD R2 ", sum(r_sgd))
print("CR R2 ", sum(r_cr))
