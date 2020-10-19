import sys
import matplotlib.pyplot as plt
from polynomial import *

N = 10 #number of points
d = 50 #degree of the polynomial
c_sample, x_sample, y_sample = getRandomPoints(N,d, seed=N*d)
x_lin = np.linspace(min(x_sample), max(x_sample), 100)
y_lin = getPolynomial(x_lin, c_sample, d)

max_iter = 1000000
lr = 0.01
thr = 1e-4

dd=110
c_svd = fitWithSVD(x_sample, y_sample, dd)
y_svd = getPolynomial(x_lin, c_svd, dd)
r_svd = getResidual(x_sample, y_sample, c_svd, dd)

X = np.array([[x_sample[sample]**degree for degree in range(dd)] for sample in range(N)])
guess = np.array([0.1 for _ in range(dd)])

g = lambda c : np.dot(X.transpose(), np.dot(X, c) - y_sample)
n_sgd, c_sgd = fitWithSGD(guess=guess, gradient=g, gamma=lr, max_iter=max_iter, conv_thresh=thr)
y_sgd = getPolynomial(x_lin, c_sgd, dd)
r_sgd = getResidual(x_sample, y_sample, c_sgd, dd)**2

r = lambda c : np.dot(X.transpose(), y_sample - np.dot(X, c))
n_cr, c_cr = fitWithCR(guess=guess, residual=r, conv_thresh=thr, subspace=None)
y_cr = getPolynomial(x_lin, c_cr, dd)
r_cr = getResidual(x_sample, y_sample, c_cr, dd)**2
del X

plt.figure()
plt.title("Degree "+str(dd)+" polynomial")
plt.scatter(x_sample, y_sample, color="r")
plt.plot(x_lin, y_lin, color="r", label="Actual function")
plt.plot(x_lin, y_svd, color="b", label="SVD R²="+str(r_svd**2))
plt.plot(x_lin, y_sgd, color="g", label="GD (#iter "+str(n_sgd)+") R²="+str(r_sgd**2))
plt.plot(x_lin, y_cr, color="y", label="CROP (#iter "+str(n_cr)+") R²="+str(r_cr**2))
plt.legend()
plt.tight_layout()
plt.show()
