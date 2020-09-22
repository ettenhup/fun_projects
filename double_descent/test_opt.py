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
thr = 0.000001

dd=50
c_svd = fitWithSVD(x_sample, y_sample, dd)
y_svd = getPolynomial(x_lin, c_svd, dd)
r_svd = getResidual(x_sample, y_sample, c_svd, dd)

n_sgd, c_sgd = fitWithSGD(x_sample, y_sample, dd, gamma=lr, max_iter = max_iter, conv_thresh = thr)
y_sgd = getPolynomial(x_lin, c_sgd, dd)
r_sgd = getResidual(x_sample, y_sample, c_sgd, dd)**2

X = np.array([[x_sample[sample]**degree for degree in range(d)] for sample in range(N)])
r = lambda c: np.dot(X.transpose(), np.dot(X, c) - y_sample)
n_cr, c_cr = fitWithCR(guess=np.array([0.0 for _ in range(d)]), residual=r, conv_thresh=thr, subspace=None)
y_cr = getPolynomial(x_lin, c_cr, dd)
r_cr = getResidual(x_sample, y_sample, c_cr, dd)**2
del X

X2 = np.array([[x_sample[sample]**degree if sample < N else degree*(degree-1)*x_sample[sample-N]**(degree-2) for degree in range(d)] for sample in range(2*N)])
r = lambda c: np.dot(X2.transpose(), np.dot(X2, c) - np.concatenate((y_sample, [0.0 for _ in range(N)])))
n_cr2, c_cr2 = fitWithCR(guess=np.array([2.2 for _ in range(d)]), residual=r, conv_thresh=thr, subspace=None)
y_cr2 = getPolynomial(x_lin, c_cr2, dd)
r_cr2 = getResidual(x_sample, y_sample, c_cr2, dd)**2
del X2

plt.figure()
plt.title("Degree "+str(dd)+" polynomial")
plt.scatter(x_sample, y_sample, color="r")
plt.plot(x_lin, y_lin, color="r", label="Actual function")
plt.plot(x_lin, y_svd, color="b", label="SVD R²="+str(r_svd**2))
plt.plot(x_lin, y_sgd, color="g", label="GD (#iter "+str(n_sgd)+") R²="+str(r_sgd**2))
plt.plot(x_lin, y_cr, color="y", label="CROP (#iter "+str(n_cr)+") R²="+str(r_cr**2))
plt.plot(x_lin, y_cr2, color="m", label="CROP (#iter "+str(n_cr2)+") R²="+str(r_cr2**2))
plt.legend()
plt.tight_layout()
plt.show()
