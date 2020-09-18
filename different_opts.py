import sys
import matplotlib.pyplot as plt
from polynomial import *

N = 10 #number of points
d = 50 #degree of the polynomial
c_sample, x_sample, y_sample = getRandomPoints(N,d, seed=N*d)
x_lin = np.linspace(min(x_sample), max(x_sample), 100)
y_lin = getPolynomial(x_lin, c_sample, d)

for dd in [5, 9, 10, 11, 15, 20, 30, 50]: 
   c_svd = fitWithSVD(x_sample, y_sample, dd)   
   y_svd = getPolynomial(x_lin, c_svd, dd)
   r_svd = getResidual(x_sample, y_sample, c_svd, dd)
   
   max_iter = 1000000
   lr = 0.01
   n_sgd, c_sgd = fitWithSGD(x_sample, y_sample, dd, gamma=lr, max_iter = max_iter, conv_thresh = 0.0000001)
   y_sgd = getPolynomial(x_lin, c_sgd, dd)
   r_sgd = getResidual(x_sample, y_sample, c_sgd, dd)
   
   n_cr, c_cr = fitWithCR(x_sample, y_sample, dd, conv_thresh=0.00000001, subspace=10)
   y_cr = getPolynomial(x_lin, c_cr, dd)
   r_cr = getResidual(x_sample, y_sample, c_cr, dd)
   
   plt.figure()
   plt.title("Degree "+str(dd)+" polynomial")
   plt.scatter(x_sample, y_sample, color="r")
   plt.plot(x_lin, y_lin, color="r", label="Actual function")
   plt.plot(x_lin, y_svd, color="b", label="SVD R2="+str(r_svd**2))
   plt.plot(x_lin, y_sgd, color="g", label="GD (#iter "+str(n_sgd)+") R2="+str(r_sgd**2))
   plt.plot(x_lin, y_cr, color="y", label="CROP (#iter "+str(n_cr)+") R2="+str(r_cr**2))
   plt.legend()
   plt.tight_layout()
plt.show()
