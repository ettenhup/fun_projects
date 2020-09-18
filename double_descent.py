import sys
import matplotlib.pyplot as plt
from polynomial import *

N = 10 #number of points
d = 50 #degree of the polynomial

c_sample, x_sample, y_sample = getRandomPoints(N,d)
x_lin = np.linspace(min(x_sample), max(x_sample), 100)
y_lin = getPolynomial(x_lin, c_sample, d)

c_svd = fitWithSVD(x_sample, y_sample, d)   
y_svd = getPolynomial(x_lin, c_svd, d)
r_svd = getResidual(x_sample, y_sample, c_svd, d)

max_iter = 1000
lr = 0.01
c_sgd = fitWithSGD(x_sample, y_sample, d, gamma=lr, max_iter = max_iter)   
y_sgd = getPolynomial(x_lin, c_sgd, d)
r_sgd = getResidual(x_sample, y_sample, c_sgd, d)

c_cr = fitWithCR(x_sample, y_sample, d, conv_thresh=0.007)
y_cr = getPolynomial(x_lin, c_cr, d)
r_cr = getResidual(x_sample, y_sample, c_cr, d)

plt.scatter(x_sample, y_sample, color="r")
#plt.plot(x_lin, y_lin, color="r", label="Actual function")
plt.plot(x_lin, y_svd, color="b", label="SVD R2="+str(r_svd**2))
plt.plot(x_lin, y_sgd, color="g", label="GD (lr "+str(lr)+", #iter"+str(max_iter)+") R2="+str(r_sgd**2))
plt.plot(x_lin, y_cr, color="y", label="CROP R2="+str(r_cr**2))
plt.legend()
plt.tight_layout()
plt.show()
