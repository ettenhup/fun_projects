import sys
import matplotlib.pyplot as plt
from polynomial import *

N = 10 #number of points
d = 50 #degree of the polynomial

c_sample, x_sample, y_sample = getRandomPoints(N,d, seed=N*d)
x_lin = np.linspace(min(x_sample), max(x_sample), 100)
y_lin = getPolynomial(x_lin, c_sample, d)

plt.title("SVD fits for polynomials of different degrees")
plt.scatter(x_sample, y_sample, color="r")
plt.plot(x_lin, y_lin, color="r", label="Actual function")

for dd in [5, 9, 10, 11, 15, 20, 30, 50]: 
   c_svd = fitWithSVD(x_sample, y_sample, dd)
   y_svd = getPolynomial(x_lin, c_svd, dd)
   plt.plot(x_lin, y_svd, label="degree "+str(dd))
   
plt.legend()
plt.tight_layout()
plt.show()
