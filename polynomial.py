import numpy as np

def getPolynomial(x, c, d):
   return np.array([sum(c[degree] * xx**degree for degree in range(d)) for xx in x])

def getResidual(x_sample, y_sample, c_calc, d):
   p = getPolynomial(x_sample, c_calc, d)
   return sum((yy_calc - yy_sample)**2 for yy_calc, yy_sample in zip(p, y_sample))**0.5

def getRandomPoints(N,d, seed=0):
   #Sample points from a noisy polynomial distribution
   np.random.seed(seed)
   c_sample = [(1 if np.random.random() > 0.5 else -1) * np.random.random() for _ in range(d)]
   x_sample = [(1 if np.random.random() > 0.5 else -1) * np.random.random() * 1 for _ in range(N)]
   y_no_error = [sum(c_sample[degree] * x_sample[sample]**degree for degree in range(d)) for sample in range(N)]
   y_sample = [ yy + np.random.normal() for yy in y_no_error ]
   return c_sample, x_sample, y_sample


def fitWithSVD(x_sample, y_sample, d):
   N = len(x_sample)
   X = np.array([[x_sample[sample]**degree for degree in range(d)] for sample in range(N)])
   u, s, vh = np.linalg.svd(X, full_matrices=False)
   rcond = 0
   s_inv = np.diag([ 1/sing if sing > max(s)*rcond else 0 for sing in s ])
   c_svd = np.dot(np.dot(vh.transpose(), np.dot(s_inv, u.transpose())), y_sample)
   return c_svd

#Fit with SGD
def fitWithSGD(x_sample, y_sample,d , conv_thresh = 0.007, gamma = 0.05, max_iter = 1000000):
   c_sgd = np.array([0.0 for _ in range(d)])
   N = len(x_sample)
   X = np.array([[x_sample[sample]**degree for degree in range(d)] for sample in range(N)])
   it = 0
   last_res_norm = float("inf")
   while it < max_iter :
      c_bef = c_sgd
      res = y_sample - np.dot(X, c_sgd) # negative gradient
      res_norm = sum(r**2 for r in res)**0.5
      if res_norm < conv_thresh or abs(last_res_norm - res_norm) < conv_thresh:
         break
      last_res_norm = res_norm
      c_sgd = c_sgd + gamma * np.dot(X.transpose(), res)
      it += 1
   return it, c_sgd

def solveLinSys(r_vecs, dim):
   B = [[ np.dot(r_vecs[iss1],r_vecs[iss2]) if iss1 < dim and iss2 < dim else 0.0 if iss1==iss2 else 1.0 for iss1 in range(dim+1)] for iss2 in range(dim+1)] 
   c = [0.0 if iss1 < dim else 1.0 for iss1 in range(dim+1)]
   return np.linalg.solve(B, c)

#Fit with CR
def fitWithCR(x_sample, y_sample, d, conv_thresh = 0.007, max_iter = 1000000, subspace = None):
   N = len(x_sample)
   X = np.array([[x_sample[sample]**degree for degree in range(d)] for sample in range(N)])
   c_cr = np.array([0.0 for _ in range(d)])
   subspace = max_iter if subspace == None else subspace
   it = 0
   reset_iter = 0
   c_vecs = [c_cr if ss==0 else [] for ss in range(subspace)]
   r_vecs = [[] for ss in range(subspace)]
   last_res_norm = float("inf")
   while it < max_iter:
      c_bef = c_cr
      res = np.dot(X.transpose(), np.dot(X, c_cr) - y_sample)
      res_norm = sum(r**2 for r in res)**0.5
      print(res_norm)
      if res_norm < conv_thresh or abs(last_res_norm - res_norm) < conv_thresh:
         break
      last_res_norm = res_norm
      r_vecs[reset_iter%subspace] = res
      dim = min(reset_iter+1,subspace)
      try:
         w = solveLinSys(r_vecs, dim)
      except:
         reset_iter = 0
         c_vecs = [c_cr if ss==0 else [] for ss in range(subspace)]
         r_vecs = [res  if ss==0 else [] for ss in range(subspace)]
         dim = min(reset_iter+1, subspace)
         w = solveLinSys(r_vecs, dim)
      r_opt = w[reset_iter%subspace] * r_vecs[reset_iter%subspace]
      c_opt = w[reset_iter%subspace] * c_vecs[reset_iter%subspace]
      for i in range(1,dim):
         r_opt += w[(reset_iter-i)%subspace] * r_vecs[(reset_iter-i)%subspace]
         c_opt += w[(reset_iter-i)%subspace] * c_vecs[(reset_iter-i)%subspace]
      r_vecs[reset_iter%subspace]=r_opt
      c_vecs[reset_iter%subspace]=c_opt
      c_cr = c_opt + r_opt
      it += 1
      reset_iter += 1
      c_vecs[reset_iter%subspace]=c_cr
   return it, c_cr 
