import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
import cvxpy as cp
from matplotlib import style
from itertools import product
style.use('ggplot')


class SVDD(object):


	def __init__(self, X, Y, mode, gamma, degree, r, C):

		self.X = X
		self.Y = Y
		self.mode = mode
		self.gamma = gamma
		self.degree = degree
		self.r = r
		self.C = C

		self.kernel =  np.zeros((X.shape[0], X.shape[0]))

		self.alpha = self._svdd_implementation()
		self.center = np.matmul(self.alpha.value.T,self.X)
		self.support_vec_idx = self._support_vectors()
		self.radius = self._calc_radius()
	
	
	def _kernels(self, x_i, x_j):

		if self.mode == "rbf":
			return np.exp(-self.gamma * np.linalg.norm(x_i - x_j)**2)

		elif self.mode == "polynomial":
			return (self.gamma*np.dot(x_i,x_j) + self.r)**self.degree

		elif self.mode == "sigmoid":
			return np.tanh(self.gamma*np.dot(x_i,x_j) + self.r )


	def _kernel_matrix(self):

		"""Calculate the Kernel matrix"""

		for i in range(n_samples):
			for j in range(n_samples):
				self.kernel[i,j] = self._kernels(self.X[i],self.X[j])


	def _svdd_implementation(self):

		self._kernel_matrix()
		
		alpha = cp.Variable((n_samples,1))

		## Defining the constraints
		constraint_1 = [alpha[i] >= 0  for i in range(n_samples)]
		constraint_2 = [alpha[i] <= self.C   for i in range(n_samples)]
		constraint_3 = [np.ones((1,n_samples))*alpha == 1 ]
		constraints = constraint_1 + constraint_2 + constraint_3

		## Defining the Objective function 
		objective_1 = alpha.T*self.kernel.diagonal()
		objective_2 = cp.quad_form(alpha, cp.Parameter(shape=self.kernel.shape, value=self.kernel, PSD=True))
		objective = cp.Minimize(-(objective_1 - objective_2))

		prob = cp.Problem(objective, constraints)
		prob.solve()

		return alpha


	def _support_vectors(self):

		"""Calculate the support vectors"""

		support_vec_idx = list()
		for i in range(self.alpha.shape[0]):
			if self.alpha.value[i] > (0+0.0001) and self.alpha.value[i] < (self.C-0.0001):
				support_vec_idx.append(i)

		return support_vec_idx


	def _calc_radius(self):

		"""Calculate the radius"""

		radius = list()
		third_term = np.matmul(self.alpha.value.T,np.matmul(self.kernel, self.alpha.value))

		for k in self.support_vec_idx:
			temp_second_term = 0
			
			for i in range(n_samples):
				temp_second_term += self.alpha.value[i]*self.kernel[i,k]

			temp_radius = np.sqrt(self.kernel[k,k] - 2*temp_second_term + third_term)
			radius.append(temp_radius)

		radius.sort()
		return radius


if __name__ == '__main__':

	
	n_samples=200
	
	X,Y = make_blobs(n_samples =n_samples , centers=1, center_box=(-1.0, 1.0), 
		cluster_std=1.1, random_state=42)

	mode_list = [['polynomial', 1, 1, 1, 0.1,'Linear SVDD (C = 0.1)', '$X_1$','$X_2$'], 
	['polynomial',1, 1, 1, 0.8,'Linear SVDD (C = 0.8)', '$X_1$','$X_2$'],
	['polynomial', 1, 3, 1, 0.1,'Polynomial SVDD (C = 0.1)', '$X_1$','$X_2$'],
	['polynomial', 1, 5, 1, 0.8,'Polynomial SVDD (C = 0.8)', '$X_1$','$X_2$'],
	['rbf',0.1, 1, 1, 0.1, 'RBF SVDD (C = 0.1)', '$X_1$','$X_2$'], 
	['rbf', 0.8, 1, 1, 0.8, 'RBF SVDD (C = 0.8)', '$X_1$','$X_2$']]

	f, ax = plt.subplots(3,2, sharex='col', sharey='row', figsize=(15, 15))

	for idx, info in zip(product([0, 1,2], [0 ,1]), mode_list):

		mode, gamma, degree, r, C, title, x_axis, y_axis  = info
		svdd = SVDD(X, Y, mode, gamma, degree, r, C)
		alpha = svdd.alpha 
		center = svdd.center      
		support_vec_idx = svdd.support_vec_idx 
		radius = svdd.radius

		ax[idx[0], idx[1]].scatter(X[support_vec_idx,0], X[support_vec_idx,1], alpha=0.5, 
		s=300, linewidth=1, facecolors='none', edgecolors='black')
		ax[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c='blue', s=40)
		ax[idx[0], idx[1]].plot(center[0][0], center[0][1],'ro')
		ax[idx[0], idx[1]].set_title(title)
		ax[idx[0], idx[1]].set_title(title)
		ax[idx[0], idx[1]].text(min(X[:, 0]-1), max(X[:, 1]),'Gamma = {}, degree = {}, r = {}'.format(gamma, degree, r))

		if title == 'Linear SVDD (C = 0.1)' or title == 'Linear SVDD (C = 0.8)':


			ax[idx[0], idx[1]].set_xlim([-7, 7])
			ax[idx[0], idx[1]].set_ylim([-5, 7])
			circ_1 = plt.Circle((center[0][0], center[0][1]), radius= radius[0], color='g', fill = False)
			circ_2 = plt.Circle((center[0][0], center[0][1]), radius= radius[-1], color='g', fill = False)
			ax[idx[0], idx[1]].add_patch(circ_1)
			ax[idx[0], idx[1]].add_patch(circ_2)


	plt.savefig('SVDD_benchmark.png')  
	plt.show()