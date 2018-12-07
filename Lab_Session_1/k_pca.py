import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles ,make_moons
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import pandas as pd
from matplotlib import style

style.use('ggplot')

class Kernel_PCA(object):

	def __init__(self, X, Y, mode, gamma, degree, r): 

		self.X = X
		self.y = Y
		self.mode = mode
		self.gamma = gamma
		self.degree = degree
		self.r = r
		self.kernel = np.zeros((X.shape[0], X.shape[0]))
		self.X_pc = self._kpca_implementation()


	def _kernels(self, x_i, x_j, mode):

		if mode == "rbf":
			return np.exp(-self.gamma * np.linalg.norm(x_i - x_j)**2)

		elif mode == "polynomial":
			return (self.gamma*np.dot(x_i,x_j) + self.r)**self.degree

		elif mode == "sigmoid":
			return np.tanh(self.gamma*np.dot(x_i,x_j) + self.r )


	def _kernel_matrix(self):

		n_samples = self.X.shape[0]
		self.kernel = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				self.kernel[i,j] = self._kernels(self.X[i],self.X[j], self.mode)


	def _kpca_implementation(self, PC = 2):

		## Calculate the Kernel matrix 
		self._kernel_matrix()

		## Centering the symmetric  kernel matrix.
		N = self.kernel.shape[0]
		one_n = np.ones((N,N)) / N
		K_centered = self.kernel - one_n.dot(self.kernel) - self.kernel.dot(one_n) + one_n.dot(self.kernel).dot(one_n)

		# Caculating eigne values and eigen vectors
		eigvals, eigvecs = eigh(K_centered)

		# Calculating the new representation of X 
		X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,PC+1)))

		return X_pc

def main():

	## Benchmark datset - Make moons

	n_samples = 400
	X, Y = make_moons(n_samples=n_samples, random_state=1)
	reds = Y == 0
	blues = Y == 1

	plt.figure(figsize= (10,10))

	mode_list = [['polynomial',1, 1, 1, 'Original space - Linear K-PCA', '$X_1$','$X_2$'],
	['rbf',12, 1, 1, 'Projection by RBF K-PCA', 'Principal Component 1','Principal Component 2'], 
	['polynomial',2, 3, 1, 'Projection by Polynomial K-PCA', 'Principal Component 1','Principal Component 2'], 
	['sigmoid',1, 1, 10, 'Projection by Sigmoid K-PCA', 'Principal Component 1','Principal Component 2']]

	for n,i in enumerate(mode_list):

		mode, gamma, degree, r, title, x_axis, y_axis  = i
		kpca = Kernel_PCA(X, Y, mode, gamma, degree, r)
		X_pc = kpca.X_pc

		plt.subplot(2, 2, n+1)
		plt.title(title)
		plt.scatter(X_pc[reds, 0], X_pc[reds, 1], c="red", s=20)
		plt.scatter(X_pc[blues, 0], X_pc[blues, 1], c="blue",s=20)
		plt.xlabel(x_axis)
		plt.ylabel(y_axis)

	plt.savefig('K_PCA_benchmark.png')
	plt.show()
	

	## NUMERAI datset
	df = pd.read_csv('dataset/numerai_training_data.csv')
	df = df[df['era'] == 'era3']
	X = df.values[:,3:-1]
	Y = df.values[:,-1]
	Y=Y.astype('int')
	reds = Y == 0
	blues = Y == 1
	plt.figure(figsize= (15,15))
	mode_list = [['polynomial',1, 1, 1, 'Original space - Linear K-PCA', '$X_1$','$X_2$'], 
	['rbf',10, 1, 1, 'Projection by RBF K-PCA', 'Principal Component 1','Principal Component 2'], 
	['polynomial',7, 5, 1, 'Projection by Polynomial K-PCA', 'Principal Component 1','Principal Component 2'], 
	['sigmoid',4, 1, 2, 'Projection by Sigmoid K-PCA', 'Principal Component 1','Principal Component 2']]

	for n,i in enumerate(mode_list):

		mode, gamma, degree, r, title, x_axis, y_axis  = i
		kpca = Kernel_PCA(X, Y, mode, gamma, degree, r)
		X_pc = kpca.X_pc

		plt.subplot(2, 2, n+1)
		plt.title(title)
		plt.scatter(X_pc[reds, 0], X_pc[reds, 1], c="red", s=20)
		plt.scatter(X_pc[blues, 0], X_pc[blues, 1], c="blue",s=20)
		plt.text(min(X_pc[:, 0]), max(X_pc[:, 1]),'Gamma = {}, degree = {}, r = {}'.format(gamma, degree, r))
		plt.xlabel(x_axis)
		plt.ylabel(y_axis)

	plt.savefig('K_PCA_Numerai.png')
	plt.show()

if __name__ == '__main__':

	np.random.seed(0)

	main()



