import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles ,make_moons
import random

class Kernel_KMEANS(object):

	
	def __init__(self, X, Y, mode, gamma, degree, r, n_clusters): 

		self.X = X
		self.y = Y
		self.mode = mode
		self.gamma = gamma
		self.degree = degree
		self.r = r
		self.n_clusters = n_clusters
		
		self.stability = False
		self.cluster_dict = dict()
		self.kernel =  np.zeros((X.shape[0], X.shape[0]))

		self._kmeans_implementation()


	def _kernels(self, x_i, x_j):

		if self.mode == "rbf":
			return np.exp(-self.gamma * np.linalg.norm(x_i - x_j)**2)

		elif self.mode == "polynomial":
			return (self.gamma*np.dot(x_i,x_j) + self.r)**self.degree

		elif self.mode == "sigmoid":
			return np.tanh(self.gamma*np.dot(x_i,x_j) + self.r )

	
	def _kernel_matrix(self):

		n_samples = self.X.shape[0]
		for i in range(n_samples):
			for j in range(n_samples):
				self.kernel[i,j] = self._kernels(self.X[i],self.X[j])


	def _random_cluster_assgin(self):

		n_samples = self.X.shape[0]
		shuffled_idx = list(np.random.permutation(n_samples))
		for cluster in range(self.n_clusters): self.cluster_dict[cluster] = list()
		for i in range(n_samples):
			random_centroid = random.randint(0, self.n_clusters-1)
			self.cluster_dict[random_centroid].append((shuffled_idx[i],self.X[shuffled_idx[i]]))



	def _third_term_obj_function(self,idx_list):

		output = 0
		for j in idx_list:
			for l in idx_list:
				output += self.kernel[j,l]

		# if statement to avoid division by zero
		if idx_list: return output/(len(idx_list)**2)
		else: return output

	
	def _second_term_obj_function(self, idx_list, i):

		output = 0
		for j in idx_list:
			output += self.kernel[i,j]

		# if statement to avoid division by zero
		if idx_list: return 2*output/len(idx_list)
		else: return output


	def _reassign(self, distance_dict, n_samples):

		temp_cluster_dict = dict()
		for cluster in range(self.n_clusters): temp_cluster_dict[cluster] = list()
		
		for i in range(n_samples):
			temp_list = [distance_dict[cluster][i] for cluster in range(self.n_clusters)]
			temp_cluster_dict[temp_list.index(min(temp_list))].append((i,self.X[i]))

		self._check(temp_cluster_dict) # checking if stability has been attained
		self.cluster_dict =  temp_cluster_dict



	def _check(self, temp_cluster_dict):

		temp_1 = [self.cluster_dict[0][i][0] for i in range(len(self.cluster_dict[0]))]
		temp_2 = [temp_cluster_dict[0][i][0] for i in range(len(temp_cluster_dict[0]))]

		if temp_1 == temp_2: self.stability = True



	def _kmeans_implementation(self):

		
		self._random_cluster_assgin() # Randomly assign points to clusters
		self._kernel_matrix()
		n_samples = self.X.shape[0]
		check = 0 # Check on infinite iterations

		while not self.stability:

			## Caculating similarity/distance between points and clusters in new space
			
			distance_dict = dict()
			for cluster in range(self.n_clusters):

				distance_dict[cluster] = list()
				
				# Calculating 3rd term in objective function 
				temp_idx_list = [self.cluster_dict[cluster][i][0] for i in range(len(self.cluster_dict[cluster]))]
				third_term = self._third_term_obj_function(temp_idx_list)

				for i in range(n_samples):

					# Calculating 2nd term in objective function
					second_term = self._second_term_obj_function(temp_idx_list, i)
					
					# Calculating the distance of instance 'i' from cluster
					dist_i = self.kernel[i,i] - 2*second_term + third_term
					distance_dict[cluster].append(dist_i)

			
			self._reassign(distance_dict, n_samples)   
			
			check += 1
			if check == 20 : break




if __name__ == '__main__':

	np.random.seed(0)

	n_samples = 400
	n_clusters = 2
	X, Y = make_circles(n_samples=n_samples, factor=.3, noise=.05, random_state=42)
	#X, Y = make_moons(n_samples=n_samples, random_state=1)
	
	plt.figure(figsize= (10,10))
	
	mode_list = [['polynomial',1, 1, 1, 'Original space - Linear K-PCA', '$X_1$','$X_2$'],
	['rbf',4, 1, 1, 'Projection by RBF K-PCA', 'Principal Component 1','Principal Component 2'], 
	['polynomial',7, 4, 1, 'Projection by Polynomial K-PCA', 'Principal Component 1','Principal Component 2'], 
	['sigmoid',3, 1, 5, 'Projection by Sigmoid K-PCA', 'Principal Component 1','Principal Component 2']]


	for n,i in enumerate(mode_list):

		mode, gamma, degree, r, title, x_axis, y_axis  = i
		k_kmeans = Kernel_KMEANS(X, Y, mode, gamma, degree, r, n_clusters)
		cluster_dict = k_kmeans.cluster_dict
		Y_ = np.ones((n_samples,))
		temp_list = [cluster_dict[0][i][0] for i in range(len(cluster_dict[0]))]
		Y_[temp_list] = 0

		reds = Y_ == 0
		blues = Y_ == 1

		plt.subplot(2, 2, n+1)
		plt.title(title)
		plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20)
		plt.scatter(X[blues, 0], X[blues, 1], c="blue",s=20)
		plt.xlabel(x_axis)
		plt.ylabel(y_axis)

	plt.show()














