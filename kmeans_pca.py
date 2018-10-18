import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles ,make_moons


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
		kernel = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				kernel[i,j] = self._kernels(self.X[i],self.X[j], self.mode)

		return kernel


	def _random_cluster_assgin(self):

		n_samples = self.X.shape[0]
		cluster_dict = dict()
		shuffled_idx = list(np.random.permutation(n_samples))
		for cluster in range(self.clusters): cluster_dict[cluster] = list()
		for i in range(n_samples):
			random_centroid = random.randint(0, self.n_clusters-1)
			cluster_dict[random_centroid].append((shuffled_idx[i],X[shuffled_idx[i]]))

		return cluster_dict


	def _third_term_obj_function(self,idx_list, kernel):

		output = 0
		for j in idx_list:
			for l in idx_list:
				output += kernel[j,l]

		# if statement to avoid division by zero
		if idx_list: return output/(len(idx_list)**2)
		else: return output

	
	def _second_term(self, idx_list, kernel, sample):

		output = 0
		for j in idx_list:
			output += kernel[i,j]

		# if statement to avoid division by zero
		if idx_list: return 2*output/len(idx_list)
		else: return output


	def _reassign(self, dist_dict, n_samples, n_clusters, X):

		global cluster_dict, stability

		temp_cluster_dict = dict()
		for c in range(n_clusters): temp_cluster_dict[c] = list()
		
		for i in range(n_samples):
			temp_list = [dist_dict[c][i] for c in range(n_clusters)]
			temp_cluster_dict[temp_list.index(min(temp_list))].append((i,X[i]))

		self._check(temp_cluster_dict) # checking if stability has been attained

		return temp_cluster_dict



	def _check(self, temp_cluster_dict):

		global cluster_dict, stability
		temp_1 = [cluster_dict[0][i][0] for i in range(len(cluster_dict[0]))]
		temp_2 = [temp_cluster_dict[0][i][0] for i in range(len(temp_cluster_dict[0]))]

		if temp_1 == temp_2: self.stability = True




	def _kmeans_implementation(self):

		
		cluster_dict = self._random_cluster_assgin() # Randomly assign points to clusters
		kernel = self._kernel_matrix
		
		check = 0 # Check on infinite iterations

		while not self.stability:

			## Caculating similarity/distance between points and clusters in new space
			
			distance_dict = dict()
			for cluster in range(self.n_clusters):

				distance_dict[cluster] = list()
				
				# Calculating 3rd term in objective function 
				temp_idx_list = [cluster_dict[c][i][0] for i in range(len(cluster_dict[c]))]
				third_term = self._third_term_obj_function(temp_idx_list, kernel)

				for sample in range(n_samples):

					# Calculating 2nd term in objective function
					second_term = self._second_term(temp_idx_list, kernel, sample)
					
					# Calculating the distance of instance 'i' from cluster
					dist_i = kernel[i,i] - 2*second_term + third_term
					distance_dict[cluster].append(dist_i)

			
			cluster_dict = _reassign(dist_dict, n_samples, n_clusters, X)   
			
			check += 1
			if check == 20 : break




if __name__ == '__main__':
	
	n_samples = 400
	n_clusters = 2
	X, Y = make_circles(n_samples, factor= 0.3, noise= 0.05, random_state=42)
	df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))


















