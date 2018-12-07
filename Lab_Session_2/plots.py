
## Dependencies
from q_learning import Q_learn 



def epsilon_accuracy():

	total_accuracy = list()
	steps = list()
	episodes = [50*i for i in range(11)]

	for episode in episodes:

		q_learn = Q_learn(rows, cols, actions, episode, epsilon, alpha, gamma, iterations, 
				nn_layers, dropout, q_learn_type, agent_dispalay, metrics_display, random_envi)
		accuracy, avg_steps = q_learn.q_testing(test_episodes = 1000, grid_display = False)

		total_accuracy.append()


def main():


	def epsilon_accuray()


	


if __name__ == '__main__':

	## Constants
	rows = 4
	cols = 4
	actions = [0, 1, 2, 3]
	
	## Hyperparameters
	epsilon = 0.1
	gamma = 1
	alpha = 0.5
	agent_random_test = 0.1 # Probability with which agent moves randomly during test time
	dropout = 0.9 # Dropuout for the frist hidden layer of the MLP (ignore if q_learn_type = "Basic")
	nn_layers = [8,4] # Number of neurons on each hidden layer of the MLP (ignore if q_learn_type = "Basic")

	## Tunable parameters
	q_learn_type = "Basic" # Basic or Deep
	random_envi = False  # True or False : True changes environment every episode
	#episodes = 11 # Number of episodes 
	iterations = 100 # Number of iterations for each episode
	agent_dispalay = 100 # Frequencey with which agent's movements within enviroment are displayed
	metrics_display = 100 # Frequence with which the policy and the state-value matrices are displayed
	

	main()

