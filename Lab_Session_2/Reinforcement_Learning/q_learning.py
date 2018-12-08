"""
Q_learn class that can be run from command line after tuning of Hyperparameters
and Tunable parameters
"""

## Dependencies
import numpy as np

from display import Displays 
from utils import Utils


class Q_learn(Displays, Utils):


	def __init__(self, rows, cols, actions, episodes, epsilon, alpha, gamma, iterations, agent_random_test,
		nn_layers, dropuout, q_learn_type, agent_dispalay, metrics_display, random_envi = False):
		super().__init__()

		self.rows = rows
		self.cols = cols
		self.actions = actions
		self.action_values = np.zeros((len(self.actions), self.rows, self.cols))
		self.episodes = episodes
		self.iterations = iterations
		self.random_envi = random_envi
		self.agent_dispalay = agent_dispalay
		self.metrics_display = metrics_display

		## Hyperparameters
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.nn_layers = nn_layers
		self.dropout = dropuout
		self.agent_random_test = agent_random_test

		if not self.random_envi:
			self.environment = [['.','.','.','.'], ['.','.','W','.'],['P','.','.','.'],['.','.','T','.']]
			self.grid_elements = {'.':[[0,0], [0,1], [0,2], [0,3], [1,0], [1,1], [1,3], [2,1], 
			[2,2], [2,3], [3,0], [3,1], [3,3]], 'W':[[1,2]], 'P':[[2,0]], 'T':[[3,2]]}

		else : self.grid_elements, self.environment = Utils.random_enviroment(self.rows ,self.cols)


		## calls the q_learning function
		if q_learn_type == 'Basic': 
			self.state_values, self.policy = self.__q_learning(grid_display = False)
		elif q_learn_type == 'Deep':
			self.state_values, self.policy = self.__deep__q_learning(grid_display = False)
	
		
	def __q_learning(self, grid_display = False):
		"""
		The method calculates the Q-table (Q-tensor in my case) from the defined 
		environment.

		Args:
		- grid_display (bool): Displays agent's movements on the grid if True

		Returns:
		- state_values (nd.array): an array specifying the value of each state 
									in the grid
		- policy (nd.array): an array based on the state_values specifying
							 action to be taken from each grid 
		"""

		## Intializations
		self.rewards = self._reward()
		self.possible_positions = self._angent_positions()

		## Displays training environment if it remains same over the episodes
		if not self.random_envi :
			print("The training environment\n")
			self.display_agent()

		for episode in range(self.episodes):

			## Begin episode on random cell on the grid (except 'W')
			state = self.possible_positions[np.random.choice(len(self.possible_positions))]

			for i in range(self.iterations):

				## Calculate reward and score at given state
				reward = self.rewards[state[0],state[1]]
				
				## If terminal state reached
				if state  == self.grid_elements['T'][0] or state == self.grid_elements['P'][0]:
				
					if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, None, None)
					self.action_values[:, state[0], state[1]] += self.alpha*(reward - 
						self.action_values[:, state[0], state[1]])
					state_values, policy = self._policy_and_values()
	
					break

				## Calculate action and new state
				action, greedy_action = self._epsilon_greedy(state)
				new_state = self._new_state(state, action)

				## Displays agent iteration if condition satisfied
				if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, action, greedy_action)

				## Calculate the action-value using action-value function
				target_value = reward + self.gamma*self._greedy_action_value(new_state)
				self.action_values[action, state[0], state[1]] += self.alpha*(target_value - 
					self.action_values[action, state[0], state[1]]) 

				## State update
				state = new_state

				## Calculate state value and policy with current Q-table
				state_values, policy = self._policy_and_values()
			

			## If condition statisfied the policy and value matrices will be displayed
			if episode % self.metrics_display == 0:
				print("The value and policy matrix after {} episodes:\n".format(episode))
				self.display_val_pol(state_values.round(1), policy)

			## Creates new random environment and re-initialize rewards, envi
			## possible_position if random_envi = True
			if self.random_envi:

				self.grid_elements, self.environment = Utils.random_enviroment(self.rows, self.cols)
				rewards = self._reward()
				possible_positions = self._angent_positions()
		

		return state_values, policy

              
              
	def __deep__q_learning(self, grid_display = False):

		"""
		The method calculates the Q-table (Q-tensor in my case) using the 
		a neural network, hence the name Deep Q learning 

		Args:
		- grid_display (bool): Displays agent's movements on the grid if True

		Returns:
		- state_values (nd.array): an array specifying the value of each state 
									in the grid
		- policy (nd.array): an array based on the state_values specifying
							 action to be taken from each grid 
		
		"""
	
		## Initializations
		replay_data = 0
		self.rewards = self._reward()
		self.possible_positions = self._angent_positions()
		mlp_model = self._neural_net()

		## Displays training environment if it remains same over the episodes
		if not self.random_envi :
			print("The training environment\n")
			self.display_agent()

		for episode in range(self.episodes):

			## Begin episode on random cell on the grid (except 'W')
			state = self.possible_positions[np.random.choice(len(self.possible_positions))]

			for i in range(self.iterations):
				
				temp_transition = 0

				## Calculate reward and score at given state
				reward = self.rewards[state[0],state[1]]
				
				## If terminal state reached
				if state  == self.grid_elements['T'][0] or state == self.grid_elements['P'][0]:
				
					if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, None, None)
					self.action_values[:, state[0], state[1]] += self.alpha*(reward - 
						self.action_values[:, state[0], state[1]])
					state_values, policy = self._policy_and_values()

					## Concatenating to the replay dataset 
					temp_transition = np.array([state[0], state[1], None, reward, state[0], state[1]]).reshape(1,-1)

					if episode == 0 and i == 0: replay_data = temp_transition
					else: replay_data = np.concatenate((replay_data, temp_transition))

					break

				## Calculate action and new state
				action, greedy_action = self._epsilon_greedy(state, model = mlp_model)
				new_state = self._new_state(state, action)

				## Displays agent iteration if condition satisfied
				if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, action, greedy_action)

				## Calculate the action-value using action-value function
				target_value = reward + self.gamma*self._greedy_action_value(new_state)
				self.action_values[action, state[0], state[1]] += self.alpha*(target_value - 
					self.action_values[action, state[0], state[1]]) 

				temp_transition = np.array([state[0], state[1], action, reward, new_state[0], new_state[1]]).reshape(1,-1)

				if episode == 0 and i == 0: replay_data = temp_transition
				else:
					replay_data = np.concatenate((replay_data, temp_transition))

				## State update
				state = new_state

				## Calculate state value and policy with current Q-table
				state_values, policy = self._policy_and_values()
				
				

			## Train model on data saved during iterations
			if len(replay_data) > 32:

				batch_x_train = replay_data[np.random.choice(len(replay_data), 32)] # Pick random batch from replay_data
				new_state_x_idx = [int(j) for j in batch_x_train[:,4].tolist()] 
				new_state_y_idx = [int(j) for j in batch_x_train[:,5].tolist()]
				max_next_state = np.max(self.action_values[:, new_state_x_idx, new_state_y_idx], axis = 0)
				y_target = batch_x_train[:,3] + self.gamma*max_next_state # reward + gamma*max(Q(t+1, a'))
				y_target = np.repeat(y_target.reshape(-1,1), len(self.actions), axis = 1)
				mlp_model.fit(batch_x_train[:,:2], y_target, epochs=3, batch_size=32, verbose=0)

			## If condition statisfied the policy and value matrices will be displayed
			if episode % self.metrics_display == 0:
				print("The value and policy matrix after {} episodes:\n".format(episode))
				self.display_val_pol(state_values.round(1), policy)


			## Creates new random environment and re-initialize rewards, envi
			## possible_position if random_envi = True
			if self.random_envi:

				self.grid_elements, self.environment = Utils.random_enviroment(self.rows, self.cols)
				rewards = self._reward()
				possible_positions = self._angent_positions()
		
		return state_values, policy



	def q_testing(self, test_episodes = 1000, grid_display = True):
		"""
		The method calculates the accuracy of the calculated q_table

		Args:
		- test_episodes (int): Number of episodes over which agents path will be traced
		- grid_display (bool): Displays agent's movements on the grid if True

		Prints:
		- Accuracy : (number of times agent reached treasure(T)) / episodes
		- Average steps : number of steps on average the agent took to reach treasure(T)
		"""
		print('##################  BEGIN TESTING ################### \n')

		
		## Create a random environment if 
		if self.random_envi:
			self.grid_elements, self.environment = Utils.random_enviroment(self.rows ,self.cols)

		## Intializations
		finish = 0
		steps = list()
		
		## Begin iterating over episodes of the game
		for episode in range(test_episodes):

			## Begin episode on random cell on the grid (except 'W') 
			state = self.possible_positions[np.random.choice(len(self.possible_positions))]
			
			## Begin iterations over enviroment until terminal or iteration limit reached
			for i in range(self.iterations):
			
				greedy_action = self.policy[state[0],state[1]]
				random_actions = [a for a in self.actions if a != greedy_action]
				action = np.random.choice([greedy_action, np.random.choice(random_actions)], 
					p = [1-self.agent_random_test , self.agent_random_test])
				
				state = self._new_state(state, action)
				
				## If terminal state reached
				if state == self.grid_elements['T'][0] or state == self.grid_elements['P'][0]:
					
					if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, None, None)
					if state == self.grid_elements['T'][0]: 
						finish += 1
						steps.append(i)
			
					break

				## Displays agent iteration if condition satisfied
				if episode % self.agent_dispalay == 0 and grid_display: self.display_agent(state, greedy_action, action)

		
		accuracy = finish/test_episodes
		avg_steps = round(sum(steps)/(len(steps) + 0.0000001))
		print("Accuracy is {}".format(accuracy))
		if len(steps) > 0: print("Average steps taken to finish is {}".format(avg_steps))

		return accuracy, avg_steps
		




def main():

	
	q_learn = Q_learn(rows, cols, actions, episodes, epsilon, alpha, gamma, iterations, 
		agent_random_test, nn_layers, dropout, q_learn_type, agent_dispalay, metrics_display, random_envi)
	accuracy, avg_steps = q_learn.q_testing(test_episodes = 1000, grid_display = True)



if __name__ == '__main__':

	## Constants
	rows = 4
	cols = 4
	actions = [0, 1, 2, 3]
	
	## Hyperparameters
	epsilon = 0.2
	gamma = 1
	alpha = 0.5
	agent_random_test = 0 # Probability with which agent moves randomly during test time
	dropout = 0.9 # Dropuout for the frist hidden layer of the MLP (ignore if q_learn_type = "Basic")
	nn_layers = [8,4] # Number of neurons on each hidden layer of the MLP (ignore if q_learn_type = "Basic")

	## Tunable parameters
	q_learn_type = "Basic" # Basic or Deep
	random_envi = False  # True or False : True changes environment every episode
	episodes = 101 # Number of episodes 
	iterations = 100 # Number of iterations for each episode
	agent_dispalay = 50 # Frequencey with which agent's movements within enviroment are displayed
	metrics_display = 50 # Frequence with which the policy and the state-value matrices are displayed
	

	main()

