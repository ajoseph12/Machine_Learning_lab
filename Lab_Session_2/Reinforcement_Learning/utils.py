"""
Utils class used by q_learning
"""

## Dependencies
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

from display import Displays


class Utils(object):
	"""Class with helper functions for q_learning script"""

	def __init__(self):
		pass


	def _reward(self):
		"""
		Creates and returns reward matrix based on grid_elements

		Returns:
		- reward_matrix (nd.array): a 4x4 matrix indicating rewards at every 
									state of teh grid
		"""

		_, rows, cols = self.action_values.shape
		reward_matrix = np.ones((self.rows, self.cols)) * -1
		reward_matrix[tuple(self.grid_elements['T'][0])] = 10
		reward_matrix[tuple(self.grid_elements['P'][0])] = -10
		reward_matrix[tuple(self.grid_elements['W'][0])] = 0

		return reward_matrix


	def _angent_positions(self):
		"""Method returns list of all possible grid cell positions 
		the agent can occupy"""

		agent_positions = list()
		for key in self.grid_elements.keys():
			if key != 'W' : agent_positions.extend(self.grid_elements[key])

		return agent_positions


	def _epsilon_greedy(self, state, model = False):
		"""
		Selects action based on epsilon greedy algo

		Args:
		- state (list): current state of the agent
		- model (keras.models.Sequential): used with Deep Q learning

		Returns: 
		- action (int): action agent should take based on epsilon greedy policy
		"""
		if model : state_action_values = model.predict(np.array([[state[0], state[1]]]))
		else : state_action_values = self.action_values[:,state[0], state[1]]

		greedy_action = np.argmax(state_action_values)
		other_actions = [action for action in self.actions if action != greedy_action]
		action = np.random.choice([greedy_action, np.random.choice(other_actions)], p = [1- self.epsilon, self.epsilon])
	
		return action, greedy_action


	def _check_move(self, i, j):
		"""
		Checks if the agent can take a given action
		
		Args:
		- i (int): row number of the agent's position
		- j (int): column number of the  agent's position

		Returns: True/False based on condition 
		"""

		if list((i, j)) in self.possible_positions: return True
		else : return False


	def _new_state(self, state, action):
		"""
		Calculates the new position of the agent given its action

		Args:
		- state (list): current state of the agent indicated by (row, col)
		- action (int): action the agent wishes to perform (0,1,2,3)

		Returns:
		- new_state (tuple): new state of the agent once action is taken 
		"""
	
		new_state = state

		if action == 0: # 0 = left
			if self._check_move(state[0], state[1] - 1): new_state = [state[0], state[1] - 1]

		elif action == 1: # 1 = right
			if self._check_move(state[0], state[1] + 1): new_state = [state[0], state[1] + 1]

		elif action == 2: # 2 = up
			if self._check_move(state[0] - 1, state[1]): new_state = [state[0] - 1, state[1]]

		elif action == 3: # 3 = down
			if self._check_move(state[0] + 1, state[1]): new_state = [state[0] + 1, state[1]]

		return new_state


	def _greedy_action_value(self, new_state, random = 0):
		"""
		Calculates new action value when in new state based on greedy approach
		
		Args:
		- new_state (tuple): state the agent is indicated by (row, col)

		Returns:
		- greedy_action_value (float): action value resulted from greedy behaviour of the agent
		"""

		state_action_values = self.action_values[:,new_state[0], new_state[1]]
		greedy_action = np.argmax(state_action_values)
		greedy_action_value = self.action_values[greedy_action, new_state[0], new_state[1]]

		return greedy_action_value


	def _policy_and_values(self):
		"""The method calculates the state values and the policy from the 
		action value tensor"""

		state_values = np.zeros((self.rows, self.cols))
		policy = np.zeros((self.rows, self.cols))

		for row in range(self.rows):
			for col in range(self.cols):

				state_action_values = self.action_values[:,row, col]
				state_values[row, col] = max(state_action_values)
				policy[row, col]= np.argmax(state_action_values)

		return state_values, policy


	def _neural_net(self, input_dim = 2):
		"""
		This method defines a simple neural network instance

		Args:
		- input_dim (int): input dimension of the data feed to the MLP

		Returns:
		- model (keras.models.Sequential): an instance of the created MLP 
		"""
	
		model = Sequential()
		model.add(Dense(units=self.nn_layers[0], activation='relu', input_dim= input_dim, 
			kernel_initializer='random_uniform'))
		model.add(Dropout(self.dropout))
		model.add(Dense(units=self.nn_layers[1], kernel_initializer='random_uniform'))
		model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

		return model


	@staticmethod
	def random_enviroment(rows, cols):
		"""
		Computes a random environment given its dimensions

		Args:
		- rows (int): Number of rows the grid wrold has
		- cols (int): Number of columns the grid world has

		Returns:
		- grid_elements (dict): A dictionary with information regarding 
								the position of different elements in the
								grid world
		- environment (list): A list representative of the environment

		"""

		indices = [(i,j) for i in range(rows) for j in range(cols)]
		shuffle_indices = np.random.permutation(indices)
		len_idx = len(shuffle_indices)
		
		grid_elements = {'.': shuffle_indices[:len_idx-3].tolist(), 'W': [shuffle_indices[len_idx-3].tolist()], 
		'P': [shuffle_indices[len_idx-2].tolist()], 'T': [shuffle_indices[len_idx-1].tolist()]}

		environment = [['' for _ in range(rows)] for _ in range(cols)]
		
		for key in grid_elements.keys():
			for i,j in grid_elements[key]:
				environment[i][j] = key

		return grid_elements, environment



