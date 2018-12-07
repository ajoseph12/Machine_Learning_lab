"""
Display class used by q_learning
"""

## Dependencies
import copy


class Displays(object):
	"""A simple text based GUI"""

	def __init__(self):
		
		self.action_dict = {0:'Left ', 1:'Right', 2:'Up   ', 3:'Down '}


	def display_agent(self, state = None, greedy_action = None, action = None):
		"""
		The method displays the agent's movement within the defined environemt

		Args:
		- state (tuple): current postion of the agent
		- action (int): the action an agnet wishes to take

		Prints: Agent environment
		"""

		a = copy.deepcopy(self.environment)
		if state: a[state[0]][state[1]] = 'A'

		if action == None and state: 
			print('Agent reached terminal state')

		elif action == greedy_action and state:
			print("Agent chose to move {}".format(self.action_dict[action]))
		
		elif action != greedy_action and state: 
			print("Agent chose to move {} but moved {} instead".format(self.action_dict[greedy_action], self.action_dict[action]))


		print(' {} | {} | {} | {}'.format(a[0][0], a[0][1], a[0][2], a[0][3]))
		print('---------------')
		print(' {} | {} | {} | {}'.format(a[1][0], a[1][1], a[1][2], a[1][3]))
		print('---------------')
		print(' {} | {} | {} | {}'.format(a[2][0], a[2][1], a[2][2], a[2][3]))
		print('---------------')
		print(' {} | {} | {} | {}'.format(a[3][0], a[3][1], a[3][2], a[3][3]))
		print('\n')


	def display_val_pol(self, value, policy):
		"""
		The method displays state values and policy for an iteration

		Args:
		- value (nd.array): a state value matrix based on the action values
		- policy (nd.array): a policy matrix based on the action values 

		Prints: State-value and Policy grid
		"""

		print(' -------Policy Matrix-------\t \t ----State Value Matrix----\n')
		print(' {} | {} | {} | {} \t \t {} | {} | {} | {}'.format(self.action_dict[policy[0][0]], 
			self.action_dict[policy[0][1]], self.action_dict[policy[0][2]], self.action_dict[policy[0][3]], 
			value[0][0], value[0][1], value[0][2], value[0][3]))
		#print('-'*len(str(value[0][0]))*6 +' \t \t --------------------------')
		print(' {} | {} | {} | {} \t \t {} | {} | {} | {}'.format(self.action_dict[policy[1][0]], 
			self.action_dict[policy[1][1]], self.action_dict[policy[1][2]], self.action_dict[policy[1][3]],
			value[1][0], value[1][1], value[1][2], value[1][3]))
		#print('------------------------------ \t \t --------------------------')
		print(' {} | {} | {} | {} \t \t {} | {} | {} | {}'.format(self.action_dict[policy[2][0]], 
			self.action_dict[policy[2][1]], self.action_dict[policy[2][2]], self.action_dict[policy[2][3]],
			value[2][0], value[2][1], value[2][2], value[2][3]))
		#print('------------------------------ \t \t --------------------------')
		print(' {} | {} | {} | {} \t \t {} | {} | {} | {} \n \n \n'.format(self.action_dict[policy[3][0]], 
			self.action_dict[policy[3][1]], self.action_dict[policy[3][2]], self.action_dict[policy[3][3]],
			value[3][0], value[3][1], value[3][2], value[3][3]))



def main():
	envi = Displays()
	envi.display_agent([1,2],0,1)


if __name__ == '__main__':
	environment = [['.','.','.','.'], ['.','.','W','.'],['P','.','.','.'],['.','.','T','.']]

	main()
