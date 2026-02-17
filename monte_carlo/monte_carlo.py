import random

class MonteCarlo:
    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1): # default value if not provided in parameters.py
        # initialize monte carlo agent
        self.num_states = num_states # number of states in the environment
        self.num_actions = num_actions # number of actions the agent can take
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        
        # Q-table
        self.Q = {} # initialize q-table as a dictionary
        for state in range(num_states):
            self.Q[state] = [0.0] * num_actions
        
        # Visit count for averaging returns
        self.returns = {} # initialize returns as a dictionary
        self.visit_count = {} # initialize visit count as a dictionary
        for state in range(num_states):
            self.returns[state] = [0.0] * num_actions
            self.visit_count[state] = [0] * num_actions
    
    def epsilon_greedy_policy(self, state):
        # select action using epsilon-greedy policy
        # input: current state
        # output: selected action
        if random.random() < self.epsilon: # if random number is less than epsilon, select a random action for exploration 
            return random.randint(0, self.num_actions - 1)
        else: # otherwise, select the action with the highest q-value for exploitation
            q_values = self.Q[state] # get q-value for current state
            max_q = max(q_values) # get maximum q-value 
            best_actions = [i for i, v in enumerate(q_values) if v == max_q] # find all actions with max q-value
            if len(best_actions) == 1:
                return best_actions[0]
            else:
                return random.choice(best_actions)
    
    def generate_episode(self, env):
        # complete one episode using current policy and store the trajetory
        episode = []
        state = env.reset() # reset environment at initial state
        
        while True:
            action = self.epsilon_greedy_policy(state) # calculate action using epsilon-greedy policy 
            next_state, reward, terminated, truncated, success = env.step(action) # take one step 
            
            episode.append((state, action, reward))# store the trajetory of this episode as a list of state, action and reward
            
            if terminated or truncated:
                break
            state = next_state
        return episode, success
    
    def train(self, env, num_episodes): # initalize the traning process with env and number of episodes
    
        stats = {
            'episodes': [],
            'success_count': [],
            'total_steps': [],
            'total_reward': []
        }
        
        success_count = 0
        
        for episode_num in range(num_episodes):
            # Generate episode
            episode, success = self.generate_episode(env)
            # update success count
            if success:
                success_count += 1
            
            # Calculate returns and update Q-values (first-visit)
            G = 0
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G # calculate 
                
                # Only update on first visit to state in this episode
                future_states = [s for s, a, r in episode[t+1:]] # get all future states in this episode
                if state not in future_states: # if current state is not in future state, update q-value 
                    self.visit_count[state][action] += 1
                    self.returns[state][action] += G
                    avg_return = self.returns[state][action] / self.visit_count[state][action]
                    self.Q[state][action] = avg_return
            
            # save statistics for this episode
            stats['episodes'].append(episode_num)
            stats['success_count'].append(success_count)
            stats['total_steps'].append(len(episode))
            stats['total_reward'].append(G)
        
        return stats
    
    def get_q_table(self):
        return self.Q # get q-table(help function)
