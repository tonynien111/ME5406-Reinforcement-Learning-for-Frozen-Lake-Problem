import random

class QLearning:
    # q-learning algorithm 
    
    def __init__(self, num_states, num_actions, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        # initialize q-learning agent
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # generate q-table
        self.Q = {}
        for state in range(num_states):
            self.Q[state] = [0.0] * num_actions
    
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
    
    def update(self, state, action, reward, next_state):
        # update q-value using q-learning update rule
        # different to sarsa: use max action at next state
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state])

        # same update formula structure as sarsa
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q
    
    def train(self, env, num_episodes):
        # train q-learning method in the same environment
        stats = {
            'episodes': [],
            'success_count': [],
            'total_steps': [],
            'total_reward': []
        }
        
        success_count = 0
        
        for episode_num in range(num_episodes):
            state = env.reset()
            cumulative_reward = 0
            steps = 0
            
            while True:
                # Select and execute action
                action = self.epsilon_greedy_policy(state)
                next_state, reward, terminated, truncated, success = env.step(action)
                
                # Update Q-value
                self.update(state, action, reward, next_state)
                
                cumulative_reward += reward
                steps += 1 
                if terminated or truncated:
                    break
                state = next_state
            if success:
                success_count += 1
            
            # save statistics for this episode
            stats['episodes'].append(episode_num)
            stats['success_count'].append(success_count)
            stats['total_steps'].append(steps)
            stats['total_reward'].append(cumulative_reward)
        
        return stats
    
    def get_q_table(self):
        return self.Q # get q-table(help function)
