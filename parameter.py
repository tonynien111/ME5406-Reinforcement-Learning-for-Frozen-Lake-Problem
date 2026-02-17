import os

# environment parameters
GRID_SIZE = 10 # size of map
NUM_EPISODES = 8000  # number of episodes for training (2000 for 4x4, 8000 for 10x10)

# tuning parameter ranges
LEARNING_RATE_VALUES = [0.05, 0.1, 0.2, 0.3]  # Learning rates to test
GAMMA_VALUES = [0.8, 0.9, 0.95, 0.99]  # Discount factors to test
EPSILON_VALUES = [0.05, 0.1, 0.15, 0.2]  # Epsilon values to test

# results directories (tuning results only)
MONTE_CARLO_TUNE_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'monte_carlo_tuning')
SARSA_TUNE_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sarsa_tuning')
QLEARNING_TUNE_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'q_learning_tuning')
