import os
import csv
import itertools

from environment import FrozenLakeEnv
from monte_carlo import MonteCarlo
from sarsa import SARSA
from q_learning import QLearning
from parameter import (
    GRID_SIZE, NUM_EPISODES, LEARNING_RATE_VALUES, GAMMA_VALUES, EPSILON_VALUES,
    MONTE_CARLO_TUNE_RESULTS, SARSA_TUNE_RESULTS, QLEARNING_TUNE_RESULTS
)


def tune_algorithm(agent_class, param_values, results_dir, results_csv_name, algo_name, param_names):
    results = []
    
    for params in itertools.product(*param_values):
        param_dict = dict(zip(param_names, params))
        
        env = FrozenLakeEnv(grid_size=GRID_SIZE)
        agent = agent_class(num_states=env.total_states, num_actions=env.num_actions, **param_dict)
        stats = agent.train(env, NUM_EPISODES)
        
        # Get final episode metrics
        final_steps = stats['total_steps'][-1]# last row
        final_reward = stats['total_reward'][-1]
        
        # Calculate success rate
        success_count = stats['success_count'][-1]
        success_rate = success_count / NUM_EPISODES if NUM_EPISODES > 0 else 0
        
        # Find first goal episode
        first_goal_episode = -1
        if len(stats['success_count']) > 0 and stats['success_count'][0] > 0:
            first_goal_episode = 1
        else:
            for i in range(1, len(stats['success_count'])):
                if stats['success_count'][i] > stats['success_count'][i-1]:
                    first_goal_episode = i + 1
                    break
        
        result_dict = dict(param_dict)
        result_dict.update({
            'first_goal_episode': first_goal_episode,
            'success_rate': success_rate,
            'final_reward': final_reward,
            'final_steps': final_steps
        })
        results.append(result_dict)
    
    grid_dir = os.path.join(results_dir, f"{GRID_SIZE}x{GRID_SIZE}")
    os.makedirs(grid_dir, exist_ok=True)
    
    # Save results
    csv_path = os.path.join(grid_dir, f"{results_csv_name}_{GRID_SIZE}x{GRID_SIZE}.csv")
    fieldnames = param_names + ['first_goal_episode', 'success_rate', 'final_reward', 'final_steps']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Find best configuration by success rate
    best = max(results, key=lambda x: x['success_rate'])
    return best

# main function to tune hyperparamters for all algorithms and save best hyperparameters to csv file
if __name__ == '__main__':
    best_configs = []
    
    # Tune Monte Carlo
    best_mc = tune_algorithm(MonteCarlo, [GAMMA_VALUES, EPSILON_VALUES], MONTE_CARLO_TUNE_RESULTS, 'monte_carlo_tuning', 'Monte Carlo', ['gamma', 'epsilon'])
    best_mc['algorithm'] = 'Monte Carlo'
    best_configs.append(best_mc)
    
    # Tune SARSA
    best_sarsa = tune_algorithm(SARSA, [LEARNING_RATE_VALUES, GAMMA_VALUES, EPSILON_VALUES], SARSA_TUNE_RESULTS, 'sarsa_tuning', 'SARSA', ['learning_rate', 'gamma', 'epsilon'])
    best_sarsa['algorithm'] = 'SARSA'
    best_configs.append(best_sarsa)
    
    # Tune Q-Learning
    best_qlearning = tune_algorithm(QLearning, [LEARNING_RATE_VALUES, GAMMA_VALUES, EPSILON_VALUES], QLEARNING_TUNE_RESULTS, 'q_learning_tuning', 'Q-Learning', ['learning_rate', 'gamma', 'epsilon'])
    best_qlearning['algorithm'] = 'Q-Learning'
    best_configs.append(best_qlearning)
    
    # Save best hyperparameters to CSV
    results_dir = os.path.join('results', 'best_hyperparameters') 
    os.makedirs(results_dir, exist_ok=True) # ensure directory exists
    best_csv_path = os.path.join(results_dir, f'best_hyperparameters_{GRID_SIZE}x{GRID_SIZE}.csv') # save best hyperparameters for all algorithms in csv file
    
    # Organize fieldnames: algorithm, hyperparameters, then metrics
    hyperparams = ['learning_rate', 'gamma', 'epsilon']
    metrics = ['first_goal_episode', 'success_rate', 'final_reward', 'final_steps']
    fieldnames = ['algorithm'] + [p for p in hyperparams if any(p in config for config in best_configs)] + metrics
    
    with open(best_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_configs)
