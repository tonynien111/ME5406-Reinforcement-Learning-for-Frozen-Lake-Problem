import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from monte_carlo import MonteCarlo
from sarsa import SARSA
from q_learning import QLearning
from parameter import GRID_SIZE, NUM_EPISODES


def read_best_hyperparameters():
    csv_path = os.path.join('results', 'best_hyperparameters', f'best_hyperparameters_{GRID_SIZE}x{GRID_SIZE}.csv')
    # read best hyperparameters from csd file
    
    best_params = {} #store best hyperparamter for each algorithm
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f) # read it 
        for row in reader:
            algorithm = row['algorithm']
            params = {}# store hyperparamter for current algorithm
            
            if 'learning_rate' in row and row['learning_rate']:
                params['learning_rate'] = float(row['learning_rate'])
            if 'gamma' in row and row['gamma']:
                params['gamma'] = float(row['gamma'])
            if 'epsilon' in row and row['epsilon']:
                params['epsilon'] = float(row['epsilon'])
            best_params[algorithm] = params
    
    return best_params


def train_algorithm(agent_class, params):
    env = FrozenLakeEnv(grid_size=GRID_SIZE) # create environment 
    agent = agent_class(num_states=env.total_states, num_actions=env.num_actions, **params)# create agent with the best hyperparamters
    stats = agent.train(env, NUM_EPISODES) # train the agent and get training statistics
    
    return agent, stats


def save_qtables(all_agents, algo_name_map): # save q-table for gif generator 
    for display_name, agent in all_agents.items():
        algo_name = algo_name_map[display_name]
        results_dir = os.path.join('results', algo_name, f'{GRID_SIZE}x{GRID_SIZE}')
        os.makedirs(results_dir, exist_ok=True)
        
        csv_path = os.path.join(results_dir, f'{algo_name}_{GRID_SIZE}x{GRID_SIZE}_qtable.csv')
        
        q_table = agent.get_q_table()
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['state', 'action_0', 'action_1', 'action_2', 'action_3']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for state, actions in q_table.items():
                row = {'state': state}
                for i, value in enumerate(actions):
                    row[f'action_{i}'] = value
                writer.writerow(row)


def save_summary_table(all_stats, results_dir):
    os.makedirs(results_dir, exist_ok=True) # create directory if it doesn't exist 
    
    csv_path = os.path.join(results_dir, f'comparison_summary_{GRID_SIZE}x{GRID_SIZE}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['algorithm', 'success_rate', 'first_goal_episode', 'final_total_steps', 'final_total_reward']
        writer = csv.DictWriter(f, fieldnames=fieldnames) # write a summary table comparing the performance
        writer.writeheader()
        
        for algorithm, statistics in all_stats.items():
            # Calculate success rate
            success_rate = statistics['success_count'][-1] / NUM_EPISODES if statistics['success_count'] else 0
            
            # Find first goal episode
            first_goal_episode = -1
            for i, count in enumerate(statistics['success_count']):
                if count > 0:
                    first_goal_episode = i
                    break
            
            # Get final metrics
            final_steps = statistics['total_steps'][-1] if statistics['total_steps'] else 0
            final_reward = statistics['total_reward'][-1] if statistics['total_reward'] else 0
            
            writer.writerow({
                'algorithm': algorithm,
                'success_rate': f"{success_rate:.4f}",
                'first_goal_episode': first_goal_episode,
                'final_total_steps': final_steps,
                'final_total_reward': f"{final_reward:.4f}"
            })


def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_metrics(all_stats, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data
    algorithms = list(all_stats.keys())
    colors = {'Monte Carlo': '#1f77b4', 'SARSA': '#ff7f0e', 'Q-Learning': '#2ca02c'}
    window_size = 10  # Moving average window size
    
    # Plot 1: total steps vs episode
    fig, ax = plt.subplots(figsize=(12, 6))
    for algorithm in algorithms:
        total_steps = np.array(all_stats[algorithm]['total_steps'])
        smoothed_steps = moving_average(total_steps, window_size)
        episodes = np.arange(len(smoothed_steps))
        ax.plot(episodes, smoothed_steps, label=algorithm, linewidth=2.5, color=colors.get(algorithm, None), alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Steps', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Comparison - Total Steps per Episode (Grid: {GRID_SIZE}x{GRID_SIZE})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_total_steps_{GRID_SIZE}x{GRID_SIZE}.png'), dpi=300)
    plt.close()
    
    # Plot 2: total reward vs episode
    fig, ax = plt.subplots(figsize=(12, 6))
    for algorithm in algorithms:
        total_reward = np.array(all_stats[algorithm]['total_reward'])
        smoothed_rewards = moving_average(total_reward, window_size)
        episodes = np.arange(len(smoothed_rewards))
        ax.plot(episodes, smoothed_rewards, label=algorithm, linewidth=2.5, color=colors.get(algorithm, None), alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Comparison - Total Reward per Episode (Grid: {GRID_SIZE}x{GRID_SIZE})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_total_reward_{GRID_SIZE}x{GRID_SIZE}.png'), dpi=300)
    plt.close()
    
    # Plot 3: average reward vs episode
    fig, ax = plt.subplots(figsize=(12, 6))
    for algorithm in algorithms:
        total_reward = np.array(all_stats[algorithm]['total_reward'])
        total_steps = np.array(all_stats[algorithm]['total_steps'])
        avg_reward = np.divide(total_reward, total_steps, where=total_steps!=0, out=np.zeros(len(total_steps)))
        smoothed_average_rewards = moving_average(avg_reward, window_size)
        episodes = np.arange(len(smoothed_average_rewards))
        ax.plot(episodes, smoothed_average_rewards, label=algorithm, linewidth=2.5, color=colors.get(algorithm, None), alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward per Step', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Comparison - Average Reward per Step (Grid: {GRID_SIZE}x{GRID_SIZE})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_average_reward_{GRID_SIZE}x{GRID_SIZE}.png'), dpi=300)
    plt.close()

# main function to train all algorithms with their best hyperparamters and plot comparsion graph 
if __name__ == '__main__':
    best_params = read_best_hyperparameters()
    
    all_stats = {}
    all_agents = {}
    algo_name_map = {
        'Monte Carlo': 'monte_carlo',
        'SARSA': 'sarsa',
        'Q-Learning': 'q_learning'
    }
    
    # Train all algorithms with their best hyperparameters
    for algo_name, agent_class in [('Monte Carlo', MonteCarlo), ('SARSA', SARSA), ('Q-Learning', QLearning)]:
        all_agents[algo_name], all_stats[algo_name] = train_algorithm(agent_class, best_params[algo_name])
    
    # Save Q-tables
    save_qtables(all_agents, algo_name_map)
    
    # Save summary table and plot comparison graphs
    results_dir = os.path.join('results', 'comparison')
    save_summary_table(all_stats, results_dir)
    plot_metrics(all_stats, results_dir)
