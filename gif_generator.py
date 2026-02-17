import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from environment import FrozenLakeEnv
from parameter import GRID_SIZE


def load_qtable(algorithm_name, results_dir):
    grid_dir = os.path.join(results_dir, f"{GRID_SIZE}x{GRID_SIZE}")# get the directory for the specific grid size
    csv_path = os.path.join(grid_dir, f"{algorithm_name}_{GRID_SIZE}x{GRID_SIZE}_qtable.csv")# get the path for the q-table
    
    q_table = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = int(row['state'])
            actions = [float(row[f'action_{i}']) for i in range(4)]
            q_table[state] = actions
    
    return q_table


def select_action(state, q_table, epsilon=0.0): # select action using deterministic policy (epsilon=0))
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)
    else:
        q_values = q_table[state]
        max_q = max(q_values)
        best_actions = [i for i, v in enumerate(q_values) if v == max_q]
        return np.random.choice(best_actions)


def run_episode_with_frames(env, q_table): # run an episode and capture frames for animation using the best final q-table. 
    frames = []
    trajectory = []
    state = env.reset()
    
    while True:
        trajectory.append(env.agent_pos)
        frames.append({
            'agent_pos': env.agent_pos,
            'trajectory': list(trajectory)
        })
        
        action = select_action(state, q_table)
        next_state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
        
        state = next_state
    
    trajectory.append(env.agent_pos)
    frames.append({
        'agent_pos': env.agent_pos,
        'trajectory': list(trajectory)
    })
    
    return frames

def generate_gif(algorithm_name, results_dir, num_episodes=5, fps=2): # generate gif for the best final q-table of the specific algorithm
    q_table = load_qtable(algorithm_name, results_dir)
    env = FrozenLakeEnv(grid_size=GRID_SIZE)
    
    all_frames = []
    
    for _ in range(num_episodes):
        frames = run_episode_with_frames(env, q_table)
        all_frames.extend(frames)
        
        if _ < num_episodes - 1:
            for _ in range(fps):
                all_frames.append(frames[-1].copy())
    
    # Create gif using matplotlib animation 
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame_idx):
        ax.clear()
        frame = all_frames[frame_idx]
        agent_pos = frame['agent_pos']
        trajectory = frame.get('trajectory', [])
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pos = (i, j)
                
                if pos in env.holes:
                    color = '#3d5a80'
                    label = 'H'
                    text_color = 'white'
                elif pos == env.start_pos:
                    color = "#8fb7ce"
                    label = 'S'
                    text_color = 'black'
                elif pos == env.goal_pos:
                    color = '#e0fbfc'
                    label = 'G'
                    text_color = 'black'
                else:
                    color = '#f0f0f0'
                    label = 'F'
                    text_color = 'black'
                
                rect = patches.Rectangle((j, GRID_SIZE - 1 - i), 1, 1,
                                        linewidth=2, edgecolor='black',
                                        facecolor=color)
                ax.add_patch(rect)
                
                ax.text(j + 0.5, GRID_SIZE - 0.5 - i, label,
                       ha='center', va='center', fontsize=20,
                       fontweight='bold', color=text_color)
        
        if len(trajectory) > 1:
            traj_x = [pos[1] + 0.5 for pos in trajectory]
            traj_y = [GRID_SIZE - 0.5 - pos[0] for pos in trajectory]
            ax.plot(traj_x, traj_y, color='#f4a261', linewidth=3, 
                   alpha=0.7, linestyle='-', zorder=5)
        
        agent_x, agent_y = agent_pos
        agent_circle = patches.Circle((agent_y + 0.5, GRID_SIZE - 0.5 - agent_x),
                                     0.3, color='#ee6c4d', zorder=10)
        ax.add_patch(agent_circle)
        
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal')
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=len(all_frames), interval=1000/fps, repeat=True)
    
    grid_dir = os.path.join(results_dir, f"{GRID_SIZE}x{GRID_SIZE}")
    os.makedirs(grid_dir, exist_ok=True)
    gif_path = os.path.join(grid_dir, f"{algorithm_name}_{GRID_SIZE}x{GRID_SIZE}_demo.gif")
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close()
    return gif_path

if __name__ == '__main__':
    algorithms = [
        ('monte_carlo', 'results/monte_carlo'),
        ('sarsa', 'results/sarsa'),
        ('q_learning', 'results/q_learning')
    ]
    
    for algo_name, results_dir in algorithms:
        generate_gif(algo_name, results_dir, num_episodes=5, fps=2)