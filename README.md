# ME5406 Reinforcement Learning - FrozenLake

Implementation of three RL algorithms on the FrozenLake environment:
- Monte Carlo
- SARSA
- Q-Learning

## Install Reqruied Package

```bash
pip install -r requirements.txt
```

## Training Step

### 0. Configure Parameters
Edit `parameter.py` to set the hyperparameters and environment settings to your desired values:

- `GRID_SIZE` - Grid size (4 or 10)
- `NUM_EPISODES` - Number of training episodes
- `LEARNING_RATE_VALUES` - The list of learning rate values to tune
- `GAMMA_VALUES` - The list of discount factor values to tune
- `EPSILON_VALUES` - The list of exploration rate values to tune

### 1. Hyperparameter Tuning
Perform grid search to find the best hyperparameters for all three algorithms:
```bash
python tune_hyperparameters.py
```

Outputs:
**Best Hyperparameters:**
- 4x4: `results/best_hyperparameters/best_hyperparameters_4x4.csv`
- 10x10: `results/best_hyperparameters/best_hyperparameters_10x10.csv`

### 2. Algorithm Comparison
Train all algorithms with best hyperparameters and generate comparison plots in term of average rewards, total rewards and total steps across the training episode:

```bash
python algorithm_comparison.py
```

Outputs:

**Q-tables:**
- Monte Carlo: `results/monte_carlo/{grid_size}/monte_carlo_{grid_size}_qtable.csv`
- SARSA: `results/sarsa/{grid_size}/sarsa_{grid_size}_qtable.csv`
- Q-Learning: `results/q_learning/{grid_size}/q_learning_{grid_size}_qtable.csv`

**Summary Statistics:**
- 4x4: `results/comparison/comparison_summary_4x4.csv`
- 10x10: `results/comparison/comparison_summary_10x10.csv`

**Comparison Plots:**

**4x4 Environment:**
![Total Steps 4x4](results/comparison/comparison_total_steps_4x4x4.png)
![Total Reward 4x4](results/comparison/comparison_total_reward_4x4x4.png)
![Average Reward 4x4](results/comparison/comparison_average_reward_4x4x4.png)

**10x10 Environment:**
![Total Steps 10x10](results/comparison/comparison_total_steps_10x10x10.png)
![Total Reward 10x10](results/comparison/comparison_total_reward_10x10x10.png)
![Average Reward 10x10](results/comparison/comparison_average_reward_10x10x10.png)


### 3. Generate Visualizations
Create GIF animations showing agent movement for different algorithm
```bash
python gif_generator.py
```
Saves GIFs to `results/{algorithm}/{grid_size}/`

#### Monte Carlo

**4x4 Environment:**
![Monte Carlo 4x4](results/monte_carlo/4x4/monte_carlo_4x4_demo.gif)

**10x10 Environment:**
![Monte Carlo 10x10](results/monte_carlo/10x10/monte_carlo_10x10_demo.gif)

#### SARSA

**4x4 Environment:**
![SARSA 4x4](results/sarsa/4x4/sarsa_4x4_demo.gif)

**10x10 Environment:**
![SARSA 10x10](results/sarsa/10x10/sarsa_10x10_demo.gif)

#### Q-Learning

**4x4 Environment:**
![Q-Learning 4x4](results/q_learning/4x4/q_learning_4x4_demo.gif)

**10x10 Environment:**
![Q-Learning 10x10](results/q_learning/10x10/q_learning_10x10_demo.gif)


## Project Structure

```
frozen_world/
├── environment/          # FrozenLake environment
├── monte_carlo/          # Monte Carlo algorithm
├── sarsa/                # SARSA algorithm
├── q_learning/           # Q-Learning algorithm
├── map/                  # Fixed Grid maps (4x4, 10x10)
├── results/              # experiment data files
├── parameter.py          # Configuration
├── tune_hyperparameters.py
├── algorithm_comparison.py
└── gif_generator.py
```
