import os
class FrozenLakeEnv:
    def __init__(self, grid_size=4):
        # initialize frozen lake environment
        self.grid_size = grid_size # size of the grid, 4 for 4x4, and 10 for 10x10 
        self.total_states = grid_size * grid_size # how many states in totals
        self.num_actions = 4  # up, down, left, right
        
        # Load the map from file
        self._load_map_from_file()
        
        # Set up agent's starting position
        self.agent_pos = self.start_pos # position of agent, initalize to start position 
        self.steps_taken = 0 # to track steps taken in current episode, and store total steps
        self.max_steps = grid_size * 10
    
    def _load_map_from_file(self):
        # Load map from map folder
        current_file_path = os.path.abspath(__file__) # get current file path
        current_directory = os.path.dirname(current_file_path) # get current directory 
        parent_directory = os.path.dirname(current_directory) # get parent directory
        
        map_filename = f'map_{self.grid_size}x{self.grid_size}.txt' # get map filename
        map_file_path = os.path.join(parent_directory, 'map', map_filename) # get map file path 
        
        with open(map_file_path, 'r') as file:
            all_lines = file.readlines()
        
        self.holes = set()# store hole position
        self.start_pos = None # get start position
        self.goal_pos = None # get goal position 
        
        for row_index, line in enumerate(all_lines): # read each line in the map file and get the position of hole, start and goal
            numbers_in_line = line.strip().split()
            for col_index, number_string in enumerate(numbers_in_line):
                number_value = int(number_string)
                current_position = (col_index, row_index) # x,y position of current cell
                if number_value == 1: # 1 stands for hole, 2 stands for start, and 3 stands for goal
                    self.holes.add(current_position)
                elif number_value == 2:
                    self.start_pos = current_position
                elif number_value == 3:
                    self.goal_pos = current_position
    
    def reset(self):
        # reset environment to initial state for new episode
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        initial_state = self._pos_to_state(self.agent_pos)
        return initial_state
    
    def _pos_to_state(self, pos):
        # convert position (x,y) to state number
        state_number = pos[1] + pos[0] * self.grid_size
        return state_number
    
    def step(self, action):
        # take one step in the environment
        current_x, current_y = self.agent_pos # get current position of the agent
        
        # Move based on action
        if action == 0:  # up
            if current_x > 0:
                   current_x -= 1
        elif action == 1:  # down
            if current_x < self.grid_size - 1:
                   current_x += 1
        elif action == 2:  # left
            if current_y > 0:
                   current_y -= 1
        elif action == 3:  # right
            if current_y < self.grid_size - 1:
                current_y += 1
        
        self.agent_pos = (current_x, current_y) # update agent position
        self.steps_taken += 1 # increment steps taken
        
        terminated = False # indicate if the episode is terminated (success or fall)
        truncated = False # indicate if the episode exceed max steps
        reward = 0 # store reward for this step 
        success = False # indicate if the episode is successful or not 

        # reward logic
        if self.agent_pos == self.goal_pos: # reach the goal, then terminated set to true
            reward = 1 # +1 reward for reaching the goal 
            terminated = True # episode ends when reach the goal 
            success = True # mark success as true when reach the goal 
        elif self.agent_pos in self.holes:
            reward = -1 # -1 reward for falling into hole
            terminated = True # episode ends when fall into hole
        elif self.steps_taken >= self.max_steps: # exceed max steps
            truncated = True # episode ends to aviod infinite loop
        
        next_state = self._pos_to_state(self.agent_pos) # convert new position to state number 
        return next_state, reward, terminated, truncated, success # return all necessary information in this step 
