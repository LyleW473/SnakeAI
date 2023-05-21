import torch
import random
import numpy as np
from collections import deque
from model import Linear_Qnet, QTrainer
from os.path import exists as os_path_exists
import pickle

class Agent:
        
    def __init__(self):

        self.max_memory = 100000
        self.batch_size = 1000
        self.learning_rate = 0.001

        self.num_simulations = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        
        # Load model / Existing weights + biases (if there are any) and 
        self.model = Linear_Qnet(input_size = 11, hidden_size = 256, output_size = 3)
        if os_path_exists("model"):

            # Load state dict
            saved_dict = torch.load("model/model.pth", map_location = torch.device("cpu"))
            self.model.load_state_dict(state_dict = saved_dict)

            # Load memory
            with open("model/memory.txt", "rb") as memory_file:
                serialised_queue = memory_file.read()
            self.memory = pickle.loads(serialised_queue)

            # Set the number of simulations to be the last saved number
            with open("model/simulations.txt", "r") as simulations_file:
                self.num_simulations = int(simulations_file.read())

        else:
            # Create new memory queue
            self.memory = deque(maxlen = self.max_memory) # Automatically discards leftmost item when exceeding memory

        self.trainer = QTrainer(self.model, learning_rate = self.learning_rate, gamma = self.gamma)

    def get_state(self, game):

        # There are 11 states

        c_direction = game.snake.current_direction
        snake_head = game.snake.parts[0]
        cell_size = game.cell_dimensions

        l_point = [snake_head[0] - cell_size[0], snake_head[1]]
        r_point = [snake_head[0] + cell_size[0], snake_head[1]]
        u_point = [snake_head[0], snake_head[1] - cell_size[1]]
        d_point = [snake_head[0], snake_head[1] + cell_size[1]]

        state = [
                # Danger straight
                (c_direction == "L" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = l_point)) or
                (c_direction == "R" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = r_point)) or
                (c_direction == "U" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = u_point)) or
                (c_direction == "D" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = d_point)),

                # Danger right
                (c_direction == "L" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = u_point)) or
                (c_direction == "R" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = d_point)) or
                (c_direction == "U" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = r_point)) or
                (c_direction == "D" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = l_point)),

                # Danger left
                (c_direction == "L" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = d_point)) or
                (c_direction == "R" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = u_point)) or
                (c_direction == "U" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = l_point)) or
                (c_direction == "D" and game.snake.check_collision(screen_width = game.dimensions[0], screen_height = game.dimensions[1], point = r_point)),

                # Booleans for current direction
                c_direction == "L", 
                c_direction == "R",
                c_direction == "U",
                c_direction == "D",

                # Food location
                game.food[0] < snake_head[0], # Food left
                game.food[0] > snake_head[0], # Food right
                game.food[1] < snake_head[1], # Food up
                game.food[1] > snake_head[1]  # Food down
                ]

        return np.array(state, dtype = int) # dtype = int converts all boolean values to 0 or 1
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # Automatically discards leftmost item when exceeding memory
    
    def train_long_memory(self):
        # Grab self.batch_size samples from memory
        if len(self.memory) >= self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)

        else:
            mini_sample = self.memory
        
        # Train on samples
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) # Put everything together
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        
        # In the beginning (< 80 simulations) perform "Exploration"
        # Once the model has been trained more (> 80 simulations), perform "Exploitation"

        self.epsilon = 80 - self.num_simulations
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        else:
            # FInd prediction
            state_0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state_0)
            
            # Find move
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # Return the chosen move
        return final_move