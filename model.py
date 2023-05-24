import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from numpy import array as np_array

class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # Forward propagation
        x = F.relu(self.linear1(x)) # RELU activation function on the input layer
        x = self.linear2(x)
        return x
    
    def save(self, file_name = "model.pth"):
        model_folder_path = "model"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model

        self.optimiser = optim.Adam(model.parameters(), lr = learning_rate)

        # Criterion / loss function (Mean squared error used)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        
        # Convert to PyTorch tensor
        action = torch.tensor(data = action, dtype = torch.long)
        reward = torch.tensor(data = reward, dtype = torch.float)
        state = torch.tensor(data = np_array(state), dtype = torch.float) # Convert to np_array for faster creation of tensor
        next_state = torch.tensor(data = np_array(next_state), dtype = torch.float) # Convert to np_array for faster creation of tensor

        if len(state.shape) == 1: # 1 dimensional
            # Reshape to: (1, x) number of batches, x
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over,) # Convert to tuple

        # Get predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()

        # Iterate over tensors and apply Bellman equation
        # Bellman equation: Q_new = r + y * max(next_predicted Q value)
        for i in range(len(game_over)):
            Q_new = reward[i]
            
            # If the AI didn't collide with itself / the borders
            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            # Set the target of the maximum value of the action to Q_new
            target[i][torch.argmax(action).item()] = Q_new  

        # Empty gradients
        self.optimiser.zero_grad()

        # Calculate loss
        loss = self.criterion(target, prediction)

        # Backpropagate
        loss.backward()
        self.optimiser.step()