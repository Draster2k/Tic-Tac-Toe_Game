import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # Input layer (board state as 9 values)
        self.fc2 = nn.Linear(64, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 9)  # Output layer (Q-values for 9 actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


class DQLAgent:
    def __init__(self, alpha=0.005, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = []
        self.batch_size = 32
        self.max_memory = 1000

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def compare_q_tables(self, old_q_values):
        """Compare old Q-values with current Q-values and return the differences."""
        current_q_values = {name: param.data.clone() for name, param in self.model.named_parameters() if
                            'weight' in name or 'bias' in name}

        differences = {}
        for key in old_q_values.keys():
            diff = torch.norm(current_q_values[key] - old_q_values[key]).item()
            differences[key] = diff

        return differences

    def build_model(self):
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
            nn.ReLU(),
        )
        return model

    def display_model(self):
        print(self.model)

    def get_board_state(self, board):
        # Convert board to tensor representation
        state = [0 if cell is None else (1 if cell == "X" else -1) for row in board for cell in row]

        # Return as a list if `get_available_actions` expects a list
        return state  # or `return torch.tensor(state)` if other parts rely on tensor format

    def get_best_action(self, state, valid_actions):
        # Convert the state to a tensor if it's not already one
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension for prediction

        # Get Q-values from the model for the current state
        q_values = self.model(state).squeeze()  # Remove batch dimension

        # Filter Q-values for valid actions only
        best_action = max(valid_actions, key=lambda a: q_values[a].item())

        return best_action

    def get_available_actions(self, board):
        # If board is a flattened list, reshape it to 3x3
        if isinstance(board, list) and len(board) == 9:
            board = [board[i:i + 3] for i in range(0, 9, 3)]

        valid_actions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # Assume 0 represents an empty cell
                    valid_actions.append(i * 3 + j)
        return valid_actions

    def choose_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = self.get_available_actions(state)  # Fetch valid actions if not provided

        if random.uniform(0, 1) < self.epsilon:  # Explore
            action = random.choice(valid_actions)
        else:  # Exploit
            action = self.get_best_action(state, valid_actions)  # Assuming you have a method for this

        return action

    def store_experience(self, state, action, reward, next_state, done):
        # Store experiences for replay
        if len(self.replay_buffer) >= self.max_memory:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert lists to tensors
        state_batch = torch.stack([torch.tensor(state, dtype=torch.float32) for state in state_batch])
        next_state_batch = torch.stack([torch.tensor(state, dtype=torch.float32) for state in next_state_batch])
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Get current Q-values for chosen actions
        current_q_values = self.model(state_batch).gather(1, torch.tensor(action_batch).view(-1, 1))

        # Compute loss and update model
        loss = self.loss_fn(current_q_values.view(-1), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.scheduler.step()
        self.optimizer.step()

        logging.info(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    '''
    def choose_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = [i for i in range(9)]  # All possible actions

        # Your existing logic to choose an action, but filtered by valid actions
        action = random.choice(valid_actions)  # Placeholder for actual logic

        return action
    '''

    def update_q_values(self, state, action, reward, next_state):

        current_q = self.model(state)[action]
        max_next_q = torch.max(self.target_model(next_state))
        target_q = reward + self.gamma * max_next_q

        logging.info(f"State: {state}, Action: {action}, Q_value: {current_q}, Target: {target_q}")

        loss = self.loss_fn(current_q, target_q)


class RandomAgent:
    def __init__(self, get_available_actions):
        self.get_available_actions = get_available_actions

    def choose_action(self, state):
        available_actions = self.get_available_actions(state)
        return random.choice(available_actions)