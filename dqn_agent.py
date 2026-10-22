import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. DQN NEURAL NETWORK
# ==========================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.network(state)


# ==========================================
# 2. REPLAY BUFFER
# ==========================================

class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 3. DQN AGENT
# ==========================================

class DQNAgent:

    def __init__(self, state_size=37, action_size=4):

        self.state_size = state_size
        self.action_size = action_size

        # Automatically use GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        # ----------------------------------
        # Main network
        # ----------------------------------

        self.policy_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # ----------------------------------
        # Target network
        # ----------------------------------

        self.target_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Initially both networks are identical
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        # Target network is only used for prediction
        self.target_network.eval()

        # ----------------------------------
        # Training settings
        # ----------------------------------

        self.gamma = 0.99

        self.learning_rate = 0.001

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        # ----------------------------------
        # Exploration settings
        # ----------------------------------

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        # ----------------------------------
        # Replay memory
        # ----------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=10000
        )

        self.batch_size = 64

        # Update target network periodically
        self.target_update_frequency = 100

        self.training_steps = 0

    # ======================================
    # SELECT ACTION
    # ======================================

    def select_action(self, state, training=True):

        # Exploration
        if training and random.random() < self.epsilon:

            return random.randrange(self.action_size)

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return q_values.argmax(dim=1).item()

    # ======================================
    # STORE EXPERIENCE
    # ======================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ======================================
    # TRAIN NETWORK
    # ======================================

    def train_step(self):

        # Not enough experiences yet
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # ----------------------------------
        # Current Q-values
        # ----------------------------------

        current_q_values = self.policy_network(
            states
        ).gather(
            1,
            actions
        ).squeeze(1)

        # ----------------------------------
        # Target Q-values
        # ----------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(
                dim=1
            )[0]

            target_q_values = rewards + (
                self.gamma
                * next_q_values
                * (1 - dones)
            )

        # ----------------------------------
        # Calculate loss
        # ----------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ----------------------------------
        # Backpropagation
        # ----------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        # ----------------------------------
        # Update target network
        # ----------------------------------

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_network.load_state_dict(
                self.policy_network.state_dict()
            )

        # ----------------------------------
        # Reduce exploration
        # ----------------------------------

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )

        return loss.item()

    # ======================================
    # SAVE MODEL
    # ======================================

    def save(self, filepath):

        torch.save(
            self.policy_network.state_dict(),
            filepath
        )

    # ======================================
    # LOAD MODEL
    # ======================================

    def load(self, filepath):

        self.policy_network.load_state_dict(
            torch.load(
                filepath,
                map_location=self.device
            )
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )