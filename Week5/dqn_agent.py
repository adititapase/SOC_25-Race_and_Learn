import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import os
from snake_env import SnakeEnv

EPISODES = 1000
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
GAMMA = 0.9
LR = 0.001
TARGET_UPDATE = 10

EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=MAX_MEMORY):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.env = SnakeEnv()
        self.state_dim = 11
        self.action_dim = 3

        self.model = DQN(self.state_dim, 256, self.action_dim).to(DEVICE)
        self.target_model = DQN(self.state_dim, 256, self.action_dim).to(DEVICE)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        global EPSILON
        if random.random() < EPSILON:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(DEVICE)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        global EPSILON
        scores = []
        avg_scores = []

        for episode in range(EPISODES+1):
            state = self.env.reset()
            done = False

            while not done:
                action = self.get_action(state)
                reward, done = self.env.play_step(action)
                next_state = self.env.get_state()

                self.memory.push((state, action, reward, next_state, done))
                state = next_state

                self.train_step()
                self.env.render()

            score = len(self.env.snake.body) - 3
            scores.append(score)

            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

            if episode % TARGET_UPDATE == 0:
                self.update_target_network()

            if episode % 10 == 0:
                avg = sum(scores[-10:]) / 10
                avg_scores.append(avg)
                print(f"Episode {episode} | Avg Score: {avg:.2f} | Epsilon: {EPSILON:.3f}")

        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.plot(avg_scores)
        plt.xlabel("Games (x10)")
        plt.ylabel("Score")
        plt.title("Score vs Number of Games (DQN)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Plots/dqn_score_plot.png")
        print("Score plot saved to plots/dqn_score_plot.png")

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
