import random
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from snake_env import SnakeEnv

EPISODES = 1000
ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor
EPSILON = 1.0       # exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Q-table: state -> [q0, q1, q2] where q0=straight, q1=right, q2=left
Q = defaultdict(lambda: [0, 0, 0])

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 2)  # Explore
    return int(np.argmax(Q[state]))  # Exploit

def train():
    global EPSILON
    scores = []
    avg_scores = []

    for episode in range(EPISODES+1):
        env = SnakeEnv()
        state = env.reset()
        done = False

        while not done:
            action = choose_action(state)
            reward, done = env.play_step(action)
            next_state = env.get_state()

            # Q-learning update
            Q[state][action] += ALPHA * (reward + GAMMA * max(Q[next_state]) - Q[state][action])
            state = next_state

            env.render()  
            # pygame.time.delay(40)  # to slow down visualization

        # Use score as final metric
        score = len(env.snake.body) - 3
        scores.append(score)

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        if episode % 10 == 0:
            avg = sum(scores[-10:]) / 10
            avg_scores.append(avg)
            print(f"Episode {episode} | Avg Score: {avg:.2f} | Epsilon: {EPSILON:.3f}")

    # Save score-vs-game graph
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(avg_scores)
    plt.xlabel("Games (x10)")
    plt.ylabel("Score")
    plt.title("Score vs Number of Games (Q-Learning)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/qlearning_score_plot.png")
    print("Score plot saved to plots/qlearning_score_plot.png")

if __name__ == "__main__":
    train()
