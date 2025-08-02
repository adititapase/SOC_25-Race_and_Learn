# RL: Race and Learn 

This README file contains a consolidated summary of everything I have learned and improved upon over the 8 weeks of the SOC 25 project Race and Learn. The repository includes the submissions for coding assignments from week 1, 2, 5 and the final week.

---

## Week 1: Game Development with Pygame

### Topics Covered

- **Game Development with PyGame:**  
  Learned to use the PyGame module to build 2D games. Understood the following key elements:
  - Game loops and event handling
  - Rendering graphics and text
  - Keyboard input handling
  - Updating game states and collision detection

### Week 1: Assignment

**Developed a Complete Snake Game using PyGame**

- Created a fully playable Snake game using Python and the PyGame library.
- The game includes all essential mechanics such as:
  - Snake movement and growth
  - Apple spawning at random locations
  - Collision detection with walls and self
  - Score display and game over screen
  - Sound effects and themed graphics using external assets

> This project significantly improved my understanding of OOP, real-time event handling, user interaction, and sprite management in PyGame.

---

## Week 2: Neural Networks & Convolutional Neural Networks (CNNs)

### Topics Covered

- **Neural Networks (NN):**  
  Learned how neural networks work from scratch:
  - Structure: input, hidden, and output layers
  - Neurons, weights, biases, and activation functions
  - Forward propagation and how inputs flow through layers
  - Backpropagation and how the network learns using gradients and loss

- **Activation Functions & Loss Functions:**  
  Understood the usage of activation functions like ReLU, Sigmoid, and Tanh, and loss functions like Cross Entropy and Mean Squared Error.

- **CNNs (Convolutional Neural Networks):**  
  Explored how CNNs are designed for image data:
  - Convolutional layers for feature extraction
  - Filters, kernels, and receptive fields
  - Pooling layers for downsampling
  - Flattening and fully connected layers

- **Overfitting & Regularization:**  
  Learned techniques such as dropout, data augmentation, and regularization to reduce overfitting in CNNs.

- **PyTorch Introduction:**  
  Practiced building neural networks using PyTorch:
  - Tensors and autograd
  - Defining models using `nn.Module`
  - Training loops, optimizers, and loss tracking

### Week 2: Assignment

**Built and Evaluated a CNN for Digit Classification on the MNIST Dataset**

- Implemented a **Convolutional Neural Network (CNN)** from scratch using PyTorch to classify handwritten digits (0–9) from the MNIST dataset.
- The model achieved over **99.4% accuracy** in just 5 epochs, showing strong performance.
- CNN Architecture:
  - Two convolutional layers for extracting local features.
  - ReLU activations and max pooling for non-linearity and downsampling.
  - Fully connected layers to map features to output digit classes.
- Plotted and analyzed:
  - **Training Loss**: Showed a consistent decline across epochs, indicating good convergence.
  - **Confusion Matrix**: Identified misclassifications like '5' confused with '3' or '8', which are reasonable given handwriting similarities.
- Prepared a detailed report documenting:
  - Model architecture and design rationale
  - Training behavior and performance metrics
  - Analysis of common misclassifications and their causes

> This assignment gave me practical experience with CNN design and training workflows, along with the ability to interpret results through evaluation metrics and visualization.

...

## Week 3: Introduction to Reinforcement Learning (RL)

### Topics Covered

- **Reinforcement Learning Basics:**
  - Learned how RL differs from supervised and unsupervised learning.
  - Understood the concepts of Agent, Environment, States, Actions, and Rewards.
  
- **Markov Decision Processes (MDPs):**  
  Explored the mathematical framework behind RL:
  - States and transitions
  - Transition probabilities
  - Reward functions
  - Policies (deterministic and stochastic)

- **Value-Based Methods:**
  - State-value function V(s) and action-value function Q(s, a)
  - Bellman equations for evaluating and improving policies

- **Exploration vs. Exploitation:**  
  Understood how agents balance trying new actions (exploration) and choosing the best-known ones (exploitation), using strategies like epsilon-greedy.

> This week was focused on deeply understanding the foundations of RL to prepare for upcoming implementation-heavy topics.

---

## Week 4: Q-Learning, Deep Q-Learning & DQN

### Topics Covered

- **Q-Learning Algorithm:**
  - Understood how agents learn optimal policies without an environment model.
  - Learned how the Q-table is initialized and updated using the Q-learning formula.
  - Practiced solving toy environments like GridWorld using tabular Q-learning.

- **Deep Q-Networks (DQN):**
  - Replaced Q-tables with neural networks to predict Q-values.
  - Learned concepts like experience replay, target networks, and training stability.
  - Understood the full training pipeline of a DQN.

- **Comparative Analysis:**
  - Compared Q-learning, Deep Q-learning, and Deep Q-Networks.
  - Identified key differences in their scalability, performance, and applicability.



## Week 5: DQN Implementation and Evaluation in Snake

### Topics Covered

**Deep Q-Network (DQN):**
- Implemented a custom Deep Q-Network agent from scratch using PyTorch for the Snake game environment.
- Covered the full reinforcement learning pipeline:
  - Replay Buffer for Experience Replay
  - Target Networks to stabilize Q-value updates
  - Epsilon-Greedy Strategy for exploration-exploitation tradeoff
  - Q-value bootstrapping and loss function computation

**Environment Interaction:**
- Encoded game states into vector representations suitable for neural networks.
- Managed frame skipping and action intervals to balance learning and game realism.

**Evaluation Techniques:**
- Tracked average score per episode and rolling mean.
- Compared trained agent vs random agent performance.

### Week 5: Assignment

Implemented and trained a Deep Q-Network agent to learn and play the Snake game environment effectively. Key features:
- Neural network with 3 fully connected layers trained to approximate Q(s, a).
- Successfully learned food-seeking and basic survival strategies.
- Performance improved consistently with increasing episodes, achieving stable convergence after training.

This week helped me solidify concepts of function approximation, stability techniques in RL, and hands-on implementation of Deep Q-learning for discrete action spaces.

---

## Week 6: Policy Gradient Methods – PPO and OpenAI Gym Environments

### Topics Covered

**Proximal Policy Optimization (PPO):**
- Studied the limitations of value-based methods and motivation behind policy-gradient methods.
- Understood how PPO optimizes a clipped surrogate objective to ensure stable policy updates.
- Learned advantages of Actor-Critic methods and how PPO strikes a balance between exploration and exploitation.

**OpenAI Gym Integration:**
- Explored Gym’s interface for observation space, action space, and episode management.

**Stable-Baselines3 Library:**
- Leveraged the SB3 library for easy and efficient PPO implementation.
- Learned how to set hyperparameters like learning rate, gamma, GAE lambda, and number of epochs.

### Week 6: Assignment

- Trained a PPO agent on CartPole-v1 and visualized performance using episode rewards.
- Modified observation space handling and logging for better interpretability.
- Compared PPO’s performance with DQN on small discrete environments.

This week expanded my understanding beyond value-based RL into continuous action domains and modern stable RL algorithms. It also familiarized me with reproducible benchmarking using standard libraries.

---

## Week 7 & 8: Final Project – PPO Agent for CarRacing-v3

### Topics Covered

**Advanced RL in Continuous Environments:**
- Tackled the challenge of solving a high-dimensional continuous control task: *CarRacing-v3*.
- Adapted PPO for image-based observations and continuous action outputs.
- Used frame-stacking and grayscale preprocessing to simplify input space and capture motion.

**Model Training & Evaluation:**
- Trained PPO agent using SB3 with custom wrappers for observation and reward shaping.
- Tracked performance over thousands of timesteps.
- Recorded and saved gameplay using Gym’s `RecordVideo` and evaluated agent behavior qualitatively.

**Challenges Solved:**
- Addressed video recording issues by switching to compatible rendering modes.
- Fixed observation shape mismatches and managed frame-skipping behavior for smoother input dynamics.

### Week 7 & 8: Final Submission

- Final submission includes:
  - PPO agent code for training and testing on CarRacing-v3
  - Preprocessing wrappers and environment setup
  - Trained model checkpoint and performance plots
  - Demo video showcasing agent performance
- The PPO agent learned lane-keeping behavior and some corner navigation despite the environment’s high complexity.

These final weeks gave me in-depth practical experience in:
- Policy gradient algorithms for continuous domains
- Visual learning in high-dimensional input spaces
- Engineering an end-to-end RL solution from training to deployment

---

## Summary

From building a Snake game to solving *CarRacing-v3* using PPO, this project has taken me through the complete stack of reinforcement learning—from foundational tabular methods to advanced actor-critic algorithms. The project enhanced my:
- Understanding of game environments and agent interaction
- Proficiency in PyTorch and Stable-Baselines3 for RL
- Ability to debug, train, and evaluate complex models
- Experience working on high-dimensional and continuous state-action problems

This repository stands as a comprehensive log of my progression and expertise gained in Reinforcement Learning during the SOC project.


