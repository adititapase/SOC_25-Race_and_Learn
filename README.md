# RL: Race and Learn 

This README file contains a consolidated summary of everything I have learned and improved upon over the 4 weeks of the SOC 25 project Race and Learn. The repository includes the submissions for coding assignments from week 1 and week 2.

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

---

## Overall Progress Reflection

- Strengthened core Python and OOP foundations through real projects.
- Created a fully playable Snake game using Python and the PyGame library.
- Built and trained CNN models from scratch using PyTorch.
- Developed a structured understanding of reinforcement learning, from MDPs to DQNs.

