# Deep Convolutional Q-Learning for Ms. Pac-Man 🎮

A Deep Reinforcement Learning project that trains an AI agent to play **Ms. Pac-Man** using a **Deep Convolutional Q-Network (DCQN)** with PyTorch and Gymnasium.

## 🧠 Overview

The agent learns directly from game frames by:

- Processing game frames with a Convolutional Neural Network
- Predicting Q-values for 9 possible actions
- Using epsilon-greedy exploration
- Storing experiences in a replay buffer
- Updating the network using Q-learning

The original environment provides observations of **210 × 160 × 3**. Frames are resized to **128 × 128** before being passed to the network.

## 🏗️ Model Architecture

```text
Input Frame
    ↓
Conv2D + BatchNorm + ReLU
    ↓
Conv2D + BatchNorm + ReLU
    ↓
Conv2D + BatchNorm + ReLU
    ↓
Conv2D + BatchNorm + ReLU
    ↓
Fully Connected Layers
    ↓
Q-values for 9 Actions
```

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---:|
| Learning Rate | `5e-4` |
| Batch Size | `64` |
| Discount Factor | `0.99` |
| Replay Buffer | `10,000` |
| Episodes | `2,000` |
| Initial Epsilon | `1.0` |
| Minimum Epsilon | `0.01` |
| Epsilon Decay | `0.995` |

## 🛠️ Tech Stack

- Python
- PyTorch
- Gymnasium
- ALE / ale-py
- NumPy
- Torchvision
- PIL
- ImageIO

## 📦 Installation

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install ale-py
pip install gymnasium[box2d]
apt-get install -y swig
```

## ▶️ Training

Create the environment and initialize the agent:

```python
env = gym.make("MsPacmanNoFrameskip-v4", full_action_space=False)
agent = Agent(number_actions)
```

The agent trains for up to 2,000 episodes. Training stops when the average score reaches **500**, and the model is saved as `checkpoint.pth`.

## 🎥 Visualization

After training, the agent can be evaluated and its gameplay recorded as:

```text
video.mp4
```

## 📚 Concepts

**Deep Q-Learning • CNNs • Experience Replay • Epsilon-Greedy • Reinforcement Learning • Atari Games**
