# maze_exit
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# One-hot encoding function
def one_hot_encode(state, state_size):
    encoded = np.zeros(state_size)
    encoded[state] = 1
    return encoded

# Environment definition
class GridWorldEnv:
    def __init__(self):
        self.state_space = 9  # 3x3 grid
        self.action_space = 4  # Up, Down, Left, Right
        self.goal_state = 8  # Goal at position 8 (bottom-right corner)
        self.state = 0  # Initial position (top-left)
        self.terminated = False

    def reset(self):
        self.state = 0  # Start at the top-left corner
        self.terminated = False
        return self.state

    def step(self, action):
        if self.terminated:
            return self.state, 0, True

        row = self.state // 3
        col = self.state % 3

        # Action mapping: 0 -> up, 1 -> down, 2 -> left, 3 -> right
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < 2:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < 2:
            col += 1

        next_state = row * 3 + col
        reward = 10 if next_state == self.goal_state else -0.1

        self.state = next_state
        if self.state == self.goal_state:
            self.terminated = True

        return self.state, reward, self.terminated

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005
        self.batch_size = 16
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network for Deep Q-learning
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Convert the state to a one-hot encoded vector
        one_hot_state = one_hot_encode(state, self.state_size).reshape(1, -1)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(one_hot_state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Convert the next_state to a one-hot encoded vector
                one_hot_next_state = one_hot_encode(next_state, self.state_size).reshape(1, -1)
                target = reward + self.gamma * np.amax(self.model.predict(one_hot_next_state, verbose=0)[0])

            # Convert the current state to a one-hot encoded vector
            one_hot_state = one_hot_encode(state, self.state_size).reshape(1, -1)
            target_f = self.model.predict(one_hot_state, verbose=0)
            target_f[0][action] = target
            self.model.fit(one_hot_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Function to print Q-values for each state
def print_q_values(agent):
    print("\nQ-values for each grid state (actions: [up, down, left, right]):")
    for state in range(9):  # States from 0 to 8 (3x3 grid)
        one_hot_state = one_hot_encode(state, agent.state_size).reshape(1, -1)
        q_values = agent.model.predict(one_hot_state, verbose=0)[0]
        print(f"State {state}: {q_values}")

# Training the agent
if __name__ == "__main__":
    env = GridWorldEnv()
    state_size = env.state_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    episodes = 200  # Reduced for faster training
    max_steps = 50  # Limit number of steps per episode

    for e in range(episodes):
        state = env.reset()
        for time in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, score: {time+1}, epsilon: {agent.epsilon:.2}")
                break
        agent.replay()

    # Save the trained model
    agent.save("dqn_gridworld.weights.h5") # Added the .weights extension to the filename

# Print Q-values for each state after training
print_q_values(agent) # Removed the unintended indentation
