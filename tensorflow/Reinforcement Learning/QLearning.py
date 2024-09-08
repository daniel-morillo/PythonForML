import gym
import numpy as np
import time

# Set render_mode based on the RENDER flag
RENDER = False  # Render the environment or not
render_mode = 'human' if RENDER else None

env = gym.make('FrozenLake-v1', render_mode=render_mode)
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Create a Q-table with all zeros
Q = np.zeros((STATES, ACTIONS))

# Constants Definition
EPISODES = 10000  # Number of episodes
MAX_STEPS = 100  # Max number of steps per episode
LEARNING_RATE = 0.75  # Alpha
GAMMA = 0.96  # Discount factor
RENDER = False  # Render the environment
epsilon = 0.85  # Exploration rate

def trainModel(env, EPISODES, MAX_STEPS, LEARNING_RATE, GAMMA, epsilon, RENDER, Q):
    rewards = []
    for episode in range(EPISODES):
        # Reset the environment and get the initial state
        state, _ = env.reset()

        for _ in range(MAX_STEPS):
            # Only render if RENDER is True
            if RENDER:
                env.render()

            # Choose action: explore or exploit
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # Take action and get next state, reward, and done flag
            next_state, reward, done, _, _ = env.step(action)

            # Update Q-table using the Q-learning update rule
            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            # Transition to the next state
            state = next_state

            # If the episode is finished, break
            if done:
                rewards.append(reward)
                # Decay epsilon after each episode for less exploration
                epsilon = max(0.01, epsilon - 0.001)  # This is to avoid getting stuck in local optima
                break  # Episode is done
    return rewards, Q

# Train the model
rewards, Q = trainModel(env, EPISODES, MAX_STEPS, LEARNING_RATE, GAMMA, epsilon, RENDER, Q)

# Display the resulting Q-table and average reward
print(Q)
print('Average Reward: ', sum(rewards) / len(rewards))



