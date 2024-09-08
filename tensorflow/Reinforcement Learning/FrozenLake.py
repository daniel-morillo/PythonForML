import gym

env = gym.make('FrozenLake-v1', render_mode = 'human')

def showEnvSpaces(env):
    print(env.observation_space)
    print(env.action_space)

# Reset the environment
env.reset()

def randomAction(env):
    action = env.action_space.sample()

    # Unpack the five values returned by env.step()
    new_state, reward, done, truncated, info = env.step(action)
    return new_state, reward, done, truncated, info

new_state, reward, done, truncated, info = randomAction(env)

def showGame(env):
    done = False
    while not done:
        new_state, reward, done, truncated, info = randomAction(env)
        env.render()

    env.close()





