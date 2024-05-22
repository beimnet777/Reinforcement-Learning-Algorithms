import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import colors

# Set up the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

def value_iteration(env, gamma=0.99, theta=1e-9):
    value_table = np.zeros(env.observation_space.n)
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            q_value = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.unwrapped.P[state][action]:
                    prob, next_state, reward, done = next_sr
                    q_value[action] += prob * (reward + gamma * value_table[next_state])
            updated_value_table[state] = max(q_value)
        if np.sum(np.fabs(updated_value_table - value_table)) <= theta:
            break
        value_table = updated_value_table
    return value_table

def policy_iteration(env, gamma=0.99):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    value_table = np.zeros(env.observation_space.n)
    while True:
        stable_policy = True
        for state in range(env.observation_space.n):
            chosen_action = policy[state]
            action_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.unwrapped.P[state][action]:
                    prob, next_state, reward, done = next_sr
                    action_values[action] += prob * (reward + gamma * value_table[next_state])
            best_action = np.argmax(action_values)
            if chosen_action != best_action:
                stable_policy = False
            policy[state] = best_action
        if stable_policy:
            break
        value_table = value_iteration(env, gamma)
    return policy, value_table

def q_learning(env, num_episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, _, _ = env.step(action)
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state
    return q_table

def epsilon_greedy_policy(q_table, epsilon=0.1):
    def policy_function(state):
        if np.random.rand() < epsilon:
            return np.random.randint(q_table.shape[1])
        return np.argmax(q_table[state])
    return policy_function

def ucb_algorithm(env, num_episodes=1000, c=2):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    action_count = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            total_count = np.sum(action_count[state]) + 1
            ucb_values = q_table[state] + c * np.sqrt(np.log(total_count) / (action_count[state] + 1))
            action = np.argmax(ucb_values)
            next_state, reward, done, _, _ = env.step(action)
            action_count[state][action] += 1
            q_table[state][action] += (reward - q_table[state][action]) / action_count[state][action]
            state = next_state
    return q_table

# Function to visualize the grid and policy
def visualize_grid_policy(env, policy):
    grid_size = int(np.sqrt(env.observation_space.n))
    policy_grid = np.reshape(policy, (grid_size, grid_size))

    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_symbols = np.vectorize(action_symbols.get)(policy_grid)

    grid_data = env.desc.astype(str)
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_data[i, j] == b'S':
                grid_data[i, j] = 'S'
            elif grid_data[i, j] == b'F':
                grid_data[i, j] = policy_symbols[i, j]
            elif grid_data[i, j] == b'G':
                grid_data[i, j] = 'G'
            elif grid_data[i, j] == b'H':
                grid_data[i, j] = 'H'
    
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(['white', 'black', 'red', 'green'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    grid_display = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_data[i, j] == 'S':
                grid_display[i, j] = 0
            elif grid_data[i, j] == 'H':
                grid_display[i, j] = 1
            elif grid_data[i, j] == 'G':
                grid_display[i, j] = 3
            else:
                grid_display[i, j] = 2
    
    ax.imshow(grid_display, cmap=cmap, norm=norm)
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax.text(j, i, grid_data[i, j],
                           ha="center", va="center", color="black")
    
    plt.show()

# Main function to run the algorithms and visualize the grid
def main():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    value_table = value_iteration(env)
    print("Value Iteration Value Table:")
    print(value_table)

    policy, value_table_pi = policy_iteration(env)
    print("Policy Iteration Policy and Value Table:")
    print(policy)
    print(value_table_pi)
    visualize_grid_policy(env, policy)

    q_table = q_learning(env)
    print("Q-Learning Q Table:")
    print(q_table)
    # Assuming the optimal policy from Q-learning would be the action with the highest value for each state
    q_policy = np.argmax(q_table, axis=1)
    visualize_grid_policy(env, q_policy)

    policy_fn = epsilon_greedy_policy(q_table)
    print("Epsilon-Greedy Policy Function (example state 0):")
    print(policy_fn(0))

    ucb_q_table = ucb_algorithm(env)
    print("UCB Algorithm Q Table:")
    print(ucb_q_table)
    # Assuming the optimal policy from UCB would be the action with the highest value for each state
    ucb_policy = np.argmax(ucb_q_table, axis=1)
    visualize_grid_policy(env, ucb_policy)

if __name__ == "__main__":
    main()
