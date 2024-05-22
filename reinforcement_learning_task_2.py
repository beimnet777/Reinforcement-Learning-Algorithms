import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.randn(k)  # True action values
        self.best_action = np.argmax(self.q_true)
    
    def get_reward(self, action):
        reward = np.random.randn() + self.q_true[action]
        return reward

# Epsilon-Greedy Policy
def epsilon_greedy_bandit(bandit, num_episodes=1000, epsilon=0.1):
    q_estimates = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    rewards = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        if np.random.rand() < epsilon:
            action = np.random.randint(bandit.k)
        else:
            action = np.argmax(q_estimates)
        
        reward = bandit.get_reward(action)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
        rewards[episode] = reward
    
    return rewards, q_estimates

# Upper Confidence Bound (UCB)
def ucb_bandit(bandit, num_episodes=1000, c=2):
    q_estimates = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    rewards = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        if 0 in action_counts:
            action = np.argmin(action_counts)
        else:
            ucb_values = q_estimates + c * np.sqrt(np.log(episode + 1) / action_counts)
            action = np.argmax(ucb_values)
        
        reward = bandit.get_reward(action)
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
        rewards[episode] = reward
    
    return rewards, q_estimates

# Q-Learning adapted for bandits
def q_learning_bandit(bandit, num_episodes=1000, alpha=0.1, epsilon=0.1):
    q_estimates = np.zeros(bandit.k)
    rewards = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        if np.random.rand() < epsilon:
            action = np.random.randint(bandit.k)
        else:
            action = np.argmax(q_estimates)
        
        reward = bandit.get_reward(action)
        q_estimates[action] += alpha * (reward - q_estimates[action])
        rewards[episode] = reward
    
    return rewards, q_estimates

# Visualization function
def visualize_bandit_results(epsilon_rewards, ucb_rewards, q_learning_rewards, num_episodes):
    plt.figure(figsize=(12, 8))
    plt.plot(np.cumsum(epsilon_rewards), label='Epsilon-Greedy')
    plt.plot(np.cumsum(ucb_rewards), label='UCB')
    plt.plot(np.cumsum(q_learning_rewards), label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.title('Cumulative Reward over Time')
    plt.show()

def main():
    bandit = MultiArmedBandit(k=10)
    num_episodes = 1000
    
    epsilon_rewards, epsilon_q_estimates = epsilon_greedy_bandit(bandit, num_episodes)
    ucb_rewards, ucb_q_estimates = ucb_bandit(bandit, num_episodes)
    q_learning_rewards, q_learning_q_estimates = q_learning_bandit(bandit, num_episodes)
    
    visualize_bandit_results(epsilon_rewards, ucb_rewards, q_learning_rewards, num_episodes)
    
    print("Epsilon-Greedy Q-Estimates:", epsilon_q_estimates)
    print("UCB Q-Estimates:", ucb_q_estimates)
    print("Q-Learning Q-Estimates:", q_learning_q_estimates)

if __name__ == "__main__":
    main()
