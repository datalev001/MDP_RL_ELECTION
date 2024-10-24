import numpy as np

# Define states and actions
states = ["Survey Improve", "Survey Reduce"]
actions = ["Do not Campaign", "Campaign"]

# Transition probabilities: T[s, a, s']
T = np.array([
    [[0.4, 0.6], [0.7, 0.3]],  # From Survey Improve: [Don't Campaign, Campaign]
    [[0.2, 0.8], [0.5, 0.5]]   # From Survey Reduce: [Don't Campaign, Campaign]
])

# Reward matrix: R[s, a]
R = np.array([
    [50, 48],  # Survey Improve: [Don't Campaign, Campaign]
    [40, 45]   # Survey Reduce: [Don't Campaign, Campaign]
])

# Discount factor (how much Trump values future rewards)
gamma = 0.9

# Value Iteration function
def value_iteration(states, actions, T, R, gamma, iterations=100):
    V = np.zeros(len(states))
    for i in range(iterations):  # Iterate until convergence
        new_V = np.zeros(len(states))
        for s in range(len(states)):
            new_V[s] = max([R[s, a] + gamma * sum(T[s, a, s_next] * V[s_next]
                            for s_next in range(len(states))) for a in range(len(actions))])
        if np.allclose(V, new_V):
            break
        V = new_V

    # Extract optimal policy
    policy = np.zeros(len(states), dtype=int)
    for s in range(len(states)):
        policy[s] = np.argmax([R[s, a] + gamma * sum(T[s, a, s_next] * V[s_next]
                               for s_next in range(len(states))) for a in range(len(actions))])

    optimal_actions = [actions[a] for a in policy]
    return V, optimal_actions

# Policy Iteration function
def policy_iteration(states, actions, T, R, gamma, iterations=100):
    policy = np.zeros(len(states), dtype=int)  # Initialize policy arbitrarily
    V = np.zeros(len(states))  # Initialize value function

    stable_policy = False
    while not stable_policy:
        # Policy Evaluation
        for i in range(iterations):
            new_V = np.zeros(len(states))
            for s in range(len(states)):
                a = policy[s]  # Follow the current policy
                new_V[s] = R[s, a] + gamma * sum(T[s, a, s_next] * V[s_next] for s_next in range(len(states)))
            if np.allclose(V, new_V):
                break
            V = new_V

        # Policy Improvement
        stable_policy = True
        for s in range(len(states)):
            best_action = np.argmax([R[s, a] + gamma * sum(T[s, a, s_next] * V[s_next] for s_next in range(len(states))) for a in range(len(actions))])
            if best_action != policy[s]:
                policy[s] = best_action
                stable_policy = False

    optimal_actions = [actions[a] for a in policy]
    return V, optimal_actions

# Run Value Iteration
V_vi, policy_vi = value_iteration(states, actions, T, R, gamma)
print("Value Iteration - Optimal Value Function:", V_vi)
print("Value Iteration - Optimal Policy:", policy_vi)

# Run Policy Iteration
V_pi, policy_pi = policy_iteration(states, actions, T, R, gamma)
print("Policy Iteration - Optimal Value Function:", V_pi)
print("Policy Iteration - Optimal Policy:", policy_pi)

   
    
    

########RL############
import numpy as np
import random

# Initialize states and actions
states = ["Survey Improve", "Survey Reduce"]
actions = ["Do not Campaign", "Campaign"]

# Q-table: Q(s, a) initialized to zero
Q = np.zeros((len(states), len(actions)))

# Rewards
R = np.array([
    [50, 48],  # Survey Improve: [Don't Campaign, Campaign]
    [40, 45]   # Survey Reduce: [Don't Campaign, Campaign]
])

# Learning rate (alpha), discount factor (gamma), exploration rate (epsilon)
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Number of iterations (simulated months)
n_iterations = 1000

# Simulating the environment's response (state transitions)
def transition(state, action):
    if state == 0:  # Survey Improve
        return 0 if random.random() < 0.7 else 1  # Based on action, 70% chance stays Improve
    elif state == 1:  # Survey Reduce
        return 1 if random.random() < 0.5 else 0  # Based on action, 50% chance improves

# Q-learning algorithm
for _ in range(n_iterations):
    state = random.choice([0, 1])  # Randomly start in a state

    for _ in range(10):  # Each episode lasts for 10 months
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])  # Explore: Random action
        else:
            action = np.argmax(Q[state])  # Exploit: Choose best action based on Q-values

        # Get the reward for the current state and action
        reward = R[state, action]

        # Transition to the next state
        next_state = transition(state, action)

        # Q-value update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # Move to the next state
        state = next_state

# Output learned Q-values
print("Learned Q-values:")
print(Q)
