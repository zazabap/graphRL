
import gym
from gym import spaces
import numpy as np
import networkx as nx

class GraphSignalSamplingEnv(gym.Env):
    def __init__(self, graph_size=10, num_nodes=100, num_signals=5):
        super(GraphSignalSamplingEnv, self).__init__()
        self.graph_size = graph_size
        self.num_nodes = num_nodes
        self.num_signals = num_signals

        self.graph = self.generate_graph()
        self.state_space = spaces.Discrete(self.graph_size)
        self.action_space = spaces.Discrete(self.num_nodes)

        # Initialize state and current node
        self.state = None
        self.current_node = None
        self.reset()

    def generate_graph(self):
        # Generate a random graph
        return nx.random_geometric_graph(self.num_nodes, 0.2)

    def reset(self):
        # Reset environment to initial state
        self.state = np.random.randint(self.graph_size)
        self.current_node = np.random.choice(list(self.graph.nodes))
        return self.state

    def step(self, action):
        # Move to the selected node
        self.current_node = action

        # Sample signals from neighboring nodes
        signals = []
        neighbors = list(self.graph.neighbors(self.current_node))
        for _ in range(self.num_signals):
            neighbor = np.random.choice(neighbors)
            signal_value = np.random.uniform(0, 1)  # Example: random signal value
            signals.append(signal_value)

        # Calculate reward (example: sum of signal values)
        reward = sum(signals)

        # Determine if the episode is done
        done = False  # You can define your own termination condition

        # Return next state, reward, done flag, and additional info
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Render the environment (optional)
        pass

    def close(self):
        # Close any resources if necessary (optional)
        pass

# Example usage:
env = GraphSignalSamplingEnv()
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action selection
    next_state, reward, done, _ = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
    state = next_state


