import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gym

class GraphSignalSamplingEnv(gym.Env):
    def __init__(self):
        super(GraphSignalSamplingEnv, self).__init__()
        
        # Create a random graph
        self.graph = nx.fast_gnp_random_graph(160, 0.5)
        self.num_nodes = len(self.graph)
        
        # Generate a random signal on the graph nodes
        self.signal = np.random.rand(self.num_nodes)
        
        # Define the sampling pattern (e.g., selecting random nodes)
        self.num_samples = 50
        self.sample_nodes = np.random.choice(list(self.graph.nodes()), self.num_samples, replace=False)
        
        # Set the observation and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_nodes,))
        self.action_space = gym.spaces.Discrete(self.num_samples)
        
    # def reset(self):
    #     # Reset the environment
    #     self.sample_nodes = np.random.choice(list(self.graph.nodes()), self.num_samples, replace=False)
        
    #     return self.signal

    def reset(self, graphNodes):
        self.sample_nodes = graphNodes
        self.sample_nodes = np.random.choice(list(self.graph.nodes()), self.num_samples, replace=False)
        return self.signal
    
    def step(self, action):
        # Sample the signal at the selected node
        sampled_signal = self.signal[self.sample_nodes[action]]
        
        # Calculate the reward (e.g., difference from the true signal value)
        reward = abs(sampled_signal - self.signal[0])
        
        # Update the observation (signal) based on the sampled node
        self.signal = np.roll(self.signal, shift=1)
        
        # Check if the episode is done (optional)
        done = False
        
        return self.signal, reward, done, {}
    

# print("test1")
# Create a gym environment instance
# env = GraphSignalSamplingEnv()

# # Training loop
# for episode in range(10):
#     state = env.reset()
#     done = False
    
#     while not done:
#         action = env.action_space.sample()
#         next_state, reward, done, _ = env.step(action)
#         break
        # Perform reinforcement learning updates using the state, action, next_state, reward
        
# Plot the final updated signal
