

import networkx as nx
import numpy as np
from sklearn.datasets import fetch_openml

# Function to generate a random matrix of given size
def generate_matrices(num_matrices, n):
    matrices = []
    for _ in range(num_matrices):
        matrix = np.random.randint(2, size=(n, n))  # Generate random 0s and 1s
        matrix = np.tril(matrix) + np.tril(matrix, -1).T  # Make the matrix symmetric
        np.fill_diagonal(matrix, 0)  # Ensure no self-loops
        matrices.append(matrix)
    return matrices


# Generate 10 random matrices of size 408x408
num_matrices = 10
matrix_size = 408

mat0=generate_matrices(10, 408)

# Function to convert image to a graph
def matrices_to_graphs(matrices):
    graphs = []
    for matrix in matrices:
        graph = nx.convert_matrix.from_numpy_array(matrix)
        graphs.append(graph)
    return graphs

print("Test0")
# Convert the first image to a graph as an example
G_list = matrices_to_graphs(mat0)

print("Test1")
# Now you can work with the graph using NetworkX functions
from graphRL import * 

env = GraphSignalSamplingEnv()
print(G_list[0].number_of_nodes())

for episode in range(1):
    # state = env.reset()
    state = env.reset(G_list[episode].nodes())
    done = False
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        break

plt.figure()
nx.draw(env.graph, with_labels=True, node_color=env.signal, cmap='viridis')
plt.title('Updated Signal')
plt.show()
