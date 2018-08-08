import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])   # Features
y = np.array([[0,1,1,0]]).T                         # Labels
alpha,hidden_dim = (0.5,4)                          # Configuration variables
synapse_0 = 2*np.random.random((3,hidden_dim)) - 1  # First hidden layer
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1  # Second hidden layer

for j in range(60000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))                              # First logistics
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))                        # Second logistics
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))                         # 
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))