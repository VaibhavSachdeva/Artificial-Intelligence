import numpy as np
np.random.seed(0)
# '''
# Batches because we want to run parallel processing.
# we use gpu over cpu because we can perform highly complex algorithms and huge data parallaly and quickly
# More the batch size, more the fitment line adjust easier. problem can be over fittment
# '''


# inputs = [[1,2,3,2.5],
#           [2.0,5.0,-1.0,2.0],
#           [-1.5,2.7,3.3,-0.8]]

# weights = [[0.2,0.8,-0.5,1.0],
#            [0.5,-0.91,0.26,-0.5],
#            [-0.26,-0.27,0.17,0.87]]
# '''
# transposing the weights
# to do so, we need to convert weights to numpy array
# '''



# biases = [2,3,0.5]

# output = np.dot(inputs,np.array(weights).T)+biases
# #print(output)

# #ADD ONE MORE LAYER
# weights2 = [[0.1,-0.14,0.5],
#             [-0.5,0.12,-0.33],
#             [-0.44,0.73,-0.13]]

# biases2 = [-1,2,-0.5]
# layer1_outputs = np.dot(inputs,np.array(weights).T)+biases
# layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T)+biases2

# print(layer2_outputs)

# # instead of doing things this way, we can convert it into objects n use it to create layeroutput

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

# input feature set denoted by capital x . a standard practice to denote input data in neural network

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases
        
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
#print(np.random.randn(4,3))
    
    
    