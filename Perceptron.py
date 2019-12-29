import numpy as np 


def sigmoid(x):
	return 1/ (1+np.exp(-x))

training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

training_out = np.array([[0,1,1,0]])
training_outputs = np.transpose(training_out)

def sigmoid_derivative(x):
	#backpropagation is always the derivative of the function operation chosen. In this case, sigmoid was chosen. Derivative of sigmoid is... 
	return x * (1 -  x) 
 
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random synaptic weights initializing')
print(synaptic_weights)

for iteration in range(10000):

	input_layer = training_inputs
	#matrix multiplication on the synaptic weights and input layer
	#Followed by passing through normalization function, sigmoid was chosen here.
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))

	error = training_outputs - outputs

	backpropagation = error * sigmoid_derivative(outputs) 
	
	synaptic_weights += np.dot(np.transpose(input_layer), backpropagation)

print('synaptic weights after training')
print(synaptic_weights)

print('Outputs after training')
print(outputs)
