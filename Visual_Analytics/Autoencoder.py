# Script to be used to Train an Autoencoder

# This script has been developed for the sole purpose of using a Neural Network
# for dimensionality reduction as an alternative to PCA, MDS or SOM's.

# TO DO: Import Data - Easy Addition
# TO DO: Get a half feed forward function to get the values the activation 
#        values at the Neurons - Medium - minus 1 from the range loop
#        range(0, len(self.weights)) to restrict the algorithm for propagating 
#        to the output layer.


# Link: http://en.wikipedia.org/wiki/Autoencoder


# Import Modules
import numpy as np
import time

# Define Activation Functions and their Respective Derivatives
def tanh(x):
    # Advised activation function taken from:
    # T.M, Heskes and B. Kappen. Online learning processes in artificial neural networks
    return 1.7159*np.tanh((2*x)/3)

def tanh_deriv(x):
    # Derivative derived from Heskes and Kappen's recommendation.
    return 1.14393*(1-(np.tanh((2*x)/3))**2)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

# Define a Class that can be used for Creating and Testing the Networks
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        # Switch Function that can change the Activation function used in the Network
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
            # Create all the Weights in the Network
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]+ 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

# Train the Neural Network
    def fit(self, X, y, learning_rate, epochs):
        # Start Clock
        start = time.time()

        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X # adding the bias unit to the input layer
        X = temp
        y = np.array(y)
        # Iterate through the Network - No. Epochs
        for k in range(epochs):
            # Number of Samples to Train on for each Epoch Pass
            for s in range(4):
                # Loop Through Forward and Backward with one Row
                i = np.random.randint(X.shape[0])
                a = [X[s]]
                # Forward Propagation
                for l in range(len(self.weights)):
                    a.append(self.activation(np.dot(a[l], self.weights[l])))
                    error = y[i] - a[-1]
                    deltas = [error * self.activation_deriv(a[-1])]
                    # For each Layer in the Network Backpropagate the Error
                    for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                        deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
                        deltas.reverse() # For ease of Process reverse the Array
                        for i in range(len(self.weights)):
                            layer = np.atleast_2d(a[i])
                            delta = np.atleast_2d(deltas[i])
                            # Update the Weights in the Network
                            self.weights[i] += learning_rate * layer.T.dot(delta)
        # End Stop Watch
        end = time.time()
        # Print a Message
        print("Training Neural Network - Time to Complete: %.4f Seconds" % (end - start))

# Define the Prediction Function
    def predict(self, x):
        # Convert the Test Array to a Numpy Array
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        # Forward the Network
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

# The Main Processing Algorithm
if __name__ == "__main__":
    # Start Clock
    start1 = time.time()
    # Data
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # Target
    y = np.array([1,0,0,1])
    # Neural Network Initialise
    nn = NeuralNetwork([2, 2, 1], 'tanh')
    # Train the Neural Network
    nn.fit(X, y, learning_rate=0.01, epochs=100000)
    # Prediction
    prediction = np.zeros(shape=(int(X.shape[0]), 1))
    for i in X:
        prediction[i] = nn.predict(i)
    # Print Output
    print prediction