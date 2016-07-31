import numpy as np
import random 
import cPickle

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(1, y) for y in layers[1:]]

    def feedforward(self, activation):
        for weight, bias in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(activation, weight)+bias)
        return activation

    def vectorized_result(self, y):
        e = np.zeros((1, self.layers[-1]))
        e[0][y] = 1.0
        return e

    def train(self, epochs, eta, batch_size, training_data, test_data=None):
        training_data = [(np.reshape(x, (1, len(x))), self.vectorized_result(y)) for x, y in training_data]
        test_data = [(np.reshape(x, (1, len(x))), y) for x, y in test_data]

        training_data_size = len(training_data)
        if test_data:
            test_data_size = len(test_data)
        for i in xrange(0, epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in xrange(0, training_data_size, batch_size)]
            for batch in batches:
                self.update(batch, eta)
            if test_data:
                p1 = self.evaluate(test_data)
                p2 = test_data_size
                print "Epoch {0}: {1} / {2} = {3}".format(i, p1, p2, p1*1.0/p2)
            else:
                print "Epoch {0} complete".format(i)

    def update(self, batch, eta):
        batch_size = len(batch)
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        
        for x, y in batch:
            delta_nabla_weights, delta_nabla_biases = self.backprop(x, y)
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb+dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]

        self.weights = [w-(eta/batch_size)*nw for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b-(eta/batch_size)*nb for b, nb in zip(self.biases, nabla_biases)]

    def backprop(self, x, y):
        activation = x
        activations = [x]
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight)+bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activation-y)*sigmoid_prime(zs[-1])

        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        nabla_weights[-1] = np.dot(activations[-2].transpose(), delta)
        nabla_biases[-1] = delta

        for l in xrange(2, len(self.layers)):
            delta = np.dot(delta, self.weights[-l+1].transpose())*sigmoid_prime(zs[-l])
            nabla_weights[-l] = np.dot(activations[-l-1].transpose(), delta)
            nabla_biases[-l] = delta

        return (nabla_weights, nabla_biases)

    def evaluate(self, test_data):
        result = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for x, y in result)
    
    def recognize(self, x):
        return np.argmax(self.feedforward(x))