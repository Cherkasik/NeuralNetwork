from numpy import exp, array, random, dot, transpose
from math import cos
class NN:
    def __init__(self):
        self.sigmoid = lambda x: 1/(1+exp(-x))
        self.function = lambda x: exp(x)/cos(x)
        self.sigmoidPrime = lambda x: x * (x - 1)
        self.training_inputs = []
        self.training_outputs = []
        self.weights1 = random.randn(1, 3)
        self.weights2 = random.randn(3, 1)
        self.error = 0
        self.adjustments = 0
    def generate_set(self, n):
        for i in range (n):
            self.training_inputs.append([float(i+1)])
            self.training_outputs.append([float(self.function(i+1))])
        self.training_inputs = array(self.training_inputs, dtype=float)
        self.training_outputs = array(self.training_outputs, dtype=float)
    
    def backward(self, o):
        self.o_error = self.training_outputs - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.weights2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.weights1 += self.training_inputs.T.dot(self.z2_delta)
        self.weights2 += self.z2.T.dot(self.o_delta)
        
    def train(self):
        o = self.predict(self.training_inputs)
        self.backward(o)

    def predict(self, input):
        self.z = dot(input, self.weights1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = dot(self.z2, self.weights2)
        o = self.sigmoid(self.z3[0])
        return o

neural = NN()
neural.generate_set(100)
for i in range(150):
    neural.train()
print("Predicted Output: " + str(neural.predict(1)))
print("Actual Output:" + str(neural.training_outputs[0]))
