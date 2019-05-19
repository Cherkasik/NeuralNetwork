from numpy import exp, array, random, dot, mean, cos, square, amax
import matplotlib.pyplot as plt
class NN:
    def __init__(self):
        random.seed(1)
        self.sigmoid = lambda x: 1/(1+exp(-x))
        self.function = lambda x: exp(x)/cos(x)
        self.sigmoidPrime = lambda x: x * (1 - x)
        self.training_inputs = []
        self.training_outputs = []
        self.weights1 = 2*random.random((1, 3)) - 1
        self.weights2 = 2*random.random((3, 1)) - 1
        self.graph = []
        self.y = []
        plt.title("Error")
        plt.plot(self.y, self.graph)

    def generate_set(self, n):
        now = -10
        step = 20/n
        for i in range (n):
            self.training_inputs.append([float(now)])
            self.training_outputs.append([float(self.function(now))])
            now += step
        self.training_inputs = array(self.training_inputs, dtype=float)
        self.training_outputs = array(self.training_outputs, dtype=float)
        self.training_inputs /= 10
        self.training_outputs /= amax(self.training_outputs)

    def backward(self, input1, output1, i):
        l2_error = output1 - self.layer2
        if i != len(self.graph):
            self.graph[i] += mean(abs(l2_error))
        else:
            if i!=0:
                self.graph[i-1] /= len(self.training_inputs)
                plt.plot(self.y, self.graph, 'g')
                plt.pause(0.000000000000000000000000000000000000000000000000000000000001)
            self.y.append(i)
            self.graph.append(mean(abs(l2_error)))
        l2_delta =  l2_error*self.sigmoidPrime(self.layer2)
        l1_error = l2_delta.dot(self.weights2.T)
        l1_delta = l1_error*self.sigmoidPrime(self.layer1)
        self.weights2 += self.layer1.T.dot(l2_delta)
        self.weights1 += input1.T.dot(l1_delta)

    def train(self, j):
        for i in range(0, len(self.training_inputs), 3):
            self.predict(self.training_inputs[i:i+3])
            self.backward(self.training_inputs[i:i+3], self.training_outputs[i:i+3], j)

    def predict(self, input1): 
        self.layer1 = self.sigmoid(dot(input1, self.weights1))
        self.layer2 = self.sigmoid(dot(self.layer1, self.weights2))

    def output_predict(self, input1):
        self.predict(input1)
        return self.layer2

neural = NN()
neural.generate_set(102)
for i in range(6000):
    neural.train(i)
plt.show()