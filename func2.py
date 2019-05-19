from numpy import exp, array, random, dot, square, round, mean
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        random.seed(1)
        self.sigmoid = lambda x: 1/(1+exp(-x))
        self.function = lambda x, y, z: (not(x) and not(z) or z and x and not(y))
        self.sigmoidPrime = lambda x: x * (1 - x)
        self.training_inputs = array([[0,0,0],[0,1,0],[1,0,1],[1,1,1]])
        self.training_outputs = array([[1,1,1,0]]).T
        self.weights1 = 2*random.random((3, 4)) - 1
        self.weights2 = 2*random.random((4, 1)) - 1
        self.graph = []
        self.y = []
        plt.title("Error")
        plt.plot(self.y, self.graph)

    def backward(self, i):
        l2_error = self.training_outputs - self.layer2
        self.graph.append(mean(abs(l2_error)))
        self.y.append(i)
        plt.plot(self.y, self.graph, 'g')
        plt.pause(0.000000000000000000000000000000000000000000000000000000000001)
        l2_delta = l2_error*self.sigmoidPrime(self.layer2)
        l1_error = l2_delta.dot(self.weights2.T)
        l1_delta = l1_error*self.sigmoidPrime(self.layer1)
        self.weights2 += self.layer1.T.dot(l2_delta)
        self.weights1 += self.training_inputs.T.dot(l1_delta)

    def train(self, i):
        self.predict(self.training_inputs)
        self.backward(i)

    def predict(self, input1): 
        self.layer1 = self.sigmoid(dot(input1, self.weights1))
        self.layer2 = self.sigmoid(dot(self.layer1, self.weights2))

    def output_predict(self, input1):
        self.predict(input1)
        return self.layer2

neural = NN()
for i in range(600000):
    neural.train(i)
a = [[0,0,0], [1,0,1], [0,1,0], [0,1,1], [1,0,0], [0,0,1], [1,1,0], [1,1,1]]
print("Predicted Output: " + str(neural.output_predict(a)))
print("Actual Output:" + str([1,1,1,0,0,0,0,0]))
plt.show()