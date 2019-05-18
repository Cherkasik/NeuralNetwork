from numpy import exp, array, random, dot, transpose, zeros, mean, amax, cos, square

class NN:
    def __init__(self):
        random.seed(1)
        self.sigmoid = lambda x: 1/(1+exp(-x))
        self.function = lambda x: exp(x)/cos(x)
        self.sigmoidPrime = lambda x: x * (x - 1)
        self.training_inputs = []
        self.training_outputs = []
        self.weights1 = 2*random.random((1, 3)) - 1
        self.weights2 = 2*random.random((3, 1)) - 1
        self.error = 0
        self.adjustments = 0

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
        self.maxi = amax(abs(self.training_outputs))
        self.training_outputs /= amax(abs(self.training_outputs), axis=0)

    def backward(self):
        l2_error = self.training_outputs - self.layer2
        l2_delta = l2_error*self.sigmoidPrime(self.layer2)
        l1_error = l2_delta.dot(self.weights1)
        l1_delta = dot(l1_error, self.sigmoidPrime(self.layer1))
        self.weights2 += self.layer1.T.dot(l2_delta)
        self.weights1 += self.training_inputs.T.dot(l1_delta)

    def train(self):
        self.predict(self.training_inputs)
        self.backward()

    def predict(self, input1): 
        self.layer1 = self.sigmoid(dot(input1, self.weights1))
        self.layer2 = self.sigmoid(dot(self.layer1, self.weights2))

    def output_predict(self, input1):
        self.predict(input1)
        return self.layer2*self.maxi

neural = NN()
neural.generate_set(3)
for i in range(1310): #1310 perfect for 1
    neural.train()
    #print ("Loss: \n" + str(mean(square(neural.training_outputs - neural.output_predict(neural.training_inputs)))))
a = 1
print("Predicted Output: " + str(neural.output_predict(a)))
print("Actual Output:" + str(neural.function(a)))
