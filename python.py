import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random

def show():
    plt.pause(.0000000000000000000001)

class Layer:
    def __init__(self,inputs,outputs):
        self.neurons = outputs
        self.weights = np.array([[0.5 for _ in range(inputs)] for _ in range(outputs)])
        self.biases = np.array([0.5 for _ in range(outputs)])
        self.data = np.array([0.5 for _ in range(outputs)])
        '''EDIT
        self.weights = np.array([[random.random() for _ in range(inputs)] for _ in range(outputs)])
        self.biases = np.array([random.random() for _ in range(outputs)])
        self.data = np.array([random.random() for _ in range(outputs)])
        '''
        

class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        self.layers = [Layer(x[0],x[1]) for x in layers]
        self.output_layer = self.layers[-1]
        self.learning_rate = learning_rate
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
        
    def plot_data(self,tests):
        plt.clf()
        for x in tests:
            #target = input_vector[-1]
            curr = x[:-1]
            if self.guess(self.predict(curr)):
                plt.plot(x[0],x[1],"g^")
            else:
                plt.plot(x[0],x[1],"r^")
        show()
        
    def predict(self, input_vector):
        for layer in self.layers:
            for neuron in range(layer.neurons):
                #print(layer.weights[neuron]) #Same size but it hates me, trying 3 inputs instead of 2
                #print(input_vector)
                layer_1 = np.dot(input_vector,layer.weights[neuron]) + layer.biases[neuron]
                layer_2 = self._sigmoid(layer_1)
                layer.data[neuron] = layer_2
            input_vector = layer.data
        return input_vector
        
    def updateParameters(self,update_weights, update_bias):
        layer = self.layers[0]
        layer.weights -= (update_weights)*self.learning_rate
        layer.biases -= (update_bias)*self.learning_rate*5 #I found my bias moved to slow
    
    def calc_error(self,input_vector,target): #Calculates the error and returns it
        prediction = self.predict(input_vector)[0] #Assumes one output
        error = np.square(prediction-target)
        return prediction, error
    
    def learn(self, input_vector):
        target = input_vector[-1]
        input_vector = input_vector[:-1]
        prediction, error = self.calc_error(input_vector,target)
        #Assumes 1 layer
        layer = self.layers[0]
        print(f"Weights: {layer.weights}")
        print(f"Biases: {layer.biases}")
        print(f"Error: {error}")
        layer_1 = np.dot(input_vector,layer.weights[0]) + layer.biases[0]
        
        
        derror_dprediction = (prediction-target)*2 #Derivative of the error
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        #dlayer1_dbias = 1 Not needed since it's one
        derror_dbias = (derror_dprediction * dprediction_dlayer1)
        
        dlayer1_dweights = (1 * input_vector) # + (0 * layer.weights)
        derror_dweights = (
            np.multiply(derror_dprediction * dprediction_dlayer1, dlayer1_dweights)
        )
        #print(derror_dprediction)
        #print(f"Errors: {derror_dweights}")
        self.updateParameters(derror_dweights,derror_dbias) #Update Parameters

        return prediction
        
    def train(self,tests,iterations): #Trains on one then calculates overall error, over any number of iterations
        cumlative_errors = []
        for itera in range(iterations):
            random_data_index = np.random.randint(len(tests))
            input_vector = tests[random_data_index]
            self.learn(input_vector)
            if itera % 100 == 0: #Observe improvements every 100
                cumlative_error = 0
                for x in tests:
                    target = x[-1]
                    curr = x[:-1]
                    _, error = self.calc_error(curr,target)
                    cumlative_error += error
                cumlative_errors.append(cumlative_error/len(tests))
                if len(cumlative_errors) > 2 and cumlative_errors[-1] == cumlative_errors[-2]:
                    print("Flatlinning")
                
                #plt.plot(itera,cumlative_error,"g.")
                self.plot_data(tests)
        return cumlative_errors
    
    def guess(self, prediction): #Guess  
        #True = poison
        return prediction < 0.5



#input1, input2, target 1 = poison
inputs_safe = np.array([np.array([x,y,0]) for x in np.arange(0,100,10) for y in np.arange(0,100,10) if x > y])
inputs_poison = np.array([np.array([x,y,1]) for x in np.arange(0,100,10) for y in np.arange(0,100,10) if y > x])

# What it should look likeplot
for x in inputs_safe:
    plt.plot(x[0],x[1],"go")
for x in inputs_poison:
    plt.plot(x[0],x[1],"ro")
plt.pause(1)

#inputs_safe = np.vstack((inputs_safe,[3,7.5,0]))
#print(inputs_safe)
#A list of layer data, [inputs, output]
layers = [
[2,1]
]

network = NeuralNetwork(layers, .1)

tests = np.array(np.concatenate((inputs_safe,inputs_poison)))


#random_data_index = np.random.randint(len(input_vectors))

errors = network.train(tests,1)
#print(errors)
plt.pause(2)
plt.clf()
plt.plot(errors)


'''
count = 0
while(True):
    if count%2 == 1:
        random_data_index = np.random.randint(len(tests))
        x = tests[random_data_index]
        network.learn(x)
        for x in tests:
            if network.guess(network.predict(x)):
                plt.plot(x[0],x[1],"g^")
            else:
                plt.plot(x[0],x[1],"r^")
    else:
        for x in inputs_safe:
            plt.plot(x[0],x[1],"go")
        for x in inputs_poison:
            plt.plot(x[0],x[1],"ro")
        plt.pause(5)
    plt.pause(0.00001)
    plt.clf()
    if count != 1:
        count+=1

for x in tests:
    if network.guess(network.predict(x)):
        plt.plot(x[0],x[1],"g^")
    else:
        plt.plot(x[0],x[1],"r^")
'''
plt.show()

