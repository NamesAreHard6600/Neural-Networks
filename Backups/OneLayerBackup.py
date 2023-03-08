import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random

def show():
    plt.pause(.0000000000000000000001)


def inList(arr_to_check_for, arr_to_check):
    for element in arr_to_check:
        if np.array_equal(element, arr_to_check_for):
            return True
    return False

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([-0.2, 0.1])
        self.biases = 0
        self.learning_rate = learning_rate
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
        
    def plot_data(self,tests):
        plt.clf()
        numbers = np.arange(MIN,MAX,STEP//2)
        combos = [[x,y] for x in numbers for y in numbers]
        for x in combos:
            color = "."
            if inList(x,tests[:,0:-1]):
                color = "^"
            if self.guess(self.predict(x)):
                color = "g" + color
            else:
                color = "r" + color
            
            plt.plot(x[0],x[1],color)
        '''
        for x in tests:
            if self.guess(self.predict(x)):
                plt.plot(x[0],x[1],"g^")
            else:
                plt.plot(x[0],x[1],"r^")
        '''
        show()
        
    def predict(self, input_vector):
        input_vector = input_vector[:2]
        layer_1 = np.dot(input_vector,self.weights) + self.biases
        layer_2 = self._sigmoid(layer_1)
        return layer_2
        
    def updateParameters(self,update_weights, update_bias):
        self.weights -= (update_weights)*self.learning_rate
        self.biases -= (update_bias)*self.learning_rate*5 #I found my bias moved to slow
    
    def calc_error(self,input_vector): #Calculates the error and returns it
        target = input_vector[-1]
        prediction = self.predict(input_vector[:2])
        error = np.square(prediction-target)
        return target, prediction, error
    
    def learn(self, input_vector):
        target, prediction, error = self.calc_error(input_vector)
        layer_1 = np.dot(input_vector[:2],self.weights) + self.biases
        #print(f"Weights: {self.weights}")
        
        derror_dprediction = (prediction-target)*2 #Derivative of the error
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        #print(dprediction_dlayer1)
        #dlayer1_dbias = 1 Not needed since it's one
        derror_dbias = (derror_dprediction * dprediction_dlayer1)
        
        dlayer1_dweights = (1 * input_vector[:2]) # + (0 * self.weights)
        derror_dweights = (
            np.multiply(derror_dprediction * dprediction_dlayer1, dlayer1_dweights)
        )
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
                    _, _, error = self.calc_error(x)
                    cumlative_error += error
                cumlative_errors.append(cumlative_error)
                if len(cumlative_errors) > 2 and cumlative_errors[-1] == cumlative_errors[-2]:
                    print("Flatlinning")
                
                #plt.plot(itera,cumlative_error,"g.")
                self.plot_data(tests)
        return cumlative_errors
    
    def guess(self, prediction): #Guess  
        #True = poison
        return prediction < 0.5
        
    

#input1, input2, target 1 = poison
MIN = 0
MAX = 100
STEP = 10
inputs_safe = np.array([np.array([x,y,0]) for x in np.arange(MIN,MAX,STEP) for y in np.arange(MIN,MAX,STEP) if x > y])
inputs_poison = np.array([np.array([x,y,1]) for x in np.arange(MIN,MAX,STEP) for y in np.arange(MIN,MAX,STEP) if y > x])

# What it should look likeplot
for x in inputs_safe:
    plt.plot(x[0],x[1],"go")
for x in inputs_poison:
    plt.plot(x[0],x[1],"ro")
plt.pause(2)

#inputs_safe = np.vstack((inputs_safe,[3,7.5,0]))
#print(inputs_safe)



network = NeuralNetwork(.1)

tests = np.array(np.concatenate((inputs_safe,inputs_poison)))


#random_data_index = np.random.randint(len(input_vectors))

errors = network.train(tests,1000)
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
'''
'''
for x in tests:
    if network.guess(network.predict(x)):
        plt.plot(x[0],x[1],"g^")
    else:
        plt.plot(x[0],x[1],"r^")
'''
plt.show()

