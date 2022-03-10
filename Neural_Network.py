
from multiprocessing.sharedctypes import Value
import os
from re import I
from tkinter import NE, Y
import numpy as np
import math
import random
import sys
import time



instances = int(input("How Many Instances/Generation? "))

variability = 1 # The factor by which weights and biases are changed, greater variability works well with more instances.

inputNodesCount = 2 # The number of nodes in the input layer
hiddenNodesCount = 2 # The number of nodes in the hidden layers, must be at least two
outputNodesCount = 2 # The number of nodes in the output layer, must be at least two
hiddenLayersCount = 3 # The number of hidden layers, must be at least one

truthDict = { # The truth table by which the netwrok assesses itself. Looking for a better solution as this one is tiresome
    "[0 0]": [np.array([0, 0]), np.array([0, 0])],
    "[1 0]": [np.array([1, 0]), np.array([1, 0])],
    "[0 1]": [np.array([0, 1]), np.array([1, 0])],
    "[1 1]": [np.array([1, 1]), np.array([0, 1])]
}

inputSynapsesBaseline = [np.array([[0.5 for x in range(inputNodesCount)]for y in range(hiddenNodesCount)])]

hiddenSynapsesBaselines = []
for z in range(hiddenLayersCount): # Runs for each hidden layer
    hiddenSynapsesBaselines.append(np.array([[0.5 for x in range(hiddenNodesCount)]for y in range(outputNodesCount)])) # Adds an array of synapses to a list of baseline synapses
synapseBaselines = [inputSynapsesBaseline, hiddenSynapsesBaselines] # A list of lists of baseline synapse arrays

hiddenBiasesBaselines = []
for y in range(hiddenLayersCount): # Runs for each hidden layer
    hiddenBiasesBaselines.append(np.array([1 for x in range(hiddenNodesCount)])) # Adds an array of biases to a list of bias baseline arrays

outputBiasesBaseline = [np.array([1 for x in range(outputNodesCount)])] 
biasBaselines = [hiddenBiasesBaselines, outputBiasesBaseline] # A list of lists of arrays of bias baselines
 


class Neural_Network:
    #public
    avgCost = 0

    synapses = []

    inputLayer = []
    inputSynapses = []

    hiddenLayers = []
    hiddenSynapses = []
    hiddenBiases = []

    outputLayer = []
    outputBiases = []

    layers = []
    synapses = []
    biases = []
    
    #private
    __costs = []
    

    def __Sigmoid(self, num): # Squishes the inputted number to be between 0 and 1
        return 1/(1+(math.e ** -num))

    def __init__ (self, inputNodes, hiddenNodes, outputNodes, numHiddenLayers): # Runs on instance creation

        self.inputLayer = [np.array([0 for i in range(inputNodes)])] # Puts the input array in the input list
        
        self.hiddenLayers = np.array([[0 for j in range(hiddenNodes)] for x in range(numHiddenLayers)])
        '''for i in range(numHiddenLayers): # Runs for every hidden layer
            self.hiddenLayers.append(np.array([0 for j in range(hiddenNodes)])) # Initialises a hidden layer array into the list '''
        
        self.outputLayer = [np.array([0 for i in range(outputNodes)])] # Puts the output array into the output list

        self.layers = [self.inputLayer, self.hiddenLayers, self.outputLayer] # A list that contains lists of all the layers in the network
        
        
        self.inputSynapses = [np.array([[float(0) for x in range(inputNodes)] for y in range(hiddenNodes)])] # Puts the input synapses array (2d) into the input synapses list
        
        for i in range(numHiddenLayers - 1): # Runs for every hidden layer except the last one
            self.hiddenSynapses.append(np.array([[float(0) for x in range(hiddenNodes)] for y in range(hiddenNodes)])) # Adds an array of snapses for every hidden layer, except the last one
        self.hiddenSynapses.append(np.array([[float(0) for x in range(hiddenNodes)] for y in range(outputNodes)])) # Adds a final array with the synapses from the last hidden layer to the output layer
        
        self.synapses = [self.inputSynapses, self.hiddenSynapses] # A list that contains all the lists of synapses

        for i in range(numHiddenLayers): # Runs for every hidden layer
            self.hiddenBiases.append(np.array([float(1) for x in range(hiddenNodes)])) # Adds an array of biases for each hidden layer

        self.outputBiases = [np.array([float(1) for x in range(outputNodes)])] # initialises the output biases into a list

        self.biases = [self.hiddenBiases, self.outputBiases] # Puts all of the lists of biases into a list

        self.__correctAnswer = np.array([0 for i in range(outputNodes)]) # Initialises the dimensions in the correct answer array to the same as those in the output array

    #region Adjustments
    def adjustSynapses(self, synapseBaselines, adjRange): # Adjusts the synapses
        # Adjusts synapses from the input layer to the first hidden layer
        for i in range(len(self.layers[1][0])): # Runs for each neuron in the first hidden layer
            for j in range(len(self.layers[0][0])): # Runs for each neuron in the input layer
                base = synapseBaselines[0][0][i, j] # Sets base  to the coordinates (i, j) in the input synapses baselines array
                self.synapses[0][0][i, j] = random.uniform(base-adjRange, base+adjRange) # Sets the synapse at the coordinates (i, j) in the input synapses to something within the range
        
        # Adjusts synapses from first hidden layer to second hidden layer, second to third, ... ... third last to second last, second last to last hidden layer
        for i in range(len(self.layers[1]) - 1): # Runs for each hidden layer, minus one. This is so it doesn't attempt the connection from the last hidden layer to the output layer
            for j in range(len(self.layers[1][i + 1])): # Runs for each neuron in hidden layer i + 1
                for k in range(len(self.layers[1][i])): # Runs for each neuron in the hidden layer i
                    base = synapseBaselines[1][i][j, k] # Sets base to the coordinates (j, k) in the 'i'th hidden layer baseline synapses
                    self.synapses[1][i][j, k] = random.uniform(base-adjRange, base+adjRange) # Sets the synapse at the coordinates (j, k) in the 'i'th hidden layers synapses
        
        # Adjusts synapses from the last hidden layer to the output layer
        for i in range(len(self.layers[2][0])): # Runs for each neuron in the output layer
            for j in range(len(self.layers[1][-1])): # Runs for each neuron in the last hidden layer
                base = synapseBaselines[1][-1][i, j] # Sets base to the coordinates (i, j) in the last hidden layer's baseline synapses
                self.synapses[1][-1][i, j] = random.uniform(base-adjRange, base+adjRange) # Changes the synapse at coordinates (i, j) in the last hidden layer according to the range

    def adjustBiases(self, biasBaselines, adjRange): # Adjusts the biases
        for i in range(len(self.layers[1])): # Runs for each hidden layer
            for j in range(len(self.layers[1][i])): # Runs for each neuron in the "i"th hidden layer
                base = biasBaselines[0][i][j] # Sets base to the 'j'th item in the list of bias baselines for the 'i'th hidden layer
                self.biases[0][i][j] = random.uniform(base-adjRange, base+adjRange) # Changes the 'j'th item in the list of biases in the 'i'th hidden layer according to the range
        
        # Adjusts output biases
        for i in range(len(self.layers[2][0])): # Runs for each neuron in the output layer
            base = biasBaselines[1][0][i] # Sets base to the "i"th item in the 1d array of output layer bias baselines
            self.biases[1][0][i] = random.uniform(base-adjRange, base+adjRange) # Changes the 'i'th item in the output layer biases according to the range
    #endregion

    def __eval(self): # Finds the correct answer with the truth dictionary
        for i in range(len(self.layers[2][0])): # Runs for each neuron in the output layer
            self.__correctAnswer = truthDict.get(str(self.layers[0][0]))[1] # Adds the desired answer to the correct answer array

    def __cost(self): # Calculates the average cost of a network for a single input and returns it
        self.__eval() # Calculates the correct answers for the current input
        costs = [] # Empties the list of costs for this input
        for i in range(len(self.layers[2][0])): # Runs for each neuron in the output layer
            costs.append((self.layers[2][0][i] - self.__correctAnswer[i])**2) # Adds the cost for each neuron into the list of costs
        return (sum(costs)/len(costs)) # Returns the average of the costs of each neuron, this is the cost of the network for this input

    def thonk(self): # Runs a network, makes it "thonk"
        self.layers[1][0] = self.__Sigmoid(np.dot(self.synapses[0][0], self.layers[0][0]) + self.biases[0][0]) # Calculates the activations of the first hidden layer

        # Calculates the activations of the hidden layers
        for i in range(len(self.layers[1]) - 1): # Runs for each hidden layer, except the last one
            self.layers[1][i + 1] = self.__Sigmoid(np.dot(self.synapses[1][i], self.layers[1][i]) + self.biases[1][i]) # Calculates the activations of the 'i+1'th hidden layer

        self.layers[2][0] = self.__Sigmoid(np.dot(self.synapses[1][-1], self.layers[1][-1]) + self.biases[1][-1]) # Calculates the activations of the output layer
    
    def __total_cost(self): # Calculates the average cost of a network for all possible inputs
        self.__costs = [] # Empties the list of costs

        # Makes a list of costs for all inputs
        for key, value in truthDict.items():  # Runs for each key, and its values, in the truth dictionary
            self.layers[0][0] = value[0] # Sets the input layer to the value given by the truth dictionary
            self.thonk() # Thonks on the given input
            self.__costs.append(self.__cost()) # Gets the cost for the given input
        self.avgCost = (sum(self.__costs)/len(self.__costs)) # Takes a list of a networks cost for every possible input, and averages it 
    
    def learn(self): # Returns the average cost of the network
        self.__total_cost()
        return self.avgCost

networks = [Neural_Network(inputNodesCount, hiddenNodesCount, outputNodesCount, hiddenLayersCount) for x in range(instances)] # Creats an instance of the Neural_Network class for each instance the user wants in a generation
for i in range(instances): # Runs for each instance
    networks[i].adjustSynapses(synapseBaselines, 0.5 * variability) # Initialises the instance's synpase weights
    networks[i].adjustBiases(biasBaselines, 0.5 * variability) # Initialises the instance's biases

prevProg = 0 # Used in progrees percentage and estimated time to completion

while True:
    gens = int(input("How many generations do you want to iterate? "))
    start = time.time() # Starts timing the simulation
    for i in range(gens): # Runs for the number of generations specified

        best = None # Sets the best network to null
        minCost = 99 
        for j in range(instances): # Runs for each instance of the Neural_Network class
            if abs(networks[j].learn()) < minCost: # Runs if the average cost of the network is less than the current best
                minCost = networks[j].avgCost # Sets the best cost to the instance's average cost
                best = networks[j] # Saves the instance as the best one so far

        change = best.avgCost * variability # Change affects the amount the weights and biases change by. By useing average cost we make it so that changes become smaller the closest we are to an optimal network
        for j in range(instances): # Runs for each instance
            networks[j].adjustSynapses(best.synapses, change) # Adjusts all instances synapse weight's according to the weight's in the best iteration of the generation
            networks[j].adjustBiases(best.biases, change) # Adjusts all the instances biases according to the biases of the best iteration in the generation

        currentProg = (i / gens) * 100 # Sets currentProg to the percentage of generations complete
        if ((currentProg - prevProg) > (4000 / gens)): # Runs this loop only infrequently, less often for greater numbers of generations
            currentTime = time.time() 
            #os.system("cls") # Clears the terminal
            print(f"Progress: {currentProg}%") # Prints the percentage of progress through the simulation

            timePassed = (currentTime - start) # Calculates the time passed since this loop last ran
            progRemaining = 100 - currentProg # Calculates the percentage of progress remaining
            ratioRemaining = progRemaining / currentProg # Calculates the ratio from percentage remaining to percentage completed
            timeToComplete = timePassed * ratioRemaining # Calculates the estimated time till completion
            print(f"Time to Complete: {timeToComplete}") # Prints the estimated time until completion

            prevProg = currentProg # Sets the previous percentage to the current percentage

    end = time.time() # Stores the finishing time of the simulation
    print(f"Runtime = {end - start}") # Prints how long the simulation took to run
    
    #region Printing
    for key, value in truthDict.items(): # Runs for each key (and value) in the truth dictionary
        best.layers[0][0] = value[0] # Sets the input layer to a value in the truth dictionary 
        best.thonk() # Thonks on the input given
        print(f"With an input of {key} the network returned {best.layers[2][0]}.") # Tells the user what ouput is given for an input
    #endregion Printing