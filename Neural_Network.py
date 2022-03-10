
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
variability = 1



inputNodesCount = 2
hiddenNodesCount = 2
outputNodesCount = 1

truthDict = {
    "[0 0]": [np.array([0, 0]), np.array([0])],
    "[1 0]": [np.array([1, 0]), np.array([0])],
    "[0 1]": [np.array([0, 1]), np.array([0])],
    "[1 1]": [np.array([1, 1]), np.array([1])]
}

numIns = 0

inputSynapsesBaseline = np.array([[0.5 for x in range(inputNodesCount)]for y in range(hiddenNodesCount)])
hiddenSynapsesBaseline = np.array([[0.5 for x in range(hiddenNodesCount)]for y in range(outputNodesCount)])

hiddenBiasesBaseline = np.array([1 for x in range(hiddenNodesCount)])
outputBiasesBaseline = np.array([1 for x in range(outputNodesCount)])



class Neural_Network:
    #public
    avgCost = 0

    #private
    __costs = []

    def __Sigmoid(self, num):
        return 1/(1+(math.e ** -num))

    def __init__ (self, inputNodes, hiddenNodes, outputNodes):
        self.inputLayer = np.array([0 for i in range(inputNodes)])
        self.hiddenLayer = np.array([0 for i in range(hiddenNodes)])
        self.outputLayer = np.array([0 for i in range(outputNodes)])
        
        self.__correctAnswer = np.array([0 for i in range(outputNodes)])
        self.avgCost = 0

        self.inputSynapses = np.array([[float(0) for x in range(inputNodes)] for y in range(hiddenNodes)])
        self.hiddenSynapses = np.array([[float(0) for x in range(hiddenNodes)] for y in range(outputNodes)])

        self.hiddenBiases = np.array([float(1) for x in range(hiddenNodes)])
        self.outputBiases = np.array([float(1) for x in range(outputNodes)])

    #region Adjustments
    def adjustSynapses(self, inputBaseline, hiddenBaseline, adjRange):
        for i in range(len(self.hiddenLayer)):
            for j in range(len(self.inputLayer)):
                base = inputBaseline[i, j]
                self.inputSynapses[i, j] = random.uniform(base-adjRange, base+adjRange)
        
        for i in range(len(self.outputLayer)):
            for j in range(len(self.hiddenLayer)):
                base = hiddenBaseline[i, j]
                self.hiddenSynapses[i, j] = random.uniform(base-adjRange, base+adjRange)

    def adjustBiases(self, hiddenBaseline, outputBaseline, adjRange):
        for i in range(len(self.hiddenLayer)):
            base = hiddenBaseline[i]
            self.hiddenBiases[i] = random.uniform(base-adjRange, base+adjRange)
        
        for i in range(len(self.outputLayer)):
            base = outputBaseline[i]
            self.outputBiases[i] = random.uniform(base-adjRange, base+adjRange)
    #endregion

    def __eval(self):
        for i in range(len(self.outputLayer)):
            self.__correctAnswer[i] = truthDict.get(str(self.inputLayer))[1]

    def __cost(self):
        self.__eval()
        costs = []
        for i in range(len(self.outputLayer)):
            costs.append((self.outputLayer[i] - self.__correctAnswer[i])**2)
        return (sum(costs)/len(costs))

    def thonk(self):
        self.hiddenLayer = self.__Sigmoid(np.dot(self.inputSynapses, self.inputLayer) + self.hiddenBiases)
        self.outputLayer = self.__Sigmoid(np.dot(self.hiddenSynapses, self.hiddenLayer) + self.outputBiases)
    
    def __train(self):
        self.__costs = []
        
        for key, value in truthDict.items():
            self.inputLayer = value[0]
            self.thonk()
            self.__costs.append(self.__cost())
        

        self.avgCost = (sum(self.__costs)/len(self.__costs))
    
    def learn(self):
        self.__train()
        return self.avgCost

nw = Neural_Network(inputNodesCount, hiddenNodesCount, outputNodesCount)


networks = [Neural_Network(inputNodesCount, hiddenNodesCount, outputNodesCount) for x in range(instances)]
for i in range(instances):      
    networks[i].adjustSynapses(inputSynapsesBaseline, hiddenSynapsesBaseline, 0.5 * variability)
    networks[i].adjustBiases(hiddenBiasesBaseline, outputBiasesBaseline, 0.5 * variability)

while True:
    gens = input("How many generations do you want to iterate? ")
    start = time.time()
    for i in range(int(gens)):

        best = None
        minCost = 99
        for j in range(instances):
            if abs(networks[j].learn()) < minCost:
                minCost = networks[j].avgCost
                best = networks[j]
    

        change = best.avgCost * variability
        for j in range(instances):      
            networks[j].adjustSynapses(best.inputSynapses, best.hiddenSynapses, change)
            networks[j].adjustBiases(best.hiddenBiases, best.outputBiases, change)


        os.system("cls")
        print(f"Progress: {(i / int(gens)) * 100}%")


    end = time.time()
    print(f"Runtime = {end - start}")
    
    #region Printing
    for key, value in truthDict.items():
        best.inputLayer = value[0]
        best.thonk()
        print(f"With an input of {key} the network returned {best.outputLayer}.")
    #endregion Printing