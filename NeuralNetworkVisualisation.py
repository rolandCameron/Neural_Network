import pygame
import numpy as np
import math

pygame.init

global numLayers
global layerWidth
global neuronRadii

numLayers = 0
layerWidth = 0
neuronRadii = 0

def initialise(NeuralNetwork, screenDimensions):
    global numLayers
    global layerWidth
    global neuronRadii

    screenX, screenY = screenDimensions[0], screenDimensions[1]
    numLayers = len(NeuralNetwork)

    layerWidth = screenX/numLayers
    neuronRadii = []
    for i in range(numLayers):
        neuronRadii.append((screenY*0.9)/len(NeuralNetwork[i].neurons))

def visualise(NeuralNetwork):
    for i in range(len(NeuralNetwork)):
        activations = NeuralNetwork[i].returnActivations()

        activations = 1/1 + (math.e ** -activations)