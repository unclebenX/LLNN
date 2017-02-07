import numpy as np
from copy import deepcopy
import random

class Node:
    function = None
    params = []
    inputNodes = []
    output = None

    def __init__(self, function, *args, **kargs):
        if 'inputNodes' in kargs:
            self.inputNodes = kargs['inputNodes']
        if 'params' in kargs:
            self.params = kargs['params']
        elif len(self.inputNodes)>0:
            self.params = [1./len(self.inputNodes) for i in range(len(self.inputNodes))]
        self.function = function
        return

    def process(self):
        for node in self.inputNodes:
            if node.output==None:
                node.process()
        inputVector = [node.output for node in self.inputNodes]
        self.output = self.function(inputVector, self.params)
        return

    def __str__(self):
        if self.function == linComb:
            func = "linComb"
        elif self.function == product:
            func = "product"
        elif self.function == const:
            func = "const"
        elif self.function == identity:
            func = "identity"
        else:
            func = str(self.function)
        return func + " " + str(self.params)

class Graph:
    nodes = []
    inputNodes = []
    outputNode = None
    alpha = 0.1
    h = 0.001
    nodeBatchSize = 4
    batchSize = 10

    def __init__(self, nodes, inputNodes, outputNode):
        self.nodes = nodes
        self.inputNodes = inputNodes
        self.outputNode = outputNode
        return

    def __str__(self):
        return str([str(node) for node in self.nodes])

    def reset(self):
        for node in self.nodes:
            node.output = None

    def process(self, inputData):
        self.reset()
        for i in range(len(inputData)):
            self.inputNodes[i].output = inputData[i]
        for node in G.nodes:
            self.outputNode.process()
        return self.outputNode.output

    def loss(self, inputBatch, outputBatch):
        loss = 0
        for i in range(len(inputBatch)):
            value = self.process(inputBatch[i])
            loss+=abs(value-outputBatch[i])
        return loss

    def gradient(self, nodes, inputBatch, outputBatch):
        gradientByNode = []
        for i in range(len(nodes)):
            gradientByNode.append([])
            for j in range(len(nodes[i].params)):
                initial_value = self.loss(inputBatch, outputBatch)
                nodes[i].params[j]+=self.h
                updated_value = self.loss(inputBatch, outputBatch)
                gradientByNode[i].append((updated_value-initial_value)/self.h)
                nodes[i].params[j]-=self.h
        return gradientByNode

    def gradientDescent(self, nodes, inputBatch, outputBatch):
        leftOver = False
        steps = 0
        while not leftOver and steps < 100:
            steps+=1
            gradientByNode = self.gradient(nodes, inputBatch, outputBatch)
            alpha = self.alpha
            originalValue = self.loss(inputBatch, outputBatch)
            for tries in range(20):
                for i in range(len(nodes)):
                    for j in range(len(nodes[i].params)):
                        nodes[i].params[j] -= alpha*gradientByNode[i][j]
                if self.loss(inputBatch,outputBatch)<originalValue:
                    break
                else:
                    for i in range(len(nodes)):
                        for j in range(len(nodes[i].params)):
                            nodes[i].params[j] += alpha*gradientByNode[i][j]
                alpha/=2
                if tries==19:
                    leftOver = True
        return

    def pick(self, l, rIndex):
        out = []
        for index in rIndex:
            out.append(l[index])
        return out

    def stochasticOptimization(self, inputBatch, outputBatch):
        batchSize = min(len(inputBatch), self.batchSize)
        nodeBatchSize = min(len(self.nodes), self.nodeBatchSize)
        for i in range(10):
            nodes = random.sample(self.nodes, nodeBatchSize)
            rIndex = random.sample(range(len(inputBatch)), batchSize)
            rInputBatch = G.pick(inputBatch, rIndex)
            rOutputBatch = G.pick(outputBatch, rIndex)
            self.gradientDescent(nodes, rInputBatch, rOutputBatch)
        return

def linComb(l, params):
    s = 0
    for i in range(len(l)):
        s+=l[i]
    return s

def product(l, params):
    prod = 1
    for e in l:
        prod*=e
    return prod

def const(l, params):
    #Single input, single parameter.
    return l[0] + params[0]

def identity(l, params):
    return l[0]

nonlin = lambda x : 1./(1+np.exp(-x))
