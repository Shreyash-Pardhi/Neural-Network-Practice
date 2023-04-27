<<<<<<< HEAD
import numpy as np

# definitions of activation functions

def mp(net, th):  # for MP neuron
    return 1 if net >= th else 0

def relu(net):
    return net if net > 0 else 0


def tlu(net):  # bipolar binary (discrete perceptron)
    return 1 if net > 0 else -1


def step(net):  # unipolar binary (discrete perceptron)
    return 1 if net > 0 else 0


def sigmoid(net, lm=1):  # Unipolar continuous perceptron
    return (1 / (1 + np.exp(-lm * net)))


def tanh(net, lm=1):  # bipolar continuous perceptron
    return 2 / (1 + np.exp(-lm * net)) - 1


class Neuron:
    def __init__(self,w):
        self.x = []
        self.w = w

    def net(self):
        nt = np.dot(self.x, self.w)
        return nt

class MPneuron(Neuron):
    ''' CLass for McCulloch Pit's Model of Neuron (uses mp activation function)'''
    def __init__(self,w,th):
        Neuron.__init__(self,w)
        self.th=th

    def out(self):
        nt=self.net()
        return mp(nt,self.th)

class dbPtron(Neuron):
    ''' CLass for Discrete Bipolar Perceptron (uses TLU activation function)'''

    def __init__(self, w):
        Neuron.__init__(self, w)

    def out(self):
        nt = self.net()
        return tlu(nt)

#define all input combinations for 3 binary inputs
X=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

#Create mp neuron object with appropriate weights and a threshold for AND gate
w=[1,1,1]  # weights
th=2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input AND Gate
print("\nPrinting Truth Table for 3-input AND Gate using MP neuron:")
print("------------------")
print("Input        AND")
print("------------------")
for x in X:
    mp1.x=x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NAND gate
w=[-1,-1,-1]  # weights
th=-2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NAND Gate
print("\nPrinting Truth Table for 3-input NAND Gate using MP neuron:")
print("------------------")
print("Input        NAND")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for OR gate
w=[1,1,1]  # weights
th=0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NAND Gate
print("\nPrinting Truth Table for 3-input OR Gate using MP neuron:")
print("------------------")
print("Input        OR")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOR gate
w=[-1,-1,-1]  # weights
th=-0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NOR Gate
print("\nPrinting Truth Table for 3-input NOR Gate using MP neuron:")
print("------------------")
print("Input        NOR")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOT gate
w=[-1,-1,-1]  # weights
Y=[[0,0,0],[1,1,1]]
th=-0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NOT Gate
print("\nPrinting Truth Table for 3-input NOT Gate using MP neuron:")
print("------------------")
print("Input        NOT")
print("------------------")
for x in Y:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")
=======
import numpy as np

# definitions of activation functions

def mp(net, th):  # for MP neuron
    return 1 if net >= th else 0

def relu(net):
    return net if net > 0 else 0


def tlu(net):  # bipolar binary (discrete perceptron)
    return 1 if net > 0 else -1


def step(net):  # unipolar binary (discrete perceptron)
    return 1 if net > 0 else 0


def sigmoid(net, lm=1):  # Unipolar continuous perceptron
    return (1 / (1 + np.exp(-lm * net)))


def tanh(net, lm=1):  # bipolar continuous perceptron
    return 2 / (1 + np.exp(-lm * net)) - 1


class Neuron:
    def __init__(self,w):
        self.x = []
        self.w = w

    def net(self):
        nt = np.dot(self.x, self.w)
        return nt

class MPneuron(Neuron):
    ''' CLass for McCulloch Pit's Model of Neuron (uses mp activation function)'''
    def __init__(self,w,th):
        Neuron.__init__(self,w)
        self.th=th

    def out(self):
        nt=self.net()
        return mp(nt,self.th)

class dbPtron(Neuron):
    ''' CLass for Discrete Bipolar Perceptron (uses TLU activation function)'''

    def __init__(self, w):
        Neuron.__init__(self, w)

    def out(self):
        nt = self.net()
        return tlu(nt)

#define all input combinations for 3 binary inputs
X=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

#Create mp neuron object with appropriate weights and a threshold for AND gate
w=[1,1,1]  # weights
th=2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input AND Gate
print("\nPrinting Truth Table for 3-input AND Gate using MP neuron:")
print("------------------")
print("Input        AND")
print("------------------")
for x in X:
    mp1.x=x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NAND gate
w=[-1,-1,-1]  # weights
th=-2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NAND Gate
print("\nPrinting Truth Table for 3-input NAND Gate using MP neuron:")
print("------------------")
print("Input        NAND")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for OR gate
w=[1,1,1]  # weights
th=0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NAND Gate
print("\nPrinting Truth Table for 3-input OR Gate using MP neuron:")
print("------------------")
print("Input        OR")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOR gate
w=[-1,-1,-1]  # weights
th=-0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NOR Gate
print("\nPrinting Truth Table for 3-input NOR Gate using MP neuron:")
print("------------------")
print("Input        NOR")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOT gate
w=[-1,-1,-1]  # weights
Y=[[0,0,0],[1,1,1]]
th=-0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NOT Gate
print("\nPrinting Truth Table for 3-input NOT Gate using MP neuron:")
print("------------------")
print("Input        NOT")
print("------------------")
for x in Y:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")
>>>>>>> refs/remotes/origin/master
