# -*- coding: utf-8 -*
"""
Created on Tue Oct  9 22:38:10 2018

@author: raphaelfeijao
"""
import numpy as np
import matplotlib.pyplot as plt

def func (X):
    return np.sin(X)

def activationFunction (X):
    return np.tanh(X)

def derivateActivationFunction (X):
    return 1 - np.tanh(np.array(X))**2

def hiddenValue (W, Z):
    #Retorna os valores de z
    return activationFunction(np.dot(Z,W))

def sigma (X):
    return 1/(1+np.exp(X))

def priorW (n):
    return np.random.rand(n)

def nLayers(topology):
    #topology is a list with what we want
    if (len(topology) >= 3):       
        return len(topology[1:-1])
    else:
        print ("Your topology " + repr (topology) + " is not right")
        exit (-1)

def createWandZ (topology):
    W = list(np.arange(np.sum(topology[:-1])))
    m = []
    aux = 0
    Z= []
    for i in np.arange(nLayers(topology) + 1):
        m.append([])
        for j in np.arange(topology[i]):
            W[aux+j] = (priorW(topology[i+1]))
        m[i] = np.array(W[aux:aux+topology[i]])
        aux += topology[i]
        Z.append(np.zeros(topology[i]))

    return m, Z

def RMS (Y, T):
    return 0.5 * np.sum((Y - T)**2)

def Delta_j (W, Z, j, delta_k):
    a = np.zeros(len(Z[j]))
    delt = np.zeros(len(Z[j]))
    for i in np.arange(len(a)):
        a[i] = np.sum(Z[j][i]*W[j][i])
        delt[i] = np.sum(W[j][i]*delta_k)
    h = derivateActivationFunction(a)
    return h * delt

def updateWeight(W, Z, eta, delta_j_1):
    for j in np.arange(len(W)-1)[::-1]:
        delta_j = Delta_j(W, Z, j, delta_j_1)
        delta_j_1 = delta_j
        delta_k = eta*(delta_j)*Z[j]
        if (len(delta_k == 1)): delta_k = delta_k[np.newaxis]
        W[j] = W[j] - np.transpose(delta_k)
        
######"Main"#######
topology = np.array([1, 20, 1])
#X = Dados experimentais seguindo algum padrão
#X = np.matrix('1 1; 0 1; 0 0; 0 1; 0 0; 0 1; 1 1; 1 0; 1 1; 1 0; 0 0; 1 1; 0 0; 0 1; 1 0; 1 1; 1 0; 0 1; 1 0') 
#T = np.matrix('0 ; 1 ; 0 ; 1; 0 ; 1; 0; 1; 0; 1; 0; 0; 0; 1; 1; 0; 1; 1; 0')

N = 5000
N_graph = int (49*N/50)
r = 3
X = r*np.random.rand(N,1) 
T = func (X)
eta = 0.1

if (len(X) != len (T) and len(X[0]) != topology[0]):
    print ("Something is wrong with your DB")
    exit(-1)
    
Interactions = len (X)
N_Layers = nLayers(topology)
W , Z = createWandZ (topology)

Error = []
Y = np.empty(topology[-1])
Y_graph = []


for i in np.arange(Interactions):
    #Forward Propagation
    print (i)
    Z[0] = X[i]
    for j in np.arange(1, N_Layers + 1):
            Z[j] = hiddenValue(W[j-1], Z[j-1])
        #print (Z[j])
    #print ("W")
    #print (W)
    #print ("Z")
    #print (Z)
    Y = np.dot(Z[N_Layers],W[N_Layers])
    print(Y)
    #Backwards Propagation
    Delta_i = Y - T[i]
    #print(Y_i)
    delta_k = eta*Delta_i*Z[-1]
    if (len(delta_k == 1)): delta_k = delta_k[np.newaxis]
    W[-1] = W[-1] - np.transpose(delta_k)
    updateWeight(W, Z, eta, Delta_i)     
    if (RMS(Y,T[i])<0.5):
        Error.append( RMS(Y, T[i]))
    else: Error.append(0.5)
    #print("error = " + repr(Error[i]))
    if (i>N_graph):
        Y_graph.append(Y[0])
    
plt.plot(np.arange(Interactions), Error,  label="Error")
plt.legend(loc="best")
plt.show()

plt.plot(X[N_graph + 1 : Interactions], Y_graph, 'bo', label="Sin")
#plt.plot(X[Interactions-9 : Interactions], Y_graph, 'bo', label="Sin")
plt.plot(np.linspace(0,r,1000), func (np.linspace(0,r,1000)), color = 'r')
plt.legend(loc="best")
plt.show()



