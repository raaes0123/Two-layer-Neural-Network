import numpy as np
import math
import matplotlib.pyplot as plt

# Two layer neural network
class neural_net:
    def __init__(self,in_layer,hid_layer,out_layer):
        self.in_layer = in_layer
        self.hid_layer = hid_layer
        self.out_layer = out_layer
        self.x = np.ndarray(in_layer)
        self.z1 = np.ndarray(hid_layer)
        self.z2 = np.ndarray(out_layer)
        self.a1 = np.ndarray(hid_layer)
        self.a2 = np.ndarray(out_layer)
        self.w1 = np.random.rand(in_layer,hid_layer)
        self.w2 = np.random.rand(hid_layer,out_layer)
        self.b1 = np.zeros(hid_layer)
        self.b2 = np.zeros(out_layer)
        self.lr = 0.1

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    def feedforward(self,data):
        self.x = np.array(data)
        self.z1 = self.x.dot(self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

    def gradient(self,y):
        delW2 = np.ndarray(self.w2.shape)
        delb2 = np.ndarray(self.out_layer)
        for j in range(self.hid_layer):
            for k in range(self.out_layer):
                #delta = (y[k] - self.a2[k])*self.sigmoid_prime(self.z2[k])
                delta = (y[k] - self.a2[k])*self.a2[k]*(1-self.a2[k])
                delb2[k] = delta
                delW2[j][k] = delta*self.a1[j]

        delW1 = np.ndarray(self.w1.shape)
        delb1 = np.ndarray(self.hid_layer)
        for i in range(self.in_layer):
            for j in range(self.hid_layer):
                delta = 0
                for k in range(self.out_layer):
                    #delta += (y[k] - self.a2[k])*self.sigmoid_prime(self.z2[k])*self.w2[j][k]
                    delta += (y[k] - self.a2[k])*self.a2[k]*(1-self.a2[k])*self.w2[j][k]
                delb1[j] = delta*self.a1[j]*(1-self.a1[j])
                delW1[i][j] = delb1[j]*self.x[i]
        
        self.w1 += delW1*self.lr
        self.w2 += delW2*self.lr
        self.b1 += delb1*self.lr
        self.b2 += delb2*self.lr

    def train(self,data,target):
        e = []
        for i in range(len(data)):
            self.feedforward(data[i])
            self.gradient(target[i])
            error = 1/2*np.sum(target[i] - self.a2)**2
            e.append(error)
        return e
        
            
nn = neural_net(2,2,1)
data = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[1],[1],[0]])
err = []
for _ in range(10000):
    err.extend(nn.train(data,target))
plt.plot(err,scaley = False)
plt.show()
test_data = [[0,0],[0,1],[1,0],[1,1]]
for i in range(len(test_data)):
    nn.feedforward(test_data[i])
    print(nn.a2)
