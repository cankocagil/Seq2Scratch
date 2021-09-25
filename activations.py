import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

class Activations:
    """
    Necessary activation functions for recurrent neural network(RNN,LSTM,GRU).
    """
    def relu_alternative(self,X):
        """
            Rectified linear unit activation(ReLU).
        """
        return np.maximum(X, 0)

    def ReLU(self,X):
        """
            Rectified linear unit activation(ReLU).
            Most time efficient version.
        """
        return (abs(X) + X) / 2

    def relu_another(self,X):
        """
            Rectified linear unit activation(ReLU).
        """
        return X * (X > 0)

    def tanh(self,X):
        return np.tanh(X)

    def tanh_manuel(self,X):
        """
            Hyperbolic tangent activation(tanh).
        """      
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

    def sigmoid(self,X):
        """
            Sigmoidal activation.
        """
        c = np.clip(X,-700,700)
        return 1/(1 + np.exp(-c))

    def softmax(self,X):
        """
            Stable version of softmax classifier, note that column sum is equal to 1.
        """
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def softmax_stable(self,X):
        """
        Less stable version of softmax activation

        """
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x)

    def ReLUDerivative(self,X): 
        """
        The derivative of the ReLU function w.r.t. given input.
        """
        return 1 * (X > 0)

    def ReLU_grad(self,X):
        """
        The derivative of the ReLU function w.r.t. given input.
        """
        X[X<=0] = 0
        X[X>1] = 1
        return X

    def dReLU(self,X):  
        """
        The derivative of the ReLU function w.r.t. given input.
        """     
        return np.where(X <= 0, 0, 1)

    def dtanh(self,X): 
        """
        The derivative of the tanh function w.r.t. given input.
        """       
        return  1-(np.tanh(X)**2)

    def dsigmoid(self,X):
        """
        The derivative of the sigmoid function w.r.t. given input.
        """
        return self.sigmoid(X) * (1-self.sigmoid(X))    
    
    def softmax_stable_gradient(self,soft_out):           
        return soft_out * (1 - soft_out)

    def softmax_grad(self,softmax):        
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax_gradient(self,Sz):
        """Computes the gradient of the softmax function.
        z: (T, 1) array of input values where the gradient is computed. T is the
        number of output classes.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        
        # -SjSi can be computed using an outer product between Sz and itself. Then
        # we add back Si for the i=j cases by adding a diagonal matrix with the
        # values of Si on its diagonal.
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D


def sigmoid(x):
    c = np.clip(x,-700,700)
    return 1 / (1 + np.exp(-c))
def dsigmoid(y):
    return y * (1 - y)
def tanh(x):
    return np.tanh(x)
def dtanh(y):
    return 1 - y * y