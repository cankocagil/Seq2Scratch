import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd 
import seaborn as sns

class Optimizers:

    def __init__(self, num_weight, learning_rate):
        # To keep previous updates in momentum :
        self.previous_updates = [0] * num_weight
        
        # For AdaGrad:
        self.cache = [0] * num_weight     
        self.cache_rmsprop = [0] * num_weight
        self.m = [0] * num_weight
        self.v = [0] * num_weight
        self.t = 1
        self.learning_rate = learning_rate

    def SGD(self,  params, grads):
      """

      Stochastic gradient descent with momentum on mini-batches.
      """
      prevs = []
     
      for param, grad, prev_update in zip(self.params(), grads, self.previous_updates):            
          delta = self.learning_rate * grad - self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)
     
      self.previous_updates = prevs     


    def AdaGrad(self, params, grads):
      """
      AdaGrad adaptive optimization algorithm.
      """         
      i = 0
      for param,grad in zip( params, grads):
        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)
        i += 1


    def RMSprop(self, params, grads,decay_rate = 0.9):
      """

      RMSprop adaptive optimization algorithm
      """


      i = 0
      for param,grad in zip( params, grads):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        i += 1


    def VanillaAdam(self, params, grads,beta1 = 0.9,beta2 = 0.999):
        """
        Adam optimizer, but bias correction is not implemented
        """
        i = 0

        for param,grad  in zip(params,grads):

          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
          i += 1


    def Adam(self, params, grads, beta1 = 0.9,beta2 = 0.999):
        """

        Adam optimizer, bias correction is implemented.
        """
      
        i = 0

        for param, grad  in zip(params, grads):
          
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
          i += 1
          
        self.t +=1