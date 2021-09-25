import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd 
import seaborn as sns
from activations import Activations
from metrics import Metrics

def sigmoid(x):
    c = np.clip(x,-700,700)
    return 1 / (1 + np.exp(-c))
def dsigmoid(y):
    return y * (1 - y)
def tanh(x):
    return np.tanh(x)
def dtanh(y):
    return 1 - y * y


class Multi_layer_GRU(object):
    """

    Gater recurrent unit, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
    """

    def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 = 64,output_class = 6,seq_len = 150,batch_size = 32,learning_rate = 1e-1,mom_coeff = 0.85):
        """

        Initialization of weights/biases and other configurable parameters.
        
        """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

       
        
        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim_1)
        lim1_hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
        self.W_z = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_z = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_z = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_r = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_r = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_r = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_h = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_h = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_h = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        lim2_hid = Xavier(self.hidden_dim_1,self.hidden_dim_2)
        self.W_hid = np.random.uniform(-lim2_hid,lim2_hid,(self.hidden_dim_1,self.hidden_dim_2))
        self.B_hid = np.random.uniform(-lim2_hid,lim2_hid,(1,self.hidden_dim_2))
        
        lim2 = Xavier(self.hidden_dim_2,self.output_class)
        self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim_2,self.output_class))
        self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
        
        # To keep previous updates in momentum :
        self.previous_updates = [0] * 13
        
        # For AdaGrad:
        self.cache = [0] * 13  
        self.cache_rmsprop = [0] * 13
        self.m = [0] * 13
        self.v = [0] * 13
        self.t = 1

    def cell_forward(self,X,h_prev):
        

        # Update gate:
        update_gate = activations.sigmoid(np.dot(X,self.W_z) + np.dot(h_prev,self.U_z) + self.B_z)
       
        # Reset gate:
        reset_gate = activations.sigmoid(np.dot(X,self.W_r) + np.dot(h_prev,self.U_r) + self.B_r)

        # Current memory content:
        h_hat = np.tanh(np.dot(X,self.W_h) + np.dot(np.multiply(reset_gate,h_prev),self.U_h) + self.B_h)

        # Hidden state:
        hidden_state = np.multiply(update_gate,h_prev) + np.multiply((1-update_gate),h_hat)

        # Hidden MLP:
        hid_dense = np.dot(hidden_state,self.W_hid) + self.B_hid
        relu = activations.ReLU(hid_dense)

        # Classifiers (Softmax) :
        dense = np.dot(relu,self.W) + self.B
        probs = activations.softmax(dense)

        return (update_gate,reset_gate,h_hat,hidden_state,hid_dense,relu,dense,probs)

        

    def forward(self,X,h_prev):
        x_s,z_s,r_s,h_hat = {},{},{},{}
        h_s = {}
        hd_s,relu_s = {},{}
        y_s,p_s = {},{}        

        h_s[-1] = np.copy(h_prev)
        

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]
            z_s[t], r_s[t], h_hat[t], h_s[t],hd_s[t],relu_s[t], y_s[t], p_s[t] = self.cell_forward(x_s[t],h_s[t-1])

        return (x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s)
    
    def BPTT(self,outs,Y):

        x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s = outs

        dW_z, dW_r,dW_h, dW = np.zeros_like(self.W_z), np.zeros_like(self.W_r), np.zeros_like(self.W_h),np.zeros_like(self.W)
        dW_hid = np.zeros_like(self.W_hid)
        dU_z, dU_r,dU_h = np.zeros_like(self.U_z), np.zeros_like(self.U_r), np.zeros_like(self.U_h)


        dB_z, dB_r,dB_h,dB = np.zeros_like(self.B_z), np.zeros_like(self.B_r),np.zeros_like(self.B_h),np.zeros_like(self.B)
        dB_hid = np.zeros_like(self.B_hid)
        dh_next = np.zeros_like(h_s[0]) 
           

        # w.r.t. softmax input
        ddense = np.copy(p_s[149])
        ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
        #ddense[np.argmax(Y,1)] -=1
        #ddense = y_s[149] - Y
        # Softmax classifier's :
        dW = np.dot(relu_s[149].T,ddense)
        dB = np.sum(ddense,axis = 0, keepdims = True)

        ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(hd_s[149])
        dW_hid = np.dot(h_s[149].T,ddense_hid)
        dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)

   
        # Backprop through time:
        for t in reversed(range(1,self.seq_len)):           

            # Curernt memort state :
            dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
            dh_hat = dh * (1-z_s[t])
            dh_hat = dh_hat * dtanh(h_hat[t])
            dW_h += np.dot(x_s[t].T,dh_hat)
            dU_h += np.dot((r_s[t] * h_s[t-1]).T,dh_hat)
            dB_h += np.sum(dh_hat,axis = 0, keepdims = True)

            # Reset gate:
            dr_1 = np.dot(dh_hat,self.U_h.T)
            dr = dr_1  * h_s[t-1]
            dr = dr * dsigmoid(r_s[t])
            dW_r += np.dot(x_s[t].T,dr)
            dU_r += np.dot(h_s[t-1].T,dr)
            dB_r += np.sum(dr,axis = 0, keepdims = True)

            # Forget gate:
            dz = dh * (h_s[t-1] - h_hat[t])
            dz = dz * dsigmoid(z_s[t])
            dW_z += np.dot(x_s[t].T,dz)
            dU_z += np.dot(h_s[t-1].T,dz)
            dB_z += np.sum(dz,axis = 0, keepdims = True)


            # Nexts:
            dh_next = np.dot(dz,self.U_z.T) + (dh * z_s[t]) + (dr_1 * r_s[t]) + np.dot(dr,self.U_r.T)


        # List of gradients :
        grads = [dW,dB,dW_hid,dB_hid,dW_z,dU_z,dB_z,dW_r,dU_r,dB_r,dW_h,dU_h,dB_h]
              
        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1],grads
    


    def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)

            # Equate 0 in every epoch:           
            h_prev = np.zeros((self.batch_size,self.hidden_dim_1))

            for i in range(round(X.shape[0]/self.batch_size) - 1): 
               
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
               
                # Forward + BPTT + Optimization:
                cache_train = self.forward(X_feed,h_prev)
                h,grads = self.BPTT(cache_train,y_feed)

                if optimizer == 'SGD':                                                                
                  self.SGD(grads)

                elif optimizer == 'AdaGrad' :
                  self.AdaGrad(grads)

                elif optimizer == 'RMSprop':
                  self.RMSprop(grads)
                
                elif optimizer == 'VanillaAdam':
                  self.VanillaAdam(grads)
                else:
                  self.Adam(grads)

                # Hidden state -------> Previous hidden state
                h_prev = h

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
            _,__,___,____,_____,______,_______,________,probs_test = self.forward(X_val,test_prevs)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
            predictions_val = np.argmax(probs_test[149],1)
            acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)
      
    
    def params(self):
        """
        Return all weights/biases in sequential order starting from end in list form.

        """        
        return [self.W,self.B,self.W_hid,self.B_hid,self.W_z,self.U_z,self.B_z,self.W_r,self.U_r,self.B_r,self.W_h,self.U_h,self.B_h]

    def SGD(self,grads):
      """

      Stochastic gradient descent with momentum on mini-batches.
      """
      prevs = []
      
      for param,grad,prev_update in zip(self.params(),grads,self.previous_updates): 
                     
          delta = self.learning_rate * grad + self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)
        

      self.previous_updates = prevs     
      self.learning_rate *= 0.99999   

    
    def AdaGrad(self,grads):
      """
      AdaGrad adaptive optimization algorithm.
      """      

      i = 0
      for param,grad in zip(self.params(),grads):

        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

        i += 1


    def RMSprop(self,grads,decay_rate = 0.9):
      """
      RMSprop adaptive optimization algorithm
      """
      i = 0
      for param,grad in zip(self.params(),grads):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        i += 1


    def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """
        Adam optimizer, but bias correction is not implemented
        """
        i = 0

        for param,grad  in zip(self.params(),grads):

          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
          i += 1


    def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """

        Adam optimizer, bias correction is implemented.
        """
      
        i = 0

        for param,grad  in zip(self.params(),grads):
          
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
          i += 1
          
        self.t +=1
    
    
    def CategoricalCrossEntropy(self,labels,preds):
        """
        Computes cross entropy between labels and model's predictions
        """
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N
    
    def predict(self,X):
        """
        Return predictions, (not one hot encoded format)
        """

        # Give zeros to hidden states:
        pasts = np.zeros((X.shape[0],self.hidden_dim_1))
        _,__,___,____,_____,______,_______,________,probs = self.forward(X,pasts)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}     

if __name__ == '__main__':
    multi_layer_gru = Multi_layer_GRU(hidden_dim_1=128,hidden_dim_2=64,learning_rate=1e-3,mom_coeff=0.0,batch_size=32)
    multi_layer_gru.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer = 'RMSprop')