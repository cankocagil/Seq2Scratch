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


class Multi_Layer_LSTM(object):
    """

    Long-Short Term Memory Recurrent neural network, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
    """

    def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 =64,output_class = 6,seq_len = 150,batch_size = 30,learning_rate = 1e-1,mom_coeff = 0.85):
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

        self.input_stack_dim = self.input_dim + self.hidden_dim_1
        
        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim_1)
        self.W_f = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_f = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_i = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_i = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_c = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_c = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_o = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_o = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))
        
        lim2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
        self.W_hid = np.random.uniform(-lim2,lim2,(self.hidden_dim_1,self.hidden_dim_2))
        self.B_hid = np.random.uniform(-lim2,lim2,(1,self.hidden_dim_2))

        lim3 = Xavier(self.hidden_dim_2,self.output_class)
        self.W = np.random.uniform(-lim3,lim3,(self.hidden_dim_2,self.output_class))
        self.B = np.random.uniform(-lim3,lim3,(1,self.output_class))

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

    def cell_forward(self,X,h_prev,C_prev):
        """

        Takes input, previous hidden state and previous cell state, compute:
        --- Forget gate + Input gate + New candidate input + New cell state + 
            output gate + hidden state. Then, classify by softmax.
        """
        #print(X.shape,h_prev.shape)
        # Stacking previous hidden state vector with inputs:
        stack = np.column_stack([X,h_prev])

        # Forget gate:
        forget_gate = activations.sigmoid(np.dot(stack,self.W_f) + self.B_f)
       
        # İnput gate:
        input_gate = activations.sigmoid(np.dot(stack,self.W_i) + self.B_i)

        # New candidate:
        cell_bar = np.tanh(np.dot(stack,self.W_c) + self.B_c)

        # New Cell state:
        cell_state = forget_gate * C_prev + input_gate * cell_bar

        # Output fate:
        output_gate = activations.sigmoid(np.dot(stack,self.W_o) + self.B_o)

        # Hidden state:
        hidden_state = output_gate * np.tanh(cell_state)

        # Classifiers (Softmax) :
        dense_hid = np.dot(hidden_state,self.W_hid) + self.B_hid
        act = activations.ReLU(dense_hid)

        dense = np.dot(act,self.W) + self.B
        probs = activations.softmax(dense)

        return (stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs,dense_hid,act)

        

    def forward(self,X,h_prev,C_prev):
        x_s,z_s,f_s,i_s = {},{},{},{}
        C_bar_s,C_s,o_s,h_s = {},{},{},{}
        v_s,y_s,v_1s,y_1s = {},{},{},{}


        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]
            z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t],v_1s[t],y_1s[t] = self.cell_forward(x_s[t],h_s[t-1],C_s[t-1])

        return (z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s)
    
    def BPTT(self,outs,Y):

        z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s = outs

        dW_f, dW_i,dW_c, dW_o,dW,dW_hid = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c),np.zeros_like(self.W_o),np.zeros_like(self.W),np.zeros_like(self.W_hid)

        dB_f, dB_i,dB_c,dB_o,dB,dB_hid  = np.zeros_like(self.B_f), np.zeros_like(self.B_i),np.zeros_like(self.B_c),np.zeros_like(self.B_o),np.zeros_like(self.B),np.zeros_like(self.B_hid)

        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0])   

        # w.r.t. softmax input
        ddense = np.copy(y_s[149])
        ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
        #ddense[np.argmax(Y,1)] -=1
        #ddense = y_s[149] - Y
        # Softmax classifier's :
        dW = np.dot(v_1s[149].T,ddense)
        dB = np.sum(ddense,axis = 0, keepdims = True)

        ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(v_1s[149])
        dW_hid = np.dot(h_s[149].T,ddense_hid)
        dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)


        # Backprop through time:
        for t in reversed(range(1,self.seq_len)):           
            
            # Just equating more meaningful names
            stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs = z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]
            C_prev = C_s[t-1]
            
            # w.r.t. softmax input
            #ddense = np.copy(probs)
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
            # Softmax classifier's :
            #dW += np.dot(hidden_state.T,ddense)
            #dB += np.sum(ddense,axis = 0, keepdims = True)

            # Output gate :
            dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
            do = dh * np.tanh(cell_state)
            do = do * dsigmoid(output_gate)
            dW_o += np.dot(stack.T,do)
            dB_o += np.sum(do,axis = 0, keepdims = True)

            # Cell state:
            dC = np.copy(dC_next)
            dC += dh * output_gate * activations.dtanh(cell_state)
            dC_bar = dC * input_gate
            dC_bar = dC_bar * dtanh(cell_bar) 
            dW_c += np.dot(stack.T,dC_bar)
            dB_c += np.sum(dC_bar,axis = 0, keepdims = True)
            
            # Input gate:
            di = dC * cell_bar
            di = dsigmoid(input_gate) * di
            dW_i += np.dot(stack.T,di)
            dB_i += np.sum(di,axis = 0,keepdims = True)

            # Forget gate:
            df = dC * C_prev
            df = df * dsigmoid(forget_gate) 
            dW_f += np.dot(stack.T,df)
            dB_f += np.sum(df,axis = 0, keepdims = True)

            dz = np.dot(df,self.W_f.T) + np.dot(di,self.W_i.T) + np.dot(dC_bar,self.W_c.T) + np.dot(do,self.W_o.T)

            dh_next = dz[:,-self.hidden_dim_1:]
            dC_next = forget_gate * dC
        
        # List of gradients :
        grads = [dW,dB,dW_hid,dB_hid,dW_o,dB_o,dW_c,dB_c,dW_i,dB_i,dW_f,dB_f]

        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1],C_s[self.seq_len -1 ],grads
    


    def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)           
            h_prev,C_prev = np.zeros((self.batch_size,self.hidden_dim_1)),np.zeros((self.batch_size,self.hidden_dim_1))
            for i in range(round(X.shape[0]/self.batch_size) - 1): 
               
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
               
                # Forward + BPTT + SGD:
                cache_train = self.forward(X_feed,h_prev,C_prev)
                h,c,grads = self.BPTT(cache_train,y_feed)

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
                # Cell state ---------> Previous cell state
                h_prev,C_prev = h,c

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
            _,__,___,____,_____,______,_______,________,probs_test,a,b = self.forward(X_val,test_prevs,test_prevs)
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
        return [self.W,self.B,self.W_hid,self.B_hid,self.W_o,self.B_o,self.W_c,self.B_c,self.W_i,self.B_i,self.W_f,self.B_f]


    def SGD(self,grads):
      """

      Stochastic gradient descent with momentum on mini-batches.
      """
      prevs = []
     
      for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
          delta = self.learning_rate * grad - self.mom_coeff * prev_update
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

        # Give zeros to hidden/cell states:
        pasts = np.zeros((X.shape[0],self.hidden_dim_1))
        _,__,___,____,_____,______,_______,_______,probs,a,b = self.forward(X,pasts,pasts)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}      
if __name__ == '__main__':
    multi_layer_lstm = Multi_Layer_LSTM(learning_rate=1e-3,batch_size=32,hidden_dim_1 = 128,hidden_dim_2=64,mom_coeff=0.0)
    multi_layer_lstm.fit(X_train,y_train,X_test,y_test,epochs=15,optimizer='Adam')