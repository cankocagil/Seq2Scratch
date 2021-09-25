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


class RNN(object):
    """
    Recurrent Neural Network for classifying human activity.
    RNN encapsulates all necessary logic for training the network.

    """
    def __init__(self,input_dim = 3,hidden_dim = 128, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):

        """

        Initialization of weights/biases and other configurable parameters.
        
        """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim_inp2hid = Xavier(self.input_dim,self.hidden_dim)
        self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim))
        self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim))

        lim_hid2hid = Xavier(self.hidden_dim,self.hidden_dim)
        self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim,self.hidden_dim))

        lim_hid2out = Xavier(self.hidden_dim,self.output_class)
        self.W2 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim,self.output_class))
        self.B2 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
        
        # Storing previous momentum updates :
        self.prev_updates = {'W1'       : 0,
                             'B1'       : 0,
                             'W1_rec'   : 0,
                             'W2'       : 0,
                             'B2'       : 0}


    def forward(self,X) -> tuple:
        """ Forward propagation of the RNN through time.
        

        Inputs:
        --- X is the bacth.
        --- h_prev_state is the previous state of the hidden layer.
        
        Returns:
        --- (X_state,hidden_state,probs) as a tuple.       
        ------ 1) X_state is the input across all time steps
        ------ 2) hidden_state is the hidden stages across time
        ------ 3) probs is the probabilities of each outputs, i.e. outputs of softmax
        """ 
        X_state = dict()
        hidden_state = dict()
        output_state = dict()
        probs = dict()

        
        self.h_prev_state = np.zeros((1,self.hidden_dim))
        hidden_state[-1] = np.copy(self.h_prev_state)

        # Loop over time T = 150 :
        for t in range(self.seq_len):

            # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
            X_state[t] = X[:,t]

            # Recurrent hidden layer :
            hidden_state[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state[t-1],self.W1_rec) + self.B1)
            output_state[t] = np.dot(hidden_state[t],self.W2) + self.B2

            # Per class probabilites :
            probs[t] = activations.softmax(output_state[t])

        return (X_state,hidden_state,probs)
        

    def BPTT(self,cache,Y):
        """

        Back propagation through time algorihm.
        Inputs:
        -- Cache = (X_state,hidden_state,probs)
        -- Y = desired output

        Returns:
        -- Gradients w.r.t. all configurable elements
        """

        X_state,hidden_state,probs = cache

        # backward pass: compute gradients going backwards
        dW1, dW1_rec, dW2 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2)

        dB1, dB2 = np.zeros_like(self.B1), np.zeros_like(self.B2)

        dhnext = np.zeros_like(hidden_state[0])

        dy = np.copy(probs[149])      
        dy[np.arange(len(Y)), np.argmax(Y,1)] -= 1
        
        dB2 += np.sum(dy,axis = 0, keepdims = True)
        dW2 += np.dot(hidden_state[149].T,dy)

        for t in reversed(range(1,self.seq_len)):

            
        
            dh = np.dot(dy,self.W2.T) + dhnext
        
            dhrec = (1 - (hidden_state[t] * hidden_state[t])) * dh

            dB1 += np.sum(dhrec,axis = 0, keepdims = True)
            
            dW1 += np.dot(X_state[t].T,dhrec)
            
            dW1_rec += np.dot(hidden_state[t-1].T,dhrec)

            dhnext = np.dot(dhrec,self.W1_rec.T)

               
        for grad in [dW1,dB1,dW1_rec,dW2,dB2]:
            np.clip(grad, -10, 10, out = grad)


        return [dW1,dB1,dW1_rec,dW2,dB2]    
        
    def earlyStopping(self,ce_train,ce_val,ce_threshold,acc_train,acc_val,acc_threshold):
        if ce_train - ce_val < ce_threshold or acc_train - acc_val > acc_threshold:
            return True
        else:
            return False
   

    def CategoricalCrossEntropy(self,labels,preds):
        """
        Computes cross entropy between labels and model's predictions
        """
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N

    def step(self,grads,momentum = True):
        """
        SGD on mini batches
        """

     
        #for config_param,grad in zip([self.W1,self.B1,self.W1_rec,self.W2,self.B2],grads):
            #config_param -= self.learning_rate * grad

        if momentum:
            
            delta_W1 = -self.learning_rate * grads[0] +  self.mom_coeff * self.prev_updates['W1']
            delta_B1 = -self.learning_rate * grads[1] +  self.mom_coeff * self.prev_updates['B1']  
            delta_W1_rec = -self.learning_rate * grads[2] +  self.mom_coeff * self.prev_updates['W1_rec']
            delta_W2 = -self.learning_rate * grads[3] +  self.mom_coeff * self.prev_updates['W2']              
            delta_B2 = -self.learning_rate * grads[4] +  self.mom_coeff * self.prev_updates['B2']
            
               
            self.W1 += delta_W1
            self.W1_rec += delta_W1_rec
            self.W2 += delta_W2
            self.B1 += delta_B1
            self.B2 += delta_B2     

            
            self.prev_updates['W1'] = delta_W1
            self.prev_updates['W1_rec'] = delta_W1_rec
            self.prev_updates['W2'] = delta_W2
            self.prev_updates['B1'] = delta_B1
            self.prev_updates['B2'] = delta_B2

            self.learning_rate *= 0.9999

    def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, earlystopping = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)           
            
            for i in range(round(X.shape[0]/self.batch_size)): 

                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size
                index = perm[batch_start:batch_finish]
                
                X_feed = X[index]    
                y_feed = Y[index]
                
                cache_train = self.forward(X_feed)                                                          
                grads = self.BPTT(cache_train,y_feed)                
                self.step(grads)
      

            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[2][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            _,__,probs_test = self.forward(X_val)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
            predictions_val = np.argmax(probs_test[149],1)
            acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

            if earlystopping:                
                if self.earlyStopping(ce_train = cross_loss_train,ce_val = cross_loss_val,ce_threshold = 3.0,acc_train = acc_train,acc_val = acc_val,acc_threshold = 15): 
                    break
            


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

    def predict(self,X):
        _,__,probs = self.forward(X)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}
 
 if __name__ == '__main__':

    input_dim = 3
    activations = Activations()
    metrics = Metrics()
    model = RNN(input_dim = input_dim,learning_rate = 1e-4, mom_coeff = 0.0, hidden_dim = 128)

    model.fit(X_train,y_train,X_test,y_test,epochs = 35)

    history = model.history()

    plt.figure()
    plt.plot(history['TestLoss'],'-o')
    plt.plot(history['TrainLoss'],'-o')
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.title('Categorical Cross Entropy over epochs')
    plt.legend(['Test Loss','Train Loss'])
    plt.show()

    plt.figure()
    plt.plot(history['TestAcc'],'-o')
    plt.plot(history['TrainAcc'],'-o')
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.title('Accuracy over epochs')
    plt.legend(['Test Acc','Train Acc'])
    plt.show()

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    confusion_mat_train = metrics.confusion_matrix(np.argmax(y_train,1),train_preds)
    confusion_mat_test = metrics.confusion_matrix(np.argmax(y_test,1),test_preds)
