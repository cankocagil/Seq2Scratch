import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd 

class Metrics: 
    """
    Necessary metrics to evaluate the model.
        Functions(labels,preds):
        * confusion_matrix
        *  accuracy_score     
    """ 
    def confusion_matrix(self,labels,preds):
        """
        Takes desireds/labels and softmax predictions,
        return a confusion matrix.
        """
        label = pd.Series(labels,name='Actual')
        pred = pd.Series(preds,name='Predicted')
        return pd.crosstab(label,pred)

    def accuracy_score(self,labels,preds): 
        """
        Takes desireds/labels and softmax predictions,
        return a accuracy_score.
        """       
        count = 0
        size = labels.shape[0]
        for i in range(size):
            if preds[i] == labels[i]:
                count +=1
        return  100 * (count/size)

    def accuracy(self,labels,preds):
        """
        Takes desireds/labels and softmax predictions,
        return a accuracy.
        """
        return 100 * (labels == preds).mean()    
