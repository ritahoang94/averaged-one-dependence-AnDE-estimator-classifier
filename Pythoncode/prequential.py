#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:21:21 2019

@author: ritahoang
"""
import math
import numpy as np

class Prequential_learning(object):
    
    def __init__(self, data , classifier, debug = 1, vis = 1): 
        self.debug = debug
        self.data = data
        self.var_list = classifier.var_list 
        self.klass_size = len(self.var_list[-1])
        self.klass_dict = {key: idx for idx, key in enumerate(self.var_list[-1])}  
        self.counter2 = 0 
        self.vis = vis 
        


        self.acc_se = 0
        self.y_predict_list = []
        self.SE_list = [] #square error list of last 100 data points
        self.classifier = classifier # Update vis
        self.name = self.classifier.name # Update Vis
        self.lamda = self.classifier.lamda # Update Vis
        self.iteration =[] # Update vis
        self.rmse100list =[] # Update vis    

    def update_predict_list(self, reponse_index):
        self.y_predict_list.append(reponse_index)
        
    def square_error(self, predict_prob, actual_prob):
        return sum((predict_prob - actual_prob)**2)
    
    def acc_rmse(self):
        
        return math.sqrt(self.acc_se/(self.klass_size*len(self.data)))
    
    def miss_rate(self):
        wrong =0
        for i in range(len(self.y_predict_list)):
            if self.y_predict_list[i] != self.klass_dict[self.data[i][-1]]:
                wrong += 1 

        wrong_ratio = wrong/len(self.data) *100
        print("misclassification ration is (%): ",wrong_ratio)
        
    def rmse100(self):
        if len(self.SE_list) == 100:
            return math.sqrt(sum(self.SE_list)/(100*self.klass_size))
        else:
            return None
        
    
    def update(self, point):
        klass_index = self.klass_dict[point[-1]] 
        
        actual_prob = np.zeros((self.klass_size),dtype=np.double)
        actual_prob[klass_index] = 1
        self.counter2 = self.counter2 +1 
        
        self.result = self.classifier.klassify(point)
     

                
        # debug mode:
        
        if self.debug == 1:
            self.update_predict_list(self.result[0])
            predict_prob = self.result[1]
            square_error = self.square_error(predict_prob, actual_prob)
            
            self.acc_se += square_error
            if len(self.SE_list) <100:               
                self.SE_list.append(square_error)
            else:
                self.SE_list.pop(0)
                self.SE_list.append(square_error)
                
        '''
        VIS MODE
        '''
        if self.vis == 1:
            if self.counter2 >= 100:
                self.iteration.append(self.counter2)
                self.rmse100list.append(self.rmse100())

        self.classifier.update(point)

    
    

    def main(self):
        for point in self.data:
            self.update(point)
        if self.debug == 1:
            print("THIS IS THE DEBUG MODE")
        else:
            print("THIS IS NOT THE DEBUG MODE. PLEASE SET DEBUG = 1 TO CHECK RESULT.")