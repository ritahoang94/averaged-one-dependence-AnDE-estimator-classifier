#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:34:13 2019

@author: ritahoang
"""
from prequential import Prequential_learning
import numpy as np
import math

class Ensemble(object):
    def __init__(self, data , ensemble_no, classifier_list=[]): #Update vis
        self.data = data  
        self.var_list = classifier_list[0].var_list #Update vis
        self.classifier_list = classifier_list
        self.classifier_no = len(classifier_list)
        self.ensemble_no = ensemble_no
        self.counter = 0
        self.klass_dict = {key: idx for idx, key in enumerate(self.var_list[-1])} #Update vis
        self.klass_size =len(self.var_list[-1])
        self.acc_se  = 0 #Update vis
        self.SE_list = [] #square error list of last 100 data points #update vis
        self.iteration = []#Update vis
        
        self.y_predict_list = []
        self.rmse100_components = [[]]*self.classifier_no
        self.rmse100list =[] # Update vis 
        
        self.preq_list =[]
        for classifier in self.classifier_list:
            self.preq_list.append(Prequential_learning(self.data, classifier)) # update vis
        
    def miss_rate(self):
        wrong =0
        for i in range(len(self.y_predict_list)):
            if self.y_predict_list[i] != self.klass_dict[self.data[i][-1]]:
                wrong += 1 

        wrong_ratio = wrong/len(self.data) *100
        print("misclassification ration is (%): ",wrong_ratio)
    
    def square_error(self, predict_prob, actual_prob):
        return sum((predict_prob - actual_prob)**2)
    
    def acc_rmse(self):
        
        return math.sqrt(self.acc_se/(self.klass_size*len(self.data)))
    
    def rmse100(self):
        if len(self.SE_list) == 100:
            return math.sqrt(sum(self.SE_list)/(100*self.klass_size))
        else:
            return None
    
    def update(self, point):
        klass_index = self.klass_dict[point[-1]]
        actual_prob = np.zeros((self.klass_size),dtype=np.double)
        actual_prob[klass_index] = 1
        
        prob_lol =[] 
        if self.counter <99:
            for i in range(self.classifier_no):
                self.preq_list[i].update(point)
                prob_lol.append(self.preq_list[i].result[1])
            prob_lol = sum(prob_lol)
            predict_prob = prob_lol/self.classifier_no
        else:
            for i in range(self.classifier_no):
                self.preq_list[i].update(point)
                self.rmse100_components[i] = self.preq_list[i].rmse100()
            
            ensemble_id_list = sorted(range(len(self.rmse100_components)), key=lambda i: self.rmse100_components[i])[:self.ensemble_no]

            for i in ensemble_id_list:
                prob_lol.append(self.preq_list[i].result[1])
            prob_lol = sum(prob_lol)
            predict_prob = prob_lol/self.ensemble_no

        predict_response =  np.argmax(predict_prob)
        
        self.y_predict_list.append(predict_response)
        
        se = self.square_error(predict_prob, actual_prob)

        self.acc_se += se
        if len(self.SE_list) <100:               
            self.SE_list.append(se)
        else:
            self.SE_list.pop(0)
            self.SE_list.append(se)
        
        if len(self.SE_list) == 100:
            self.iteration.append(self.counter)
            self.rmse100list.append(self.rmse100())

        self.counter +=1
       
        return predict_response, predict_prob

    def main (self):
        for i in self.data:
            self.update(i)

            
#        print(self.rmse100list)
            

        
