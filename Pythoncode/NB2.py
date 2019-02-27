#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:29:09 2019

@author: ritahoang
"""


import math
import numpy as np


#  In the training dataset, the dependent variable must be put in the last column
class NBClassifier2(object):
 
    # Create constructor
    def __init__(self, var_list , lamda=0):
        self.decay_rate = math.exp(-lamda)
        self.counter =0    # count number of data point
        self.dict_list =[]
        for i in var_list[:-1]:
            self.dict_list.append({key: idx for idx, key in enumerate(i)})
        self.var_size_ls = list(len(i) for i in var_list)
        
        """
        1. Count class (klass_count)
        - klass_dict: class value : true index
        - klass_count: array contains the count of each klass
        """
        self.klass_dict = {key: idx for idx, key in enumerate(var_list[-1])} ## lookup array
        self.klass_size = self.var_size_ls[-1]
        self.klass_count = np.zeros((self.klass_size),dtype=np.float) ## count array


        """
        2. x_v_y table (feature_value_klass)
        """
        self.feature_number = len(var_list) - 1
        self.x_v_y = np.zeros((self.feature_number*self.klass_size+self.klass_size,max(self.var_size_ls[:-1])),dtype=np.float)    
            
        """
        Output
        """
        self.predict_prob = np.zeros((self.klass_size),dtype=np.float)  # array contains probability  
        
        
    def update(self, point):        
        """
        Update counter
        """
        features = point[:-1]
        klass_index = self.klass_dict[point[-1]]
        
        self.counter = 1 + self.counter*self.decay_rate                      # Undate counter
        self.laplace_counter = self.counter +1
        
        """
        update klass count
        """
        self.klass_count = self.klass_count*self.decay_rate                 # decay first
        self.klass_count[klass_index] = self.klass_count[klass_index] +1    # Add new
        
        """
        update x_v_y
        """  
        self.x_v_y = self.x_v_y*self.decay_rate                              # Decay all matrix
        
        for i in range(self.feature_number):                                 # iterate over all predictors
            value_index = self.dict_list[i][features[i]]                         # find value index of each predictor
            self.x_v_y[i*self.klass_size + klass_index][value_index] = self.x_v_y[i*self.klass_size + klass_index][value_index] +1 # Update
                                   
    """
    Using Laplace smmothing to calculate prior and conditional probabilities
    """
    def log_prior(self, klass_index):
        return math.log((self.klass_count[klass_index] + 1/self.klass_size)/self.laplace_counter)  
    
    def log_conditional(self, klass_index, feature): # features here is an array
        sum_condition = 0
        
        '''
        How to remove the loop
        '''
        for i in range(self.feature_number): 
            value_index = self.dict_list[i][feature[i]]
            sum_condition += math.log((self.x_v_y[i*self.klass_size + klass_index][value_index] + 1/self.var_size_ls[i])\
                                /(self.klass_count[klass_index] + 1))
        return sum_condition
        
    
    #Evidence using logsumexp
    def log_evidence(self, features):
        join_array = np.zeros((self.klass_size),dtype=np.float)
        for klass_index in range(self.klass_size):
            join_array[klass_index] = self.log_conditional(klass_index, features) + self.log_prior(klass_index)
            
        A = max(join_array)
        return A + math.log(sum(np.exp(join_array-A)))
    
    '''
    klassify will return index of true response and predict probabilities
    '''
    
    def klassify(self, point):
        features = point[:-1]
        response = point[-1]
        if self.counter ==0:
            predict_response =  self.klass_dict[response]
            for klass_id in range(self.klass_size):
                self.predict_prob[self.klass_dict[response]] = 1/self.klass_size
            
        else:
            for klass_id in range(self.klass_size): # remove log evidence, keep doing in log space
                self.predict_prob[klass_id] = self.log_conditional(klass_id, features) + self.log_prior(klass_id)
              
            '''
            Normalise probability
            '''
            predict_response =  np.argmax(self.predict_prob)
            self.predict_prob = np.exp(self.predict_prob -self.log_evidence(features))
            
        return predict_response, self.predict_prob

