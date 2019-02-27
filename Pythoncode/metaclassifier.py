#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:25:43 2019

@author: ritahoang
"""
import numpy as np

class MetaClassifier(object):
    
    def __init__(self, classifier_list):
         self.classifier_list = classifier_list
         self.meta_size = len(classifier_list)
    
    def update(self, point):
        for classifier in self.classifier_list:
            classifier.update(point)
    
    def klassify(self, point):
        prob_list = []
        for classifier in self.classifier_list:
            prob_list.append(classifier.klassify(point)[1])
            
        prob_list = sum(prob_list)
        predict_prob = prob_list/self.meta_size
        predict_response =  np.argmax(predict_prob)
        
        return predict_response, predict_prob
    