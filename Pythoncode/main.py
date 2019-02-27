#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:08:38 2019

@author: ritahoang
"""

from prequential import Prequential_learning
from data import dt, var_list
import cProfile as profile
import pstats

from A1DE2 import A1DE
from NB import NBClassifier

###
#### Normal test
Preq = Prequential_learning(dt, NBClassifier(var_list,0),debug=1, vis=1)
profile.run('Preq.main()','restats')
p = pstats.Stats('restats')
p.sort_stats('cumtime')
p.print_stats()
Preq.miss_rate()
print(Preq.acc_rmse())
print(Preq.rmse100()) # update 
Preq.y_predict_list[:10]


# MetaClassifier
#classifier_list = [NBClassifier(var_list,0.0001), NBClassifier(var_list,0.0002)]
#from metaclassifier import MetaClassifier
#Preq2 = Prequential_learning(dt[:10000],MetaClassifier(classifier_list),debug=1)
#profile.run('Preq2.main()','restats')
#p2 = pstats.Stats('restats')
#p2.sort_stats('cumtime')
#p2.print_stats()
#Preq2.miss_rate()

#######################################################
# Ensemble
#from ensemble import Ensemble
#classifier_list = [NBClassifier(var_list,0.0), NBClassifier(var_list,0),  A1DE(var_list,0.0)]
#en = Ensemble(dt, 2, classifier_list)
#en.main()
#en.miss_rate()
#print(en.acc_rmse())
#print(en.rmse100()) # update 
#print(en.y_predict_list[:10])

########################################################
##A1DE
#
#Preq = Prequential_learning(dt, A1DE(var_list,0),debug=1)
#profile.run('Preq.main()','restats')
#p = pstats.Stats('restats')
#p.print_stats()
#Preq.miss_rate()
#print(Preq.acc_rmse())
#print(Preq.rmse100()) # update 
#Preq.y_predict_list[:10]



