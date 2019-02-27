#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:43:49 2019

@author: ritahoang
"""

from data import dt, var_list
import matplotlib.pyplot as plt
from A1DE2 import A1DE
from NB import NBClassifier
from ensemble import Ensemble
lamda_list = [0.0, 0.05, 0.1]

        
def visualise_result(lamda_list, data, var_list,startpoint, endpoint, candidate_no =1, NBc = 0, A1DEc = 0, A2DE =0): 
    classifierlist = []
    for lamda in lamda_list:
        if NBc == 1:
            preqNB = NBClassifier(var_list,lamda)
#                preqNB.main()
          
            classifierlist.append(preqNB)
            
        if A1DEc == 1:
            preqA1DE = A1DE(var_list,lamda)
#                preqA1DE.main()
            
            classifierlist.append(preqA1DE)
           
    en = Ensemble(data[startpoint:endpoint], candidate_no, classifierlist)
    en.main()

    f = plt.figure()
    
    for i in range(len(classifierlist)):
        plt.plot(en.preq_list[1].iteration,en.preq_list[i].rmse100list,label= en.preq_list[i].name +" " + str(en.preq_list[i].lamda), linewidth=0.15)
#        plt.plot(en.preq_list[1].iteration,en.preq_list[i].rmse100list,label= en.preq_list[i].name +" " + str(en.preq_list[i].lamda), linewidth=0.1, color ='y')
        
    plt.plot(en.preq_list[1].iteration,en.rmse100list,label="ensemble" , linewidth=0.3, color ='black')
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .200), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()    
    f.savefig("airline-test.pdf", bbox_inches='tight')
    print(en.acc_rmse())
    for i in en.preq_list:
        print(i.name, i.lamda)
        i.miss_rate()
    en.miss_rate()

visualise_result(lamda_list,dt, var_list, startpoint =0, endpoint = 20000, candidate_no=3, NBc = 1, A1DEc = 1)





