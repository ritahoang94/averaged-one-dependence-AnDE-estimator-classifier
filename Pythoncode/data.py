#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:20:59 2019

@author: ritahoang
"""

import arff
import pandas as pd
# DATA POKER
#data = arff.load(open('poker-lsn.arff'))
#
#data_attribute = data['attributes']
#column_name = []
#for i in data_attribute:
#    column_name.append(i[0])
#
#
#df2 = pd.DataFrame(data['data'])
#df2.columns = column_name
#
#df4 = df2.loc[:,['s1','s2','s3','s4','s5','class']]
#dt = df4.values
#var_dict = dict(data_attribute)
#var_list = list(var_dict[k] for k in ('s1','s2','s3','s4','s5','class') if k in var_dict)
#klass_list = var_dict['class']

## DATA AIRLINES
#data = arff.load(open('airlines.arff'))
#
#data_attribute = data['attributes']
#column_name = []
#for i in data_attribute:
#    column_name.append(i[0])
#
#
#df2 = pd.DataFrame(data['data'])
#df2.columns = column_name
#
#df4 = df2.loc[:,['Delay','DayOfWeek','Airline']]
#dt = df4.values
#var_dict = dict(data_attribute)
#var_list = list(var_dict[k] for k in ('Delay','DayOfWeek','Airline') if k in var_dict)
#klass_list = var_dict['Airline']
#print("use new dataset")

# DATA SIMULATED_POKER
data = arff.load(open('abrupt-drift-10000.arff'))

data_attribute = data['attributes']
column_name = []
for i in data_attribute:
    column_name.append(i[0])


df2 = pd.DataFrame(data['data'])
df2.columns = column_name

df4 = df2.loc[:,['x1','x2','x3','x4','x5','class']]
dt = df4.values
var_dict = dict(data_attribute)
var_list = list(var_dict[k] for k in ('x1','x2','x3','x4','x5','class') if k in var_dict)
klass_list = var_dict['class']