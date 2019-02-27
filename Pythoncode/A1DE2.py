import math
import numpy as np

class A1DE(object):
    def __init__(self, var_list , lamda=0, name = 'A1DE'):
        self.name = name 
        self.lamda = lamda 
        self.var_list = var_list
        self.decay_rate = math.exp(-lamda)
        self.counter =0    # count number of data point
        self.laplace_counter = self.counter + 1
        self.dict_list =[]
        for i in var_list[:-1]:
            self.dict_list.append({key: idx for idx, key in enumerate(i)})
        
        self.var_size_ls = list(len(i) for i in var_list)
        self.feature_number = len(var_list) - 1
        
        '''
        Look up 
        '''
        
        self.klass_dict = {key: idx for idx, key in enumerate(var_list[-1])} ## lookup array
        self.klass_size = self.var_size_ls[-1]
        self.max_len_x = max(self.var_size_ls[:-1])
        
        '''
        COUNT
        '''
        # Count y

        self.klass_count = np.zeros((self.klass_size),dtype=np.double) ## count array
       
        # xy count
        self.xy_v = np.zeros((self.feature_number*self.klass_size+self.klass_size, self.max_len_x),dtype=np.double) 
        
        #x1x2y_v count
        self.x1x2y_v = np.zeros((self.feature_number+ self.feature_number**2 + (self.feature_number**2)*self.klass_size,self.max_len_x, self.max_len_x),dtype=np.double) 
                
        self.predict_prob = np.zeros((self.klass_size),dtype=np.double)
    
        """
    Using Laplace smmothing to calculate prior and conditional probabilities
    """
    def conver3to1(self,x1id,x2id,yid):
        return x1id + x2id*self.feature_number + yid*(self.feature_number**2)
    
    def conver2to1(self,x1id,yid):
        return x1id*self.klass_size + yid
    
    def log_prior_y(self, k_index):
        return math.log((self.klass_count[k_index] + 1/self.klass_size)/self.laplace_counter)  
    
    def log_prior_xy(self, k_index, mf_index,  mf_value): # x|y
        converted_xy = self.conver2to1(mf_index,k_index)
        
        return math.log((self.xy_v[converted_xy][mf_value] + 1/self.var_size_ls[mf_index])/(self.klass_count[k_index] + 1))

    def log_prior(self, k_index, mf_index,  mf_value):
        return self.log_prior_y(k_index) + self.log_prior_xy(k_index, mf_index,  mf_value)

        
    def log_conditional(self, k_index, mf_index,  mf_value, features): # features here is an array
        sum_condition = 0
        
        '''
        How to remove the loop
        '''
        for i in range(self.feature_number): 
            if i != mf_index:
                value_index = self.dict_list[i][features[i]]
                converted_xy = self.conver2to1(mf_index,k_index)
                converted_x1x2y = self.conver3to1(mf_index,i, k_index)
                
                sum_condition += math.log((self.x1x2y_v[converted_x1x2y][mf_value][value_index]  + 1/self.var_size_ls[i])/(self.xy_v[converted_xy][mf_value] + 1))
                
        return sum_condition
    
    def logsumexp(self, array):
        A = max(array)
        
        return A + math.log(sum(np.exp(array-A)))

    def numerator(self, k_index, features):
        
        join_array = np.zeros((self.feature_number),dtype=np.double)
        
        for i in range(self.feature_number): 
            value_index = self.dict_list[i][features[i]]
            join_array[i] = self.log_prior(k_index, i, value_index) + self.log_conditional(k_index, i, value_index, features)
        return self.logsumexp(join_array)
    
        #Evidence using logsumexp
    def denominator(self, features):
        klass_array = np.zeros((self.klass_size),dtype=np.double)
        for k_index in range(self.klass_size):
            klass_array[k_index] = self.numerator(k_index, features)
        
        return self.logsumexp(klass_array)

    def update(self, point):        
        """
        Update counter
        """
        features = point[:-1]
        k_index = self.klass_dict[point[-1]]
        
        self.counter = 1 + self.counter*self.decay_rate                     
        self.laplace_counter = self.counter +1
        
        """
        update  count
        """
        
        self.xy_v = self.xy_v*self.decay_rate  
        self.x1x2y_v = self.x1x2y_v * self.decay_rate
        self.klass_count = self.klass_count* self.decay_rate  
        self.klass_count[k_index] = self.klass_count[k_index] +1  
        
        for i in range(self.feature_number):                                 # iterate over all predictors
            value_index = self.dict_list[i][features[i]]                         # find value index of each predictor
            converted_xy = self.conver2to1(i,k_index)
            self.xy_v[converted_xy][value_index] = self.xy_v[converted_xy][value_index]  +1 # Update
            
            for j in range(self.feature_number):

                if j != i:
                    value_index2 = self.dict_list[j][features[j]]
                    converted_x1x2y = self.conver3to1(i,j,k_index)
                    
                    self.x1x2y_v[converted_x1x2y][value_index][value_index2] = self.x1x2y_v[converted_x1x2y][value_index][value_index2] + 1.0
                    
    
    def klassify(self, point):
        features = point[:-1]
        if self.counter ==0:
            for klass_id in range(self.klass_size):
                self.predict_prob[klass_id] = 1/self.klass_size  # THERE IS AN UPDATE HERE
            predict_response =  np.argmax(self.predict_prob)
           
        else:
            for k_index in range(self.klass_size): # remove log evidence, keep doing in log space
                self.predict_prob[k_index] = self.numerator(k_index, features) 
                
            '''
            Normalise probability
            '''
            predict_response =  np.argmax(self.predict_prob)
            
            self.predict_prob = np.exp(self.predict_prob-self.denominator(features))

        return predict_response, self.predict_prob

