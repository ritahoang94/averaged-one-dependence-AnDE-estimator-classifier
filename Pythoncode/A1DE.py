import math
import numpy as np

class A1DE(object):
    def __init__(self, var_list , lamda=0):
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
        
        '''
        Count y,x
        '''
        

#        self.klass_count = np.zeros((self.klass_size),dtype=np.float) ## count array
       
        # xy count
        self.x_v_y = np.zeros((self.feature_number,max(self.var_size_ls[:-1]), self.klass_size),dtype=np.float) 
        
        #xxy count
        self.x2v2_x1v1_y = np.zeros((self.feature_number,max(self.var_size_ls[:-1]), \
                                     self.feature_number,max(self.var_size_ls[:-1]), self.klass_size),dtype=np.float) 
        
        self.predict_prob = np.zeros((self.klass_size),dtype=np.float)
    
        """
    Using Laplace smmothing to calculate prior and conditional probabilities
    """
    def log_prior(self, k_index, mf_index,  mf_value):
        size = self.klass_size*len(self.var_list[mf_index]) #size y,x
        return math.log((self.x_v_y[mf_index][mf_value][k_index]+ 1/size)/self.laplace_counter)
    
        
    def log_conditional(self, k_index, mf_index,  mf_value, features): # features here is an array
        sum_condition = 0
        
        '''
        How to remove the loop
        '''
        for i in range(self.feature_number): 
            if i != mf_index:
                value_index = self.dict_list[i][features[i]]
                sum_condition += math.log((self.x2v2_x1v1_y[i][value_index][mf_index][mf_value][k_index] \
                                    + 1/self.var_size_ls[i])\
                                    /(self.x_v_y[mf_index][mf_value][k_index] + 1))
        return sum_condition

    def numerator(self, k_index, features):
        numerator =0
        for i in range(self.feature_number): 
            value_index = self.dict_list[i][features[i]]
            numerator+= \
            np.exp(self.log_prior(k_index, i, value_index) + self.log_conditional(k_index, i, value_index, features))
        return numerator
    
        #Evidence using logsumexp
    def denumerator(self, features):
        denumerator =0
        for k_index in range(self.klass_size):
            denumerator += self.numerator(k_index, features)
        
        return denumerator

    def update(self, point):        
        """
        Update counter
        """
        features = point[:-1]
        k_index = self.klass_dict[point[-1]]
        
        self.counter = 1 + self.counter*self.decay_rate                      # Undate counter
        self.laplace_counter = self.counter +1
        
        """
        update  count
        """
        
        self.x_v_y = self.x_v_y*self.decay_rate  
        self.x2v2_x1v1_y = self.x2v2_x1v1_y * self.decay_rate
        
        for i in range(self.feature_number):                                 # iterate over all predictors
            value_index = self.dict_list[i][features[i]]                         # find value index of each predictor
            self.x_v_y[i][value_index][k_index] = self.x_v_y[i][value_index][k_index] +1 # Update
            
            for j in range(self.feature_number):

                if j != i:
                    value_index2 = self.dict_list[j][features[j]]
                    self.x2v2_x1v1_y[j][value_index2][i][value_index][k_index] = self.x2v2_x1v1_y[j][value_index2][i][value_index][k_index]+ 1.0
                    
    
    def klassify(self, point):
        features = point[:-1]
        
        if self.counter ==0:
            for k_index in range(self.klass_size):
                self.predict_prob[k_index] = 1/self.klass_size
            predict_response =  np.argmax(self.predict_prob)
          
        else:
            for k_index in range(self.klass_size): # remove log evidence, keep doing in log space
                self.predict_prob[k_index] = self.numerator(k_index, features)/self.denumerator(features)
              
            '''
            Normalise probability
            '''
            predict_response =  np.argmax(self.predict_prob)
            
        return predict_response, self.predict_prob

