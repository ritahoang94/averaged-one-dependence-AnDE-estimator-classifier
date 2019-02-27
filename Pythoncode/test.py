from prequential import Prequential_learning
from data import dt, var_list
import matplotlib.pyplot as plt
from A1DE2 import A1DE
from NB import NBClassifier

lamda_list = [0.0, 0.05, 0.1]

class vis(object):
    def __init__(self, lamda_list, data, var_list,startpoint, endpoint, NB = 0, A1DE = 0, A2DE =0):
        self.lamda_list = lamda_list
        self.data = data
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.NB = NB
        self.A1DE = A1DE
        self.A2DE = A2DE
        self.var_list = var_list
        self.classifierlist = []
        
    def build_classifiers(self): 
        for lamda in lamda_list:
            if self.NB == 1:
                preqNB = Prequential_learning(self.data[self.startpoint:self.endpoint],NBClassifier(self.var_list,lamda),debug=1, vis=1)
                preqNB.main()
                self.classifierlist.append(preqNB)
            if self.A1DE == 1:
                preqA1DE = Prequential_learning(self.data[self.startpoint:self.endpoint], A1DE(self.var_list,lamda),debug=1, vis=1)
                preqA1DE.main()
                self.classifierlist.append(preqA1DE)
    
    def visualise(self): 
        f = plt.figure()
        for i in range(len(self.classifierlist)):
            plt.plot(self.classifierlist[1].iteration, self.classifierlist[i].rmse100list,label= self.classifierlist[i].classifier.name  \
                     + str(self.classifierlist[i].classifier.lamda) , linewidth=0.3)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .200), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.show()    
        f.savefig("airline-test.pdf", bbox_inches='tight')

Visample = vis(lamda_list,dt, var_list, startpoint =0, endpoint = 10000, NB = 1, A1DE = 1)
Visample.build_classifiers()
Visample.visualise()