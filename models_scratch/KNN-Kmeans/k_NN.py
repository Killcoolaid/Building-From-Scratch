import numpy as np
from collections import Counter 

class KNN: 
    def __init__(self,dataset,target,k=5):
        self.k = k ; 
        self.dataset = dataset 
        self.target = target 
    def euclidian_distance(self,x): 
        return np.array([sum((data-x)**2) for data in self.dataset])
    def _predict(self,test_data): 
        if(len(test_data) != len(self.dataset[0])): 
            print("input data does not match with dataset")

        sorted_index = self.euclidian_distance(test_data).argsort()[:self.k]
        counter = Counter([self.target[sorted_index[i]] for i in sorted_index])
        most_common_element = counter.most_common(1)[0]
        return most_common_element
        
        