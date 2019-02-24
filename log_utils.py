"""
@author: orrivlin
"""

class mean_val:
    def __init__(self):
        self.k = 0
        self.val = 0
        self.mean = 0
        
    def append(self,x):
        self.k += 1
        self.val += x
        self.mean = self.val/self.k
        
    def get(self):
        return self.mean
        
    
class logger:
    def __init__(self):
        self.log = dict()
        
    def add_log(self,name):
        self.log[name] = []
        
    def add_item(self,name,x):
        self.log[name].append(x)
        
    def get_log(self,name):
        return self.log[name]
    
    def get_keys(self):
        return self.log.keys()
    
    def get_current(self,name):
        return self.log[name][-1]
        
