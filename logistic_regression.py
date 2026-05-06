import math 

class Logistic_regression:
    def __init__(self, lr = 0.5 , epochs = 500):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))   

    def predict_single(self,x):
        z = 0
        #z=w1​x1 ​+w2​x2 ​+⋯+ wd​xd ​+bias
        for j in range(len(self.w)):
            z += x[j] * self.w[j]
        z += self.b
        return self.sigmoid(z)
    
    def predict_proba(self,X):
        return [self.predict_single(x) for x in X] #x is a list of x1,x2 pairs

    def loss(self,y,y_preds):
        total = 0
        n = len(y) #number of samples 
        for i in range(n):
            p = y_preds[i]
            p = max(min(p, 1 - 1e-10), 1e-10) # epsilon value to avoid log(0)
            
        return total / n


#creating a dataset 
import random
random.seed(42)
X = []
y = []
for _ in range(100):
    x1 = random.random()
    x2 = random.random()
    X.append([x1,x2])
    
    # decision rule (ground truth)
    if 2*x1 + x2 > 1.2:
        y.append(1)
    else:
        y.append(0)
#print(X)    #shape should be (100,2)    
#print(y)    #shape should be (100,)   
    
