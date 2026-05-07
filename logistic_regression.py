import math 

class Logistic_regression:
    def __init__(self, lr = 0.5 , epochs = 500):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def sigmoid(self, z):
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
            total += y[i]*math.log(p) + (1 - y[i])*math.log(1-p)
        return -total / n

    def fit(self,X,y):
        n = len(X)
        d = len(X[0]) # d is number of features, n is number of samples
        self.w = [0] * d # initialize weights with small random values
        self.b = 0

        #gradient descent loop
        for epoch in range(self.epochs):
            y_pred = self.predict_proba(X)

            #initialising dw and db
            dw = [0] * d
            db = 0 

            for i in range(n):
                error = y_pred[i] - y[i]

                #updating weights and bias
                for j in range(d):
                    dw[j] += error * X[i][j]
                db += error
            
            for j in range(d):
                dw[j] /= n

            db/=n 

            for j in range(d):
                self.w[j] -= self.lr * dw[j] #updated the weights using the formula w = w - lr*dw
                self.b -= self.lr * db #updated the bias using the formula b = b - lr*db

            if epoch % 100 == 0:
                loss = self.loss(y,y_pred)
                print(f"epoch: {epoch}, loss: {loss}") 

    def predict(self,X):
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]            
                
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
    
#training a model
log_reg = Logistic_regression(lr=0.1 , epochs = 1000)
log_reg.fit(X,y)

#making predictions 
predictions = log_reg.predict(X)
print(predictions)   

#checking accuracy
correct = 0

for i in range(len(y)):
    if predictions[i] == y[i]:
        correct += 1

print("Accuracy:", correct / len(y))

#visualising the decision boundary
import matplotlib.pyplot as plt

# -----------------------------
# Separate points by class
# -----------------------------
class0_x1 = []
class0_x2 = []

class1_x1 = []
class1_x2 = []

for i in range(len(X)):
    if y[i] == 0:
        class0_x1.append(X[i][0])
        class0_x2.append(X[i][1])
    else:
        class1_x1.append(X[i][0])
        class1_x2.append(X[i][1])

# -----------------------------
# Plot data points
# -----------------------------
plt.scatter(class0_x1, class0_x2, label="Class 0")
plt.scatter(class1_x1, class1_x2, label="Class 1")

# -----------------------------
# Decision boundary
# w1*x1 + w2*x2 + b = 0
# => x2 = -(w1*x1 + b)/w2
# -----------------------------
x1_vals = [0, 1]

x2_vals = []

for x1 in x1_vals:
    x2 = -(log_reg.w[0] * x1 + log_reg.b) / log_reg.w[1]
    x2_vals.append(x2)

# plot boundary line
plt.plot(x1_vals, x2_vals)

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Logistic Regression Decision Boundary")

plt.show()