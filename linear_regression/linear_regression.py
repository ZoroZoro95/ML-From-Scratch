import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create synthetic dataset
# -----------------------------
np.random.seed(42)

X = np.linspace(0, 10, 100)              # 100 points from 0 to 10
print(X)
y = 3 * X + 5 + np.random.randn(100) * 2 # y = 3x + 5 + noise

# -----------------------------
# 2. Loss Function (MSE)
# -----------------------------
def loss_function(m, b, X, y):
    y_pred = m * X + b
    return np.mean((y_pred - y) ** 2)

# -----------------------------
# 3. Gradient Descent
# -----------------------------
def gradient_descent(m, b, X, y, learning_rate):
    n = len(X)

    y_pred = m * X + b
    error = y_pred - y

    # gradients
    Dm = (2/n) * np.sum(X * error)
    Db = (2/n) * np.sum(error)

    # update
    m = m - learning_rate * Dm
    b = b - learning_rate * Db

    return m, b

# -----------------------------
# 4. Training Loop
# -----------------------------
m = 0
b = 0
learning_rate = 0.01
epochs = 1000

loss_history = []

for i in range(epochs):
    m, b = gradient_descent(m, b, X, y, learning_rate)
    loss = loss_function(m, b, X, y)
    loss_history.append(loss)

    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

# -----------------------------
# 5. Final Parameters
# -----------------------------
print("\nFinal learned parameters:")
print(f"m ≈ {m:.2f} (expected ~3)")
print(f"b ≈ {b:.2f} (expected ~5)")

# -----------------------------
# 6. Plot Data + Regression Line
# -----------------------------
y_pred = m * X + b

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.legend()
plt.title("Linear Regression from Scratch")
plt.show()

# -----------------------------
# 7. Plot Loss Curve
# -----------------------------
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss decreasing over time")
plt.show()

"""
Deep Learning Term		Your Code
Model			        y = mx + b
Loss			        loss_function()
Forward pass			y_pred = m*X + b
Backward pass			gradient calculation
Optimizer			    update step
Training loop			epochs loop

"""
