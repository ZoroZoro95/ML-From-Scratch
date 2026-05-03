import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Synthetic multi-feature data
# -----------------------------
np.random.seed(42)

n = 100
X = np.random.rand(n, 2)  # 2 features

# true relationship: y = 4*x1 - 2*x2 + 3
y = 4*X[:,0] - 2*X[:,1] + 3 + np.random.randn(n) * 0.5

# -----------------------------
# 2. Loss function
# -----------------------------
def loss_function(w, b, X, y):
    y_pred = X.dot(w) + b
    return np.mean((y_pred - y) ** 2)

# -----------------------------
# 3. Gradient descent
# -----------------------------
def gradient_descent(w, b, X, y, lr):
    n = len(X)

    y_pred = X.dot(w) + b
    error = y_pred - y

    # gradients
    Dw = (2/n) * X.T.dot(error)   # shape (d,)
    Db = (2/n) * np.sum(error)

    # update
    w = w - lr * Dw
    b = b - lr * Db

    return w, b

# -----------------------------
# 4. Training loop
# -----------------------------
w = np.zeros(X.shape[1])  # (2,)
b = 0

lr = 0.1
epochs = 500

for i in range(epochs):
    w, b = gradient_descent(w, b, X, y, lr)

    if i % 100 == 0:
        loss = loss_function(w, b, X, y)
        print(f"Epoch {i}, Loss: {loss:.4f}")

# -----------------------------
# 5. Results
# -----------------------------
print("\nLearned parameters:")
print("w:", w)   # should be ~ [4, -2]
print("b:", b)   # should be ~ 3


from mpl_toolkits.mplot3d import Axes3D

# scatter actual data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = X[:, 0]
x2 = X[:, 1]

ax.scatter(x1, x2, y, color='blue')

# create grid for plane
x1_surf, x2_surf = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 20),
    np.linspace(x2.min(), x2.max(), 20)
)

# plane equation
y_surf = w[0]*x1_surf + w[1]*x2_surf + b

# plot plane
ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.5)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

plt.title("Hyperplane (actually a plane in 3D)")
plt.show()