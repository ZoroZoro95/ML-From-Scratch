# 🧠 ML-From-Scratch

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<br />

> **"What I cannot create, I do not understand."** — *Richard Feynman*

This repository is a journey through the fundamentals of Machine Learning, implementing core algorithms from the ground up using only **NumPy** and **Python**. No high-level libraries like Scikit-Learn or TensorFlow—just math and code.

---

## 🚀 Implemented Algorithms

| Algorithm | Status | Complexity | Description |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | ✅ | $O(n)$ | Simple Gradient Descent implementation for 1D features. |
| **Logistic Regression** | 🛠️ | - | *Coming soon...* |
| **K-Nearest Neighbors** | 🛠️ | - | *Coming soon...* |
| **Neural Network** | 🛠️ | - | *Coming soon...* |

---

## 📈 Featured: Linear Regression

The current implementation features a **Linear Regression** model optimized via **Gradient Descent**.

### 🧪 The Math
We find the line $y = mx + b$ by minimizing the **Mean Squared Error (MSE)**:
$$J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{i})^2$$

### 🛠️ Core Components
- **Forward Pass**: $y_{pred} = mX + b$
- **Loss Function**: Mean Squared Error
- **Optimizer**: Vanilla Gradient Descent

```python
def gradient_descent(m, b, X, y, learning_rate):
    # Calculate gradients
    Dm = (2/n) * np.sum(X * error)
    Db = (2/n) * np.sum(error)
    # Update parameters
    m = m - learning_rate * Dm
    b = b - learning_rate * Db
    return m, b
```

---

## 🛠️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/ZoroZoro95/ML-From-Scratch.git
cd ML-From-Scratch
```

### 2. Install dependencies
```bash
pip install numpy matplotlib
```

### 3. Run the scripts
```bash
python linear_regression.py
```

---

## 🗺️ Roadmap
- [x] Linear Regression (One Feature)
- [ ] Multiple Linear Regression
- [ ] Logistic Regression (Binary Classification)
- [ ] K-Means Clustering
- [ ] Decision Trees
- [ ] Basic Neural Network (Backpropagation)

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/ZoroZoro95">ZoroZoro</a></sub>
</div>
