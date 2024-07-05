import numpy as np
from numpy.linalg import inv

class LinearRegression:

  def __init__(self, X: np.array, y: np.array, lmbda: float = 10e-11):
    '''
    `X` is the training data - a numpy array where each row is a training point

    `y` is the target data - a numpy array where each row is a target point
    '''
    self.X = X
    self.y = np.array([y])

    self.w = X @ X.T + lmbda * np.eye(X.shape[0])

    self.w = inv(self.w)

    self.w = self.y @ self.w @ self.X

  def __call__(self, x: np.array):
    return self.w @ x


# test the model
  
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression(X, y)

print(model(np.array([100, 100])))