import numpy
import numpy as np

class Perceptron:
  '''
  Perceptron model - used for binary classification over a linearly seperable dataset
  '''
  
  def __init__(self, X: numpy.array, y: numpy.array):
    '''
    `X` is the training data - a 2D numpy array where each row is a training point

    `y` is the training labels
    '''

    # Ensure that the input data dimensions are correct

    self.check_invariants(X, y)

    self.X = X

    # Add a column of ones to the input data
    self.X = numpy.hstack([self.X, numpy.ones((self.X.shape[0], 1))])

    self.y = y
    self.weights = numpy.zeros(X.shape[1] + 1)
    self.training = True
    self.train()

  def check_invariants(self, X: numpy.array, y: numpy.array):
    '''
    Check if the input data dimensions are correct
    '''

    num_samples_x = X.shape[0]
    num_features = X.shape[1]
    num_samples_y = y.shape[0]

    assert num_samples_x == num_samples_y, 'Number of samples in X and y should be equal'


  def __call__(self, x: numpy.array):
    '''
    Predict the label of a single input `x`
    '''

    # Add a 1 to the input data

    if not self.training:
      x = np.hstack([x, 1])

    # use @ for matrix multiplication
    return 1 if self.weights.T @ x > 0 else 0

  def train(self, max_epochs: int = 1000):
    '''
    Train the perceptron model
    '''

    for epoch in range(max_epochs):
      # Iterate over each training example
      updates = 0
      for x, label in zip(self.X, self.y):
        # Predict the label
        prediction = self(x)
        if prediction != label:
          # Update the weights, since the prediction is incorrect
          self.weights += label * x
          updates += 1
        
      # If no updates were made, the model has converged
      if updates == 0:
        break

    self.training = False

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = Perceptron(X, y)

print(model(np.array([0.5, 0.5])))