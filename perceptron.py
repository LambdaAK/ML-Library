import numpy

class Perceptron:
  '''
  Perceptron model - used for binary classification over a linearly seperable dataset
  '''
  def __init__(self, X: numpy.array, y: numpy.array):
    '''
    `X` is the training data - a 2D numpy array where each row is a training point

    `y` is the training labels
    '''
    self.X = X
    self.y = y
    self.weights = numpy.zeros(X.shape[1])
    self.train()

  def __call__(self, x: numpy.array):
    '''
    Predict the label of a single input `x`
    '''
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

