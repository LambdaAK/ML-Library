import numpy

'''
K-Nearest Neighbors model
'''
class KNN:
  def __init__(self, k: int, X: numpy.array, y: numpy.array):
    '''
    `k` is the number of neighbors to consider

    `X` is the training data - a 2D numpy array where each row is a training point

    `y` is the training labels
    '''
    self.k = k
    self.X = X
    self.y = y

  def __call__(self, x: numpy.array):
    '''
    Predict the label of a single input `x`
    '''
    # Calculate the distance between `x` and all training data`
    distances = numpy.linalg.norm(self.X - x, axis=1)

    # Find the indices of the `k` nearest neighbors
    nearest_indices = numpy.argsort(distances)[:self.k]

    # Get the labels of the `k` nearest neighbors

    nearest_labels = self.y[nearest_indices]

    # Return the most common label

    return numpy.bincount(nearest_labels).argmax()
