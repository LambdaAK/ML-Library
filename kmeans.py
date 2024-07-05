import numpy

class KMeans:
  '''
    Assumptions:
    - We know there are k clusters
    - Euclidean distance measures similarity
    - Clusters are spherical
    - Clusters have similar variance
  '''

  def __init__(self, k: int, X: numpy.array):
    '''
    `k` is the number of clusters

    `X` is the training data - a numpy array where each row is a training point
    '''

    self.k = k
    self.X = X
    # randomly initialize k centroids
    self.centroids = X[numpy.random.choice(X.shape[0], k, replace=True)]
    self.cluster_assignment = { i: [] for i in range(k) }
    # randomly assign each point to a cluster
    for i, x in enumerate(X):
      cluster = numpy.random.randint(k)
      self.cluster_assignment[cluster].append(i)

    self.train()

  def update_centroids(self):
    '''
    Make each centroid the mean of the points assigned to it
    '''
    for cluster_number in self.cluster_assignment.keys():
      points = [self.X[i] for i in self.cluster_assignment[cluster_number]]
      self.centroids[cluster_number] = numpy.mean(points, axis=0)

  def assign_clusters(self):
    '''
      Assign each point to the closest centroid
    '''
    # clear the cluster assignment
    for cluster_number in self.cluster_assignment.keys():
      self.cluster_assignment[cluster_number] = []

    for i, x in enumerate(self.X):
      cluster_number = numpy.argmin([numpy.linalg.norm(x - centroid) for centroid in self.centroids])
      self.cluster_assignment[cluster_number].append(i)

  def train(self):
    '''
    Train the kmeans model
    '''
    for epoch in range(1000):
      # Update the centroids
      self.update_centroids()
      self.assign_clusters()

  def loss_function(self):
    # the mean distance between a point and its centroid
    s = 0
    for cluster_number in self.cluster_assignment.keys():
      for point_index in self.cluster_assignment[cluster_number]:
        s += numpy.linalg.norm(self.X[point_index] - self.centroids[cluster_number])
    return s / len(self.X)

