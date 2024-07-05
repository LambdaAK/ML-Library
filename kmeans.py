from typing import List
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
      
      if len(points) == 0:
        continue

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

      loss_before = self.loss_function()

      # Update the centroids
      self.update_centroids()
      self.assign_clusters()

      loss_after = self.loss_function()

      if loss_before == loss_after:
        break

  def loss_function(self):
    # the mean distance between a point and its centroid
    s = 0
    for cluster_number in self.cluster_assignment.keys():
      for point_index in self.cluster_assignment[cluster_number]:
        s += numpy.linalg.norm(self.X[point_index] - self.centroids[cluster_number])
    return s / len(self.X)
  

  def wcss(self):
    '''
    Within cluster sum of squares
    '''
    s = 0
    for cluster_number in self.cluster_assignment.keys():
      for point_index in self.cluster_assignment[cluster_number]:
        s += numpy.linalg.norm(self.X[point_index] - self.centroids[cluster_number]) ** 2
    return s

  @staticmethod
  def multiple_trials(k: int, X: numpy.array, n: int = 10):
    '''
    Run the kmeans algorithm n times and return the best model
    '''
    best_model = None
    best_loss = float('inf')
    for _ in range(n):
      model = KMeans(k, X)
      loss = model.loss_function()
      if loss < best_loss:
        best_loss = loss
        best_model = model
    return best_model
  
  @staticmethod
  def kmeans_no_k(X: numpy.array, max_k: int = None, iter_per_model: int = 10):
    '''
    Run the kmeans algorithm for k = 1 to max_k and return the model that
    is most likely to have the correct number of clusters

    The value of k where the loss functions starts decreasing much more slowly is the best value of k
    '''
    
    models: List[KMeans] = []

    for k in range(1, max_k):
      model = KMeans.multiple_trials(k, X, iter_per_model)
      models.append(model)

    # compute wcss for each model
    wcss_values = [model.wcss() for model in models]

    # find the percent change in wcss between each model

    wcss_percent_change = [abs(wcss_values[i] - wcss_values[i + 1]) / wcss_values[i] for i in range(len(wcss_values) - 1)]

    # pick the k where the percent change is the smallest, starting from the end

    best_k = max_k - numpy.argmax(wcss_percent_change[::-1])

    return models[best_k - 1]


# test the model
  
X = numpy.array([
  [1,1],
  [1,2],
  [2,1],
  [2,2],

  [1,10],
  [1,11],
  [2,10],
  [2,11],

  [20, 21],
  [20, 22],
  [21, 21],
  [21, 22],

  [7, 2],
  [7, 1],
  [8, 2],
  [8 ,1],


  # another cluster

  [100, 100],
  [100, 101],
  [101, 100],
  [101, 101],

  [1000, 1000],
  [1000, 1001],
  [1001, 1000],
  [1001, 1001],

  [10000, 10000],
  [10000, 10001],
  [10001, 10000],
  [10001, 10001],

  [100000, 100000],
  [100000, 100001],
  [100001, 100000],
  [100001, 100001],

  [1000000, 1000000],
  [1000000, 1000001],
  [1000001, 1000000],
  [1000001, 1000001],

])

model = KMeans.kmeans_no_k(X, 20, 20)

print(model.k)
