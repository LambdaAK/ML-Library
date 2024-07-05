import numpy as np
from collections import deque

class DBSCAN:
  '''
  Assumptons:
  - Clusters are dense regions of points seperated by low-density regions
  - All clusters have similar density
  '''

  def __init__(self, X, eps, min_pts):
    '''
    `X` is the training data: a numpy array where each row is a training point

    `eps` is the maximum distance between two points for one to be considered as in the neighborhood of the other

    `min_pts` is the number of points in a neighborhood for a point to be considered as a core point
    '''

    self.X = X
    self.eps = eps
    self.min_pts = min_pts
    
    # compute the core points

    self.core_points = {}

    for i, x in enumerate(X):
      points_in_neighborhood = []
      for j, y in enumerate(X):
        if np.linalg.norm(x - y) < eps:
          points_in_neighborhood.append(j)

      if len(points_in_neighborhood) >= min_pts:
        self.core_points[i] = points_in_neighborhood

    self.cluster_assignment = {} # cluster number -> list of point indices
    self.noise_points = []

    self.train()
    

  def train(self):
    '''
    Train the DBSCAN model
    '''

    # make a copy of core_points

    remaining_core_points = list(self.core_points.keys())
    assigned_points = set()

    while len(remaining_core_points) > 0:
      # randomly select an unassigned core point and assign it to a new cluster
      core_point = np.random.choice(remaining_core_points)

      remaining_core_points.remove(core_point)

      core_points_to_add = deque([core_point])

      cluster = []

      while len(core_points_to_add) > 0:
        current_core_point = core_points_to_add.popleft()
        # add the core point to the cluster
        if current_core_point not in assigned_points:
          cluster.append(int(current_core_point))
          assigned_points.add(current_core_point)
        assigned_points.add(current_core_point)
        # add all points in the neighborhood of the core point to the cluster
        # if any of the points are a core point, add it to the queue
        for point in self.core_points[current_core_point]:
          # if the point is not already in a cluster
          if point not in assigned_points:
            cluster.append(point)
            assigned_points.add(point)
            
          if point in remaining_core_points:
            core_points_to_add.append(point)
            remaining_core_points.remove(point)
      # assign all points in the cluster to the same cluster number
      self.cluster_assignment[len(self.cluster_assignment)] = cluster

    # assign all remaining points to noise
    for i in range(len(self.X)):
      if i not in [point for cluster in self.cluster_assignment.values() for point in cluster]:
        self.noise_points.append(i)


      # assign all core points with a core point neighbor in the neighborhood to the same cluster
