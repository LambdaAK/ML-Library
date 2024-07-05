import numpy as np

class PCA:

  def __init__(self, X: np.array, n_components: int):
    '''
    `X` is the training data: a numpy array where each row is a training point

    `n_components` is the number of principal components to keep
    '''

    self.X = X
    self.n_components = n_components
    # make Z have dimensions (n_samples, n_components)
    self.Z = np.zeros((X.shape[0], n_components))
    self.train()

  def train(self):

    # compute the covariance matrix of X
    cov_X = np.cov(self.X.T)
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_X)
    # sort the eigenvectors by their eigenvalues
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]
    # take the first n_components eigenvectors
    self.W = eig_vecs[:, :self.n_components]
    # project the data onto the principal components
    self.Z = self.X @ self.W

  def get_data(self):
    return self.Z
  