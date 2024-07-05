import numpy as np
import sympy as sp

class UnivarOptimizer:

  def __init__(self, f, lr = 0.01):
    '''
    `f` is the function to optimize
    '''
    self.f = f
    self.lr = lr

    
  def train(self, iter = 1000):
    '''
    Train the optimizer
    '''
    x = 0
    for _ in range(iter):
      f_prime = sp.diff(self.f)
      x = x - self.lr * f_prime.subs('x', x)

    return x

