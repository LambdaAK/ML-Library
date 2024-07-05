# ML-Library

This is a machine learning library that I am building using only numpy and sympy.

Currently, this library supports the following algorithms

1. Perceptron
2. K-Nearest Neighbors
3. K-Means Clustering
4. DBSCAN: Density-Based Spatial Clustering of Applications with Noise
5. Linear Regression
6. PCA: Principal Component Analysis
7. Univariate function optimization using Gradient Descent

## Example usages

## Perceptron
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = Perceptron(X, y)

print(model(np.array([0.5, 0.5])))
```

## K-Nearest Neighbors
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = KNearestNeighbors(X, y, 3)

print(model(np.array([0.5, 0.5])))
```

## K-Means Clustering
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])

model = KMeans(X, 2)

centroids = model.centroids
cluster_assignment = model.cluster_assignment
```

## DBSCAN
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])

model = DBSCAN(X, 2, 2)

cluster_assignment = model.cluster_assignment
```

## Linear Regression
```python
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3, 4, 5])

model = LinearRegression(X, y)

print(model(np.array([6])))
```

## PCA
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])

model = PCA(X, 1)

Z = model.get_data()
```

## Univariate function optimization using Gradient Descent
```python
x = sp.symbols('x')
f = x**2

optimizer = UnivarOptimizer(f, x, 0.1)
minimizer = optimizer.train()
```



