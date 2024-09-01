import numpy as np

def rbf_kernel(x, y, params):
    gamma = params.get('gamma', 1.0)
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def polynomial_kernel(x, y, params):
    degree = params.get('degree', 3)
    coef0 = params.get('coef0', 1)
    return (np.dot(x, y) + coef0) ** degree

def linear_kernel(x, y, params):
    return np.dot(x, y)

def sigmoid_kernel(x, y, params):
    gamma = params.get('gamma', 1.0)
    coef0 = params.get('coef0', 0.0)
    return np.tanh(gamma * np.dot(x, y) + coef0)

def laplacian_kernel(x, y, params):
    gamma = params.get('gamma', 1.0)
    return np.exp(-gamma * np.linalg.norm(x - y, ord=1))

def rational_quadratic_kernel(x, y, params):
    alpha = params.get('alpha', 1.0)
    length_scale = params.get('length_scale', 1.0)
    return (1 + np.linalg.norm(x - y) ** 2 / (2 * alpha * length_scale ** 2)) ** -alpha

def exponential_kernel(x, y, params):
    gamma = params.get('gamma', 1.0)
    return np.exp(-gamma * np.linalg.norm(x - y))

def compute_mmd(X, Y, kernel_params):
    kernel = kernel_params.pop('kernel', rbf_kernel)  # Default to RBF kernel if not provided
    m, n = len(X), len(Y)
    
    # Compute kernel sums
    K_XX = sum(kernel(x, x_prime, kernel_params) for i, x in enumerate(X) for j, x_prime in enumerate(X) if i != j)
    K_YY = sum(kernel(y, y_prime, kernel_params) for i, y in enumerate(Y) for j, y_prime in enumerate(Y) if i != j)
    K_XY = sum(kernel(x, y, kernel_params) for x in X for y in Y)
    
    # Compute MMD
    mmd = (K_XX / (m * (m - 1))) + (K_YY / (n * (n - 1))) - (2 * K_XY / (m * n))
    
    return np.sqrt(mmd)