import numpy as np
import matplotlib.pyplot as plt
import math

def f(x, y):
    return np.sin(x/2) * np.cos(y/2)

A, B = -5, 5 # предел по x
C, D = -5, 5 # предел по y
EPSILON = 1e-4
X0, Y0 = 0.5, 1.0
METHOD = 'coordinate' # coordinate или gradient
ALPHA_MAX = 5.0
MAX_ITER = 5000

def golden_section_1d(func, a, b, eps):
    phi = (1 + math.sqrt(5)) / 2
    ak, bk = a, b
    x_est = a
    while bk - ak >= eps:
        alpha = bk - (bk - ak) / phi
        beta = ak + (bk - ak) / phi
        if func(alpha) <= func(beta):
            bk = beta
            x_est = alpha
        else:
            ak = alpha
            x_est = beta
    return x_est

def get_gradient(func, x, y, h=1e-6):
    return (func(x + h, y) - func(x - h, y)) / (2 * h), \
           (func(x, y + h) - func(x, y - h)) / (2 * h)

def coordinate_descent(func, a, b, c, d, x0, y0, eps):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(MAX_ITER):
        x_new = golden_section_1d(lambda t: func(t, y), a, b, eps)
        y_new = golden_section_1d(lambda t: func(x_new, t), c, d, eps)
        path.append((x_new, y_new))
        if math.hypot(x_new - x, y_new - y) < eps:
            return x_new, y_new, path
        x, y = x_new, y_new
    return x, y, path

def gradient_descent(func, a, b, c, d, x0, y0, eps, alpha_max):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(MAX_ITER):
        gx, gy = get_gradient(func, x, y)
        if math.hypot(gx, gy) < 1e-9:
            return x, y, path
        alpha = golden_section_1d(lambda t: func(x - t * gx, y - t * gy), 0, alpha_max, eps)
        x_new = x - alpha * gx
        y_new = y - alpha * gy
        path.append((x_new, y_new))
        if math.hypot(x_new - x, y_new - y) < eps:
            return x_new, y_new, path
        x, y = x_new, y_new
    return x, y, path

if METHOD == 'coordinate':
    res_x, res_y, path = coordinate_descent(f, A, B, C, D, X0, Y0, EPSILON)
else:
    res_x, res_y, path = gradient_descent(f, A, B, C, D, X0, Y0, EPSILON, ALPHA_MAX)

x_grid = np.linspace(A, B, 100)
y_grid = np.linspace(C, D, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=50)
px, py = zip(*path)
plt.plot(px, py, 'ro-', linewidth=1.5, markersize=4)
plt.plot(res_x, res_y, markersize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()