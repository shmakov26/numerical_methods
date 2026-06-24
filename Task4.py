import numpy as np
import matplotlib.pyplot as plt

a = 0.0
b = 1.0
h = 1e-4
M = 1000
N = 5

def p(t): return t
def q(t): return 1.0
def f(t): return np.exp(t)

t = np.linspace(a, b, M)
dt = t[1] - t[0]

def trapz(y):
    return np.sum((y[:-1] + y[1:]) / 2 * dt)

L_phi = np.zeros((N, M))
for i in range(1, N + 1):
    phi = np.sin(i * np.pi * t)
    phi_ph = np.sin(i * np.pi * (t + h))
    phi_mh = np.sin(i * np.pi * (t - h))
    dphi = (phi_ph - phi_mh) / (2 * h)
    ddphi = (phi_mh - 2 * phi + phi_ph) / (h**2)
    L_phi[i - 1] = ddphi + p(t) * dphi + q(t) * phi

A = np.zeros((N, N))
b_vec = np.zeros(N)

for j in range(1, N + 1):
    phi_j = np.sin(j * np.pi * t)
    b_vec[j-1] = trapz(f(t) * phi_j)
    for i in range(1, N + 1):
        A[j - 1, i - 1] = trapz(L_phi[i - 1] * phi_j)

def solve_linear(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()
    for k in range(n):
        imax = k + np.argmax(np.abs(A[k:, k]))
        A[[k, imax]] = A[[imax, k]]
        b[[k, imax]] = b[[imax, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

C = solve_linear(A, b_vec)

phi_matrix = np.sin(np.outer(np.arange(1, N + 1), np.pi * t))
x_N = C @ phi_matrix

plt.plot(t, x_N)
plt.grid(True)
plt.show()