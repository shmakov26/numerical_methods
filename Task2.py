import math
import matplotlib.pyplot as plt

def task1():
    A, B = -4.0, 4.0
    ALPHA, BETA = -1.0, 1.0
    D_MU = 0.1
    EPS = 1e-6
    METHOD = 'bisection' # 'bisection' или 'newton'
    SCAN_STEP = 0.01

    def f(x, mu):
        return math.sin(x) - mu

    def df(x, mu, h=1e-7):
        return (f(x + h, mu) - f(x - h, mu)) / (2.0 * h)

    def bisection(a, b, mu, eps):
        fa, fb = f(a, mu), f(b, mu)
        if fa * fb > 0:
            return None
        while b - a > eps:
            c = (a + b) / 2.0
            fc = f(c, mu)
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2.0

    def newton(x0, mu, eps, max_iter=100):
        x = x0
        for _ in range(max_iter):
            fx = f(x, mu)
            dfx = df(x, mu)
            if abs(dfx) < 1e-12:
                return None
            x_new = x - fx / dfx
            if abs(x_new - x) < eps:
                return x_new
            x = x_new
        return x

    mu_values = []
    x_values = []

    mu = ALPHA
    while mu <= BETA + 1e-9:
        roots = []
        x = A
        while x < B:
            x_end = min(x + SCAN_STEP, B)
            r = bisection(x, x_end, mu, EPS) if METHOD == 'bisection' else newton((x + x_end) / 2.0, mu, EPS)
            if r is not None and A - EPS <= r <= B + EPS:
                r = round(r, 6)
                if not any(abs(r - r0) < EPS for r0 in roots):
                    roots.append(r)
            x = x_end
        for r in sorted(roots):
            mu_values.append(mu)
            x_values.append(r)
        mu += D_MU

    plt.figure(figsize=(8, 6))
    plt.plot(mu_values, x_values, 'bo', markersize=3, label='Найденные корни')
    plt.xlabel('Параметр μ', fontsize=12)
    plt.ylabel('Корень x', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def task2():
    EPS = 1e-8
    MAX_ITER = 100
    X0 = [0.0, 0.0, 0.0, 0.0]

    def F(x):
        return [
            0.77 * x[0] + 0.04 * x[1] - 0.21 * x[2] + 0.18 * x[3] - 1.24,
            -0.45 * x[0] + 1.23 * x[1] - 0.06 * x[2] + 0.88,
            -0.26 * x[0] - 0.34 * x[1] + 1.11 * x[2] - 0.62,
            -0.05 * x[0] + 0.26 * x[1] - 0.34 * x[2] + 1.12 * x[3] + 1.17
        ]

    def jacobi_solve(A, b, eps_jac=1e-10, max_iter_jac=2000):
        n = len(b)
        x = [0.0] * n

        for _ in range(max_iter_jac):
            x_new = [0.0] * n

            for i in range(n):
                s = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - s) / A[i][i]

            if max(abs(x_new[i] - x[i]) for i in range(n)) < eps_jac:
                return x_new

            x = x_new

        return x

    def jacobian(F, x, h=1e-7):
        n = len(x)
        J = [[0.0] * n for _ in range(n)]
        fx = F(x)

        for j in range(n):
            xp = x[:]
            xp[j] += h
            fp = F(xp)
            for i in range(n):
                J[i][j] = (fp[i] - fx[i]) / h

        return J

    def newton_system(F, x0, eps, max_iter):
        x = x0[:]

        for _ in range(max_iter):
            fx = F(x)
            J = jacobian(F, x)
            dx = jacobi_solve(J, [-v for v in fx])
            x = [x[i] + dx[i] for i in range(len(x))]

            if max(abs(d) for d in dx) < eps:
                return x

        return x

    sol = newton_system(F, X0, EPS, MAX_ITER)
    print(" ".join(f"{v:.6f}" for v in sol))


if __name__ == '__main__':
    print("Задача 2. Ответ:")
    task2()
    task1()
