import math
import matplotlib.pyplot as plt

TASK = "2"
EPS = 1e-4
A, B = 0.0, 5.0
METHOD = "5" # Методы: 1-Левые, 2-Правые, 3-Средние, 4-Трапеции, 5-Симпсон

# Задача 1
def f_task1(x, t):
    return math.sin(x + t)

# Задача 2.
def f_task2(x, t):
    if x <= 0:
        return 1.0 if t == 0 else 0.0
    return (x ** t) * math.exp(-x)

def left_rect(f, a, b, t, N):
    h = (b - a) / N
    return h * sum(f(a + i * h, t) for i in range(N))

def right_rect(f, a, b, t, N):
    h = (b - a) / N
    return h * sum(f(a + (i + 1) * h, t) for i in range(N))

def mid_rect(f, a, b, t, N):
    h = (b - a) / N
    return h * sum(f(a + (i + 0.5) * h, t) for i in range(N))

def trapezoid_rule(f, a, b, t, N):
    h = (b - a) / N
    s = 0.5 * (f(a, t) + f(b, t))
    for i in range(1, N):
        s += f(a + i * h, t)
    return h * s

def simpson_rule(f, a, b, t, N):
    h = (b - a) / N
    s = 0.0
    for i in range(N):
        x0 = a + i * h
        x_mid = a + (i + 0.5) * h
        x1 = a + (i + 1) * h
        s += (h / 6.0) * (f(x0, t) + 4.0 * f(x_mid, t) + f(x1, t))
    return s

METHODS = {
    '1': ('Левые прямоугольники', left_rect, 1),
    '2': ('Правые прямоугольники', right_rect, 1),
    '3': ('Средние прямоугольники', mid_rect, 2),
    '4': ('Трапеции', trapezoid_rule, 2),
    '5': ('Симпсон', simpson_rule, 4)
}

# Правило Рунге
def runge_integrate(f, a, b, t, method_func, s, eps, N_start=2, max_N=10**6):
    N = N_start
    I_N = method_func(f, a, b, t, N)
    while N < max_N:
        N *= 2
        I_2N = method_func(f, a, b, t, N)
        if abs(I_2N - I_N) / (2**s - 1) < eps:
            return I_2N
        I_N = I_2N
    return I_2N

def estimate_segment(f, a, b, t, N=10):
    h = (b - a) / N
    s = 0.5 * (f(a, t) + f(b, t))
    for i in range(1, N):
        s += f(a + i * h, t)
    return h * s

def find_bounds(f, t, eps):
    a_eff = 0.0
    if t < 0:
        delta = 1.0
        while abs(estimate_segment(f, delta/2, delta, t)) >= eps:
            delta /= 2.0
            if delta < 1e-12: break
        a_eff = delta

    b_eff = 1.0
    while abs(estimate_segment(f, b_eff, 2*b_eff, t)) >= eps:
        b_eff *= 2.0
        if b_eff > 1e6: break
    return a_eff, b_eff

def main():
    _, method_func, s = METHODS[METHOD]
    t_values = [A + i * (B - A) / 20 for i in range(21)]
    I_values = []

    for t in t_values:
        if TASK == '1':
            I_values.append(runge_integrate(f_task1, 0.0, math.pi, t, method_func, s, EPS))
        else:
            if t <= -1:
                I_values.append(float('nan'))
                continue
            a, b = find_bounds(f_task2, t, EPS)
            I_values.append(runge_integrate(f_task2, a, b, t, method_func, s, EPS))

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, I_values, marker='o', linestyle='-')
    plt.xlabel('t')
    plt.ylabel('I(t)')
    plt.title(f'Задача {TASK}, ε={EPS}, метод: {METHODS[METHOD][0]}')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()