import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve
import pandas as pd

# Problem settings
a, b = 0, 1
alpha, beta = 1, 2
h = 0.1
n = int((b - a) / h)

x = np.linspace(a, b, n + 1)

def r(x):
    return (1 - x**2) * np.exp(-x)

# ------------------------------------------------
# (a) Shooting Method
# ------------------------------------------------
def f_system(x, Y):
    y, yp = Y
    return [yp, -(x + 1) * yp + 2 * y + r(x)]

def solve_shooting():
    sol1 = solve_ivp(f_system, [a, b], [alpha, 0], t_eval=x)
    sol2 = solve_ivp(f_system, [a, b], [0, 1], t_eval=x)

    y1b = sol1.y[0, -1]
    y2b = sol2.y[0, -1]
    c = (beta - y1b) / y2b

    y = sol1.y[0] + c * sol2.y[0]
    return y

# ------------------------------------------------
# (b) Finite-Difference Method
# ------------------------------------------------
def solve_finite_difference():
    A = np.zeros((n - 1, n - 1))
    F = np.zeros(n - 1)

    for i in range(1, n):
        xi = x[i]
        pi = -(xi + 1)
        qi = 2
        ri_val = r(xi)

        a_lower = 1 / h**2 - pi / (2 * h)
        a_diag = -2 / h**2 + qi
        a_upper = 1 / h**2 + pi / (2 * h)

        if i != 1:
            A[i - 1, i - 2] = a_lower
        A[i - 1, i - 1] = a_diag
        if i != n - 1:
            A[i - 1, i] = a_upper

        F[i - 1] = ri_val

    F[0] -= (1 / h**2 - (-(x[1] + 1)) / (2 * h)) * alpha
    F[-1] -= (1 / h**2 + (-(x[-2] + 1)) / (2 * h)) * beta

    y_inner = solve(A, F)
    y = np.concatenate(([alpha], y_inner, [beta]))
    return y

# ------------------------------------------------
# (c) Variation Approach
# ------------------------------------------------
def phi(j, x):
    return np.sin(j * np.pi * x)

def dphi(j, x):
    return j * np.pi * np.cos(j * np.pi * x)

def y1_variation(x):
    return alpha + (beta - alpha) * x

def solve_variation(m=5):
    A = np.zeros((m, m))
    b_vec = np.zeros(m)

    for i in range(m):
        for j in range(m):
            A[i, j], _ = quad(lambda x: dphi(i+1, x)*dphi(j+1, x) + 2*phi(i+1, x)*phi(j+1, x), 0, 1)
        b_vec[i], _ = quad(lambda x: r(x)*phi(i+1, x) - (dphi(i+1, x)*(beta - alpha) + 2*phi(i+1, x)*y1_variation(x)), 0, 1)

    c = solve(A, b_vec)

    def y_approx(x):
        return y1_variation(x) + sum(c[j] * phi(j+1, x) for j in range(m))

    y_vals = np.array([y_approx(xi) for xi in x])
    return y_vals

# ------------------------------------------------
# Main: Compute and show data
# ------------------------------------------------
y_s = solve_shooting()
y_f = solve_finite_difference()
y_v = solve_variation()

# Combine into DataFrame
df = pd.DataFrame({
    "x": x,
    "Shooting": y_s,
    "Finite-Difference": y_f,
    "Variation": y_v
})

print(df.to_string(index=False))



# ------------------------------------------------
# Plotting
# ------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x, y_s, 'r-o', label='Shooting Method')
plt.plot(x, y_f, 'g-s', label='Finite-Difference Method')
plt.plot(x, y_v, 'b-^', label='Variation Approach')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Comparison of BVP Approximation Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
