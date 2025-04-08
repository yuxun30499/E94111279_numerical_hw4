import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import dblquad

# === 定義被積函數 ===
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# === 真實值（用 dblquad 做） ===
exact, _ = dblquad(lambda y, x: f(x, y), 0, np.pi/4, lambda x: np.sin(x), lambda x: np.cos(x))

# === 3a. Simpson’s Rule (n=4, m=4) ===
def simpsons_2d(f, a, b, n, m):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = 0
    for i in range(0, n + 1):
        xi = x[i]
        wi = 1
        if i == 0 or i == n:
            wi = 1
        elif i % 2 == 0:
            wi = 2
        else:
            wi = 4

        ymin = np.sin(xi)
        ymax = np.cos(xi)
        k = (ymax - ymin) / m
        y = np.linspace(ymin, ymax, m + 1)

        inner_sum = 0
        for j in range(0, m + 1):
            yj = y[j]
            wj = 1
            if j == 0 or j == m:
                wj = 1
            elif j % 2 == 0:
                wj = 2
            else:
                wj = 4
            inner_sum += wj * f(xi, yj)
        inner_sum *= k / 3
        result += wi * inner_sum
    result *= h / 3
    return result

# === 3b. Gaussian Quadrature (n=3, m=3) ===
def gaussian_2d(f, a, b, n, m):
    xi, wi = leggauss(n)
    eta, wj = leggauss(m)
    result = 0

    for i in range(n):
        # x: [a,b] -> [-1,1]
        x = 0.5 * (b - a) * xi[i] + 0.5 * (a + b)
        wx = wi[i] * 0.5 * (b - a)

        ymin = np.sin(x)
        ymax = np.cos(x)

        for j in range(m):
            # y: [ymin,ymax] -> [-1,1]
            y = 0.5 * (ymax - ymin) * eta[j] + 0.5 * (ymin + ymax)
            wy = wj[j] * 0.5 * (ymax - ymin)

            result += wx * wy * f(x, y)
    return result

# === 結果 ===
simpson_result = simpsons_2d(f, 0, np.pi/4, 4, 4)
gauss_result = gaussian_2d(f, 0, np.pi/4, 3, 3)

# === 印出來 ===
print("=== 題目 3 ===")
print(f"3a. Simpson's Result  : {simpson_result:.12f}")
print(f"3b. Gaussian Result    : {gauss_result:.12f}")
print(f"3c. Exact Value (dblquad): {exact:.12f}")
print(f"Simpson Error          : {abs(simpson_result - exact):.2e}")
print(f"Gaussian Error         : {abs(gauss_result - exact):.2e}")
# 精確值大約是0.5116666666 但程式跑出來約有0.0002的誤差