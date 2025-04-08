import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

# === 被積分函數 ===
def f(x):
    return x**2 * np.log(x)

# === Gaussian Quadrature 主程式 ===
def gaussian_quadrature(f, a, b, n):
    x, w = leggauss(n)
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(t))

# === 計算參數 ===
a, b = 1, 1.5
exact, _ = quad(f, a, b)

# === 執行 n=3 與 n=4 的 Gaussian Quadrature ===
results = {}
for n in [3, 4]:
    approx = gaussian_quadrature(f, a, b, n)
    error = abs(approx - exact)
    results[n] = (approx, error)

# === 印出結果 ===
print("=== Gaussian Quadrature ===")
print(f"Exact Value: {exact:.12f}")
for n, (val, err) in results.items():
    print(f"n = {n} | Approx = {val:.12f} | Error = {err:.2e}")
