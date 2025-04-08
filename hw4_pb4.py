import numpy as np
from scipy.integrate import quad

# === a. 積分 f(x) = x^(-1/4) * sin(x), x ∈ [0, 1] ===
def f1(x):
    return x**(-1/4) * np.sin(x)

# === b. 改變變數後的函數 ===
def f2(t):
    return t**2 * np.sin(1 / t)

# === Composite Simpson's Rule 通用寫法 ===
def simpson_composite(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n 必須為偶數")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1::2]) + y[-1])

# === 計算 a. 使用 epsilon 避免 0 ===
eps = 1e-8
result_a = simpson_composite(f1, eps, 1, 4)
exact_a, _ = quad(f1, 0, 1)

# === 計算 b. 利用變數變換後 t ∈ (0, 1] ===
result_b = simpson_composite(f2, 1e-8, 1, 4)
exact_b, _ = quad(f2, 0, 1)

# === 印出結果 ===
print(f"a.")
print(f"Simpson Result : {result_a:.12f}")
print(f"Exact (quad)   : {exact_a:.12f}")
print(f"Error          : {abs(result_a - exact_a):.2e}")
print()
print(f"b.")
print(f"Simpson Result : {result_b:.12f}")
print(f"Exact (quad)   : {exact_b:.12f}")
print(f"Error          : {abs(result_b - exact_b):.2e}")
