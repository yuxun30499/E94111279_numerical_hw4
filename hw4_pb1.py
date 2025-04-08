import numpy as np

# 定義 f(x)
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 通用設定
a, b = 1, 2
h = 0.1
n = int((b - a) / h)

# 1a. Composite Trapezoidal Rule
def composite_trapezoidal(f, a, b, h):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# 1b. Composite Simpson's Rule
def composite_simpson(f, a, b, h):
    if n % 2 == 1:
        raise ValueError("n 必須是偶數才能使用 Simpson’s Rule")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1::2]) + y[-1])

# 1c. Composite Midpoint Rule
def composite_midpoint(f, a, b, h):
    midpoints = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(midpoints))

# 計算結果
result_trap = composite_trapezoidal(f, a, b, h)
result_simp = composite_simpson(f, a, b, h)
result_mid = composite_midpoint(f, a, b, h)

# 顯示
print("Composite Trapezoidal:", result_trap)
print("Composite Simpson's  :", result_simp)
print("Composite Midpoint   :", result_mid)
