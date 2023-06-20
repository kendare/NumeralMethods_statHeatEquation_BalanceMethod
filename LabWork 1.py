import numpy as np
from matplotlib import pyplot as plt

# Значения основных параметров Вариант 12 (2)
KSI = 0.25
MU1 = 0
MU2 = 1
x0 = 0
xn = 1
n = 10000
h = float(1/n)
v = np.zeros(n+1)
v[-1] = MU2
X = np.linspace(x0, xn, n+1)

# Коэффициенты ai
def ai(k1, k2, kmid, x):
    if KSI >= x:
        return k1(x - h/2)
    elif KSI <= x - h:
        return k2(x-h/2)
    else:
        return kmid(x)

def di(q1, q2, qmid, x):
    if KSI >= x+h/2:
        return q1(x)
    elif KSI <= x - h/2:
        return q2(x)
    else:
        return qmid(x)

def fi(f1, f2, fmid, x):
    if KSI >= x + h/2:
        return f1(x)
    elif KSI <= x - h/2:
        return f2(x)
    else:
        return fmid(x)

def k1TEST(x):
    return 0.5

def k2TEST(x):
    return 1.25

def q1TEST(x):
    return 1

def q2TEST(x):
    return 0.0625

def f1TEST(x):
    return 1

def f2TEST(x):
    return 2.5

def analytSol():
    a = 2**0.5
    b = 0.05**0.5
    c1 = 0.5874420413692708
    c2 = -1.5874420413692708
    c3 = -16.23731986542228
    c4 = -23.37825943652864
    g = np.zeros(n+1)
    for i in range(n+1):
        x = i * h
        if x < KSI:
            g[i] = c1 * np.exp(a*x) + c2 * np.exp(-a*x) + 1
        else:
            g[i] = c3 * np.exp(b*x) + c4 * np.exp(-b*x) + 40
    return g

def numericSol():
    alpha = [0]
    beta = [0]
    for i in range(1, n):
        x = i * h
        A = ai(k1TEST, k2TEST, k2TEST, x) / h**2
        B = ai(k1TEST, k2TEST, k2TEST, x+h) / h**2
        C = A + B + di(q1TEST, q2TEST, q2TEST, x)
        alpha.append(B / (C - A * alpha[i-1]))
        beta.append((fi(f1TEST, f2TEST, f2TEST, x) + A * beta[i-1]) / (C - A * alpha[i-1]))
    for i in range(n, 1, -1):
        v[i-1] = alpha[i-1] * v[i] + beta[i-1]
    y = analytSol()
    return v, y

v, y = numericSol()
Eps1 = max(abs(v - y))
print(Eps1)
plt.plot(X, v, 'r')
plt.plot(X, y, 'g')
plt.show()
