import numpy as np
from matplotlib import pyplot as plt
import opt_functions
import time


# ---- parameters ----
n = 3
a = [0, 1, 0.5, -2]
m = 10000
sigma = 0.5
r = 4
D = 2 * r


# ---- define x and f(x) ----
rand_gen = np.random.RandomState(0)  # generate a random generator with a fixed seed
x = 2 * rand_gen.random_sample([m]) - 1
f_x = opt_functions.calculate_poly(x, a)
rand_gen = np.random.RandomState(1)  # generate a random generator with another fixed seed
y = f_x + rand_gen.normal(0, sigma, [m])


# Q10 - def a Project Stochastic Gradient Descent
def PSGD_poly(x, y, b, epsilon=0.001, T=1000, order=3, time_stop=None):
    X = opt_functions.calc_x_matrix(x)
    m = len(x)
    rand_gen = np.random.RandomState(4)  # generate a random generator with a fixed seed
    a0 = rand_gen.normal(0, 1, [order + 1])
    indices = rand_gen.choice(m, size=b, replace=False)
    t = 0
    g0 = opt_functions.calc_gradient(X[indices, :], y[indices], a0)
    if time_stop is not None:
        timeout = time.time() + time_stop
    while t < T and np.linalg.norm(g0) > epsilon:
        t += 1
        eta = D / (G * np.sqrt(t))
        a1 = a0 - eta * g0
        if np.linalg.norm(a1) > r:
            a1 = r * a1 / np.linalg.norm(a1)
        a0 = a1
        indices = rand_gen.choice(m, size=b, replace=False)
        g0 = opt_functions.calc_gradient(X[indices, :], y[indices], a0)
        if time_stop is not None:
            if time.time() > timeout:
                break
    return a0


# Q11
X = opt_functions.calc_x_matrix(x, n+1)
G = D * opt_functions.max_eig(X.transpose() @ X / m) + np.linalg.norm(X.transpose() @ y / m)
L = opt_functions.max_eig(X.transpose() @ X / m)
meo = opt_functions.min_eig(X.transpose() @ X / m)
h_star = opt_functions.calculate_poly(x, opt_functions.estimation_solve(x, y))
e_1 = []
e_2 = []
e_3 = []
e_4 = []

# run as function of steps
# T = np.arange(10, 10000, 10)
# for t in T:
#     print(t/1000)
#     # calc rems of mini=batch 1
#     a_1 = PSGD_poly(x, y, 1, T=t)
#     h_1 = opt_functions.calculate_poly(x, a_1)
#     e_1.append(opt_functions.calc_rmse(h_1, h_star))
#     # calc rems of mini=batch 10
#     a_2 = PSGD_poly(x, y, 10, T=t)
#     h_2 = opt_functions.calculate_poly(x, a_2)
#     e_2.append(opt_functions.calc_rmse(h_2, h_star))
#     # calc rems of mini=batch 100
#     a_3 = PSGD_poly(x, y, 100, T=t)
#     h_3 = opt_functions.calculate_poly(x, a_3)
#     e_3.append(opt_functions.calc_rmse(h_3, h_star))
#     # calc rems of mini=batch 10000
#     a_4 = PSGD_poly(x, y, 10000, T=t)
#     h_4 = opt_functions.calculate_poly(x, a_4)
#     e_4.append(opt_functions.calc_rmse(h_4, h_star))
# plt.xscale('log')
# plt.plot(T, e_1, label='b = 1')
# plt.plot(T, e_2, label='b = 10')
# plt.plot(T, e_3, label='b = 100')
# plt.plot(T, e_4, label='b = 10000')
# plt.plot(T, 1/np.sqrt(T), label='1/sqrt(T)')
# plt.legend()
# plt.grid()
# plt.xlabel('T - number of steps')
# plt.ylabel('RMSE')
# plt.title('Error as function of steps')
# plt.show()

# run as function of time
T = np.arange(0.0001, 0.01, 0.0001)
for t in T:
    print(t*10000)
    # calc rems of mini=batch 1
    a_1 = PSGD_poly(x, y, 1, T=10000, time_stop=float(t))
    h_1 = opt_functions.calculate_poly(x, a_1)
    e_1.append(opt_functions.calc_rmse(h_1, h_star))
    # calc rems of mini=batch 10
    a_2 = PSGD_poly(x, y, 1, T=10000, time_stop=t)
    h_2 = opt_functions.calculate_poly(x, a_2)
    e_2.append(opt_functions.calc_rmse(h_2, h_star))
    # calc rems of mini=batch 100
    a_3 = PSGD_poly(x, y, 1, T=10000, time_stop=t)
    h_3 = opt_functions.calculate_poly(x, a_3)
    e_3.append(opt_functions.calc_rmse(h_3, h_star))
    # calc rems of mini=batch 10000
    a_4 = PSGD_poly(x, y, 1, T=10000, time_stop=t)
    h_4 = opt_functions.calculate_poly(x, a_4)
    e_4.append(opt_functions.calc_rmse(h_4, h_star))
plt.xscale('log')
plt.plot(T, e_1, label='b = 1')
plt.plot(T, e_2, label='b = 10')
plt.plot(T, e_3, label='b = 100')
plt.plot(T, e_4, label='b = 10000')
plt.legend()
plt.grid()
plt.xlabel('time [sec]')
plt.ylabel('RMSE')
plt.title('Error as function of time')
plt.show()
