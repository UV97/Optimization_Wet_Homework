import numpy as np
from matplotlib import pyplot as plt
import opt_functions


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


# Q5 - def a Project Gradient Descent
def PGD_poly(x, y, step_size='const', epsilon=0.001, T=1000, order=3, r=4):
    X = opt_functions.calc_x_matrix(x)
    m = len(x)
    D = 2 * r
    G = D * opt_functions.max_eig(X.transpose() @ X / m) + np.linalg.norm(X.transpose() @ y / m)
    rand_gen = np.random.RandomState(3)  # generate a random generator with a fixed seed
    a0 = rand_gen.normal(0, 1, [order+1])
    t = 0
    g0 = opt_functions.calc_gradient(X, y, a0)
    gradiant_norms = [np.linalg.norm(g0) ** 2]
    while t < T and np.linalg.norm(g0) > epsilon:
        t += 1
        if step_size == 'const':
            eta = 0.1
        elif type(step_size) is not str:
            eta = step_size
        elif step_size == 'decay':
            eta = D / (G * np.sqrt(t))
        elif step_size == 'AdaGrad':
            eta = D / np.sqrt(2 * sum(gradiant_norms))

        a1 = a0 - eta * g0
        if np.linalg.norm(a1) > r:
            a1 = r * a1 / np.linalg.norm(a1)
        a0 = a1
        g0 = opt_functions.calc_gradient(X, y, a0)
        gradiant_norms.append(np.linalg.norm(g0) ** 2)
    return a0


# # Q6
# h_star = opt_functions.calculate_poly(x, opt_functions.estimation_solve(x, y))
# e_decay = []
# e_ada_grad = []
# T = np.arange(10, 100000, 10)
# for t in T:
#     # calc rems of decay step size
#     a_decay = PGD_poly(x, y, step_size='decay', T=t)
#     h_decay = opt_functions.calculate_poly(x, a_decay)
#     e_decay.append(opt_functions.calc_rmse(h_decay, h_star))
#     # calc rems of AdaGrad step size
#     a_ada_grad = PGD_poly(x, y, step_size='AdaGrad', T=t)
#     h_ada_grad = opt_functions.calculate_poly(x, a_ada_grad)
#     e_ada_grad.append(opt_functions.calc_rmse(h_ada_grad, h_star))
# plt.xscale('log')
# plt.plot(T, e_decay, label='Decay step size')
# plt.plot(T, e_ada_grad, label='AdaGrad step size')
# plt.plot(T, 1/np.sqrt(T), label='1/sqrt(T)')
# plt.legend()
# plt.grid()
# plt.xlabel('T - number of steps')
# plt.ylabel('RMSE')
# plt.title('Error as function of steps')
# plt.show()


# Q8
X = opt_functions.calc_x_matrix(x, n+1)
L = opt_functions.max_eig(X.transpose() @ X / m)
meo = opt_functions.min_eig(X.transpose() @ X / m)
h_star = opt_functions.calculate_poly(x, opt_functions.estimation_solve(x, y))
e_1 = []
e_2 = []
e_3 = []
T = np.arange(10, 10000, 10)
for t in T:
    print(t/100)
    # calc rems of 1/10L const step size
    a_1 = PGD_poly(x, y, step_size=1/(10*L), T=t)
    h_1 = opt_functions.calculate_poly(x, a_1)
    e_1.append(opt_functions.calc_rmse(h_1, h_star))
    # calc rems of 1/L const step size
    a_2 = PGD_poly(x, y, step_size=1/L, T=t)
    h_2 = opt_functions.calculate_poly(x, a_2)
    e_2.append(opt_functions.calc_rmse(h_2, h_star))
    # calc rems of 10/L const step size
    a_3 = PGD_poly(x, y, step_size=10/L, T=t)
    h_3 = opt_functions.calculate_poly(x, a_3)
    e_3.append(opt_functions.calc_rmse(h_3, h_star))
plt.xscale('log')
plt.plot(T, e_1, label='1/10L const step size')
plt.plot(T, e_2, label='1/L const step size')
plt.plot(T, e_3, label='10/L const step size')
plt.plot(T, 1/T, label='1/T')
plt.legend()
plt.grid()
plt.xlabel('T - number of steps')
plt.ylabel('RMSE')
plt.title('Error as function of steps')
plt.show()