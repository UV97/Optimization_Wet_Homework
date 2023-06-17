import numpy as np
from matplotlib import pyplot as plt
import opt_functions


# ---- parameters ----
n = 3
a = [0, 1, 0.5, -2]
m = 10000
sigma = 0.5

# ---- define x and f(x) ----
rand_gen = np.random.RandomState(0)  # generate a random generator with a fixed seed
x = 2 * rand_gen.random_sample([m]) - 1
f_x = opt_functions.calculate_poly(x, a)
rand_gen = np.random.RandomState(1)  # generate a random generator with another fixed seed
y = f_x + rand_gen.normal(0, sigma, [m])


# Q3
def estimation_solve(x, y):
    X = opt_functions.calc_x_matrix(x)
    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y


# Q4
opt_functions.plot_poly(a, -1, 1, 'f(x)')
plt.scatter(x, y, c='red', s=0.1, label='y')
a_hat = estimation_solve(x, y)
print(['a0', 'a1', 'a2', 'a3'])
print(a_hat)
y_hat = opt_functions.calculate_poly(x, a_hat)
opt_functions.plot_poly(a_hat, -1, 1, 'f_hat(x)')
plt.legend()
plt.xlabel('x')
plt.title('Part 1')
plt.grid()
plt.show()
