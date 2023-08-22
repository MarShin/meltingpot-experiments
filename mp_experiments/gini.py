import numpy as np

# define function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        print(xi)
        print(x[i:])
        print(np.abs(xi - x[i:]))
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


# def gini2(x):
#     total = 0
#     for i, xi in enumerate(x[:-1], 1):
#         total += np.sum(np.abs(xi - x[i:]))
#     return total / (2 * len(x) * np.sum(x))


# incomes = np.array([50, 50, 70, 70, 70, 90, 150, 150, 150, 150])
incomes = np.array([10, 10, -70, 70, -70, 70, 70, 150])

# calculate Gini coefficient for array of incomes
print(gini(incomes))
# print(gini2(incomes))
