import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import triang

# Parameters
a = 3   # minimum
c = 5   # mode
b = 7   # maximum

# Calculate c for scipy's triang (mode location relative to a-b)
c = (c - a) / (b - a)

# Generate x values
x = np.linspace(a - 1, b + 1, 500)

# Triangular PDF using scipy
pdf = triang.pdf(x, c=c, loc=a, scale=(b - a))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label=f"Triangular PDF (a={a}, c={c}, b={b})", color='teal')
plt.fill_between(x, pdf, alpha=0.2, color='teal')

# Vertical lines for a, m, b
plt.axvline(a, color='gray', linestyle='--', label='a (min)')
plt.axvline(c, color='blue', linestyle='--', label='c (mode)')
plt.axvline(b, color='gray', linestyle='--', label='b (max)')

# Labels and title
plt.title( " Probability Density Function of a Triangular Distribution", fontsize=14)
plt.xlabel("Duration (days)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


def triangular_cdf(x, a, c, b):
    if x <= a:
        return 0
    elif a < x <= c:
        return ((x - a) ** 2) / ((b - a) * (c - a))
    elif c < x < b:
        return 1 - ((b - x) ** 2) / ((b - a) * (b - c))
    else:
        return 1

import random
import math

def triangular_inverse_cdf(U, a, c, b):
    c = (c - a) / (b - a)
    if 0 < U < c:
        return a + math.sqrt(U * (b - a) * (c - a))
    else:
        return b - math.sqrt((1 - U) * (b - a) * (b - c))

# Example usage
U = random.uniform(0, 1)
sample = triangular_inverse_cdf(U, 3, 5, 7)
print("Sample from inverse CDF:", sample)
