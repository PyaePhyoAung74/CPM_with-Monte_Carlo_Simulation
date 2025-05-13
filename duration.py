import numpy as np

# Example list of durations from simulation
durations = [42.5, 45.1, 43.3, 44.0, 46.2, 41.7, 44.8]  # Example

# Convert to numpy array
durations = np.array(durations)

# Number of simulations
N = len(durations)

# Mean
mu = np.mean(durations)

# Variance
variance = np.var(durations, ddof=1)  # ddof=1 for sample variance

# Standard Deviation
std_dev = np.sqrt(variance)

# Output
print(f"Mean duration (μ): {mu:.2f}")
print(f"Variance (σ²): {variance:.2f}")
print(f"Standard Deviation (σ): {std_dev:.2f}")
