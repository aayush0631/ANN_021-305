import numpy as np

np.random.seed(42)

def true_fn(x):
    return 2 * x[0] + 3 * x[1] - x[2] + 5

X = np.random.rand(10, 3)
y = np.array([true_fn(x) for x in X])

print("Dataset (x1, x2, x3, y):")
for xi, yi in zip(X, y):
    print(f"{xi[0]:.3f}\t{xi[1]:.3f}\t{xi[2]:.3f}\t{yi:.3f}")

weights = np.random.randn(3)
bias = np.random.randn()

lr = 0.01
epochs = 100

for ep in range(1, epochs + 1):
    total_error = 0
    print(f"\nEpoch {ep}: Weights = {weights}, Bias = {bias:.4f}")

    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        pred = np.dot(weights, xi) + bias
        error = target - pred
        total_error += error**2

        weights += lr * error * xi
        bias += lr * error

        print(f"Sample {i}: x = {xi}, y = {target:.3f}, pred = {pred:.3f}, err = {error:.3f}")
        print(f"         Updated weights = {weights}, bias = {bias:.4f}")

    mse = total_error / len(X)
    print(f"Epoch {ep} MSE: {mse:.6f}")
    if mse < 1e-6:
        print("Stopping early.")
        break

print(f"\nFinal Weights: {weights}, Final Bias: {bias:.4f}")
