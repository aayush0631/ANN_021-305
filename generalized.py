import numpy as np


def create_data(num_features, num_samples=10):
    """Generate synthetic dataset with known linear relation."""
    np.random.seed(42)
    features = np.random.rand(num_samples, num_features)  # Input matrix
    actual_weights = np.random.uniform(-1, 1, size=num_features)
    true_bias = 5
    targets = np.dot(features, actual_weights) + true_bias
    return features, targets, actual_weights, true_bias


def fit_perceptron(X, y, alpha=0.01, max_epochs=100):
    """Train a simple perceptron using gradient descent for regression."""
    samples, features = X.shape
    w = np.random.randn(features)
    b = np.random.randn()

    for ep in range(1, max_epochs + 1):
        squared_error = 0
        for xi, yi in zip(X, y):
            prediction = np.dot(w, xi) + b
            error = yi - prediction
            squared_error += error ** 2

            # Gradient update
            w += alpha * error * xi
            b += alpha * error

        mse = squared_error / samples
        print("Epoch {:3d} | Loss (MSE): {:.6f}".format(ep, mse))

    return w, b


def run_experiment(n):
    print("\n📊 Training Perceptron for {} features".format(n))
    X, y, true_w, true_b = create_data(n)
    print("🎯 True Weights:", true_w)
    print("🎯 True Bias: {:.4f}".format(true_b))

    learned_w, learned_b = fit_perceptron(X, y)
    print("\n✅ Learned Weights:", learned_w)
    print("✅ Learned Bias: {:.4f}".format(learned_b))


def main():
    run_experiment(4)
    run_experiment(5)


if __name__ == "__main__":
    main()
