def activation(value, threshold=0):
    return 1 if value >= threshold else 0

def perceptron_train(gate, samples, alpha=0.1, max_epochs=100):
    weights = [0.0, 0.0]  # weights for x1, x2
    bias = 0.0

    print(f"\n🔧 Training Perceptron for {gate} Gate")
    print(f"Initial weights: w = {weights}, bias = {bias:.2f}")

    for epoch in range(1, max_epochs + 1):
        has_error = False
        print(f"\n📘 Epoch {epoch}")

        for x1, x2, expected in samples:
            net = x1 * weights[0] + x2 * weights[1] + bias
            prediction = activation(net)
            delta = expected - prediction

            if delta != 0:
                weights[0] += alpha * delta * x1
                weights[1] += alpha * delta * x2
                bias += alpha * delta
                has_error = True

            print(f"Input: ({x1}, {x2}) | Target: {expected} | Output: {prediction} | "
                  f"Weights: {weights} | Bias: {bias:.2f}")

        if not has_error:
            print("\n✅ Converged Successfully!")
            break
    else:
        print("\n⚠️ Stopped after maximum epochs without full convergence.")

    # Final summary
    print("\n🧾 Final Parameters")
    print(f"Weights: {weights}")
    print(f"Bias: {bias:.2f}")

    # Evaluate accuracy
    correct = sum(1 for x1, x2, y in samples if activation(x1 * weights[0] + x2 * weights[1] + bias) == y)
    acc = correct / len(samples) * 100
    print(f"🎯 Final Accuracy: {acc:.2f}%")
    print("-" * 50)

# Logic gate truth tables
AND_data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

OR_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]

# Run training
perceptron_train("AND", AND_data)
perceptron_train("OR", OR_data)
