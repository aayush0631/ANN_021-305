import itertools

def activation(val):
    return 1 if val >= 0 else 0

def make_truth_table(n, logic):
    table = []
    for row in itertools.product([0, 1], repeat=n):
        if logic == "AND":
            result = int(all(row))
        elif logic == "OR":
            result = int(any(row))
        else:
            raise ValueError("Unsupported gate")
        table.append((list(row), result))
    return table

def perceptron_train(n, logic, lr=0.1, epochs=100):
    data = make_truth_table(n, logic)
    weights = [0.0 for _ in range(n)]
    bias = 0.0

    print(f"\n🔧 Training {logic} Gate with {n} inputs")
    print(f"Start => Weights: {weights}, Bias: {bias:.2f}\n")

    for ep in range(1, epochs + 1):
        updated = False
        print(f"Epoch {ep}")
        for inputs, target in data:
            net = sum(w * x for w, x in zip(weights, inputs)) + bias
            pred = activation(net)
            delta = target - pred

            if delta != 0:
                updated = True
                for i in range(n):
                    weights[i] += lr * delta * inputs[i]
                bias += lr * delta

            print(f"In: {inputs} | Target: {target} | Pred: {pred} | W: {weights} | B: {bias:.2f}")

        if not updated:
            print("\n✅ Converged.")
            break
        print()

    score = sum(activation(sum(w * x for w, x in zip(weights, inputs)) + bias) == y for inputs, y in data)
    acc = score / len(data) * 100

    print("\n📘 Final State:")
    print(f"Weights: {weights}")
    print(f"Bias: {bias:.2f}")
    print(f"Accuracy: {acc:.2f}%")
    print("-" * 50)

# Tests
perceptron_train(3, "AND")
perceptron_train(3, "OR")
perceptron_train(4, "AND")
perceptron_train(4, "OR")
