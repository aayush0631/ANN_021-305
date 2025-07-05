# ANN_021-305
# Perceptron Training for Logic Gates and Linear Regression

This repository contains multiple Python implementations of the Perceptron algorithm, demonstrating its use for:

- Training perceptrons on binary logic gates with 2 inputs (AND, OR).
- Training perceptrons on logic gates with *n* inputs (AND, OR).
- Simple linear regression with multiple features trained by gradient descent.

---

## Contents

1. **Basic Perceptron for 2-input AND/OR Gates**  
   Train a perceptron to classify 2-input AND and OR gates using a step activation function.

2. **Perceptron for n-input AND/OR Gates**  
   Generalized perceptron training supporting any number of inputs.

3. **Linear Regression with 3 Features**  
   Train weights and bias for a linear function with 3 input features using mean squared error.

4. **Linear Regression with Variable Features (4 and 5)**  
   Similar to (3), but supports any number of features with random dataset generation.

---

## How to Use

Run each script independently to see:

- Training progress per epoch.
- Weight and bias updates.
- Final accuracy or mean squared error.

Example:

```bash
python perceptron_basic.py
python perceptron_n_input.py
python linear_regression_3features.py
python linear_regression_variable_features.py
