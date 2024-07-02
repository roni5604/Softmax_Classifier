# Softmax Classifier Implementation

This task involves implementing a Softmax classifier for the CIFAR-10 dataset. The goal is to build a fully-vectorized loss function and its gradient, check the implementation with numerical gradient, use a validation set to tune hyperparameters, optimize the classifier using Stochastic Gradient Descent (SGD), and visualize the final learned weights.

## Overview

The Softmax classifier is a generalization of the logistic regression classifier that can handle multiple classes. It is useful for classification problems with more than two classes. The Softmax function converts raw scores from the linear classifier into probabilities, which are then used to compute the loss and gradient for the optimization process.

## Steps

### 1. Loading and Preprocessing Data

We start by loading the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into training, validation, test, and development sets.

```python
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    # ... (code as provided)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
```

### 2. Implementing the Softmax Loss Function

We need to implement the naive and vectorized versions of the Softmax loss function and its gradient. The naive implementation uses loops, while the vectorized implementation uses matrix operations for efficiency.

#### Naive Implementation

In the naive implementation, we iterate over each training example and compute the loss and gradient manually.

```python
def softmax_loss_naive(W, X, y, reg):
    # ... (code as provided)
    return loss, dW
```

#### Vectorized Implementation

The vectorized implementation leverages NumPy operations to eliminate explicit loops, making the code faster and more efficient.

```python
def softmax_loss_vectorized(W, X, y, reg):
    # ... (code as provided)
    return loss, dW
```

### 3. Checking Implementation with Numerical Gradient

To ensure our implementation is correct, we compare the analytical gradient with the numerical gradient using the provided `grad_check_sparse` function.

```python
from CV7062610.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
```

### 4. Tuning Hyperparameters Using the Validation Set

We experiment with different learning rates and regularization strengths to find the best hyperparameters that maximize validation accuracy.

```python
learning_rates = [1e-7, 5e-7, 1e-6]
regularization_strengths = [2.5e4, 5e4, 7.5e4]

results = {}
best_val = -1
best_softmax = None

for lr in learning_rates:
    for reg in regularization_strengths:
        softmax = Softmax()
        softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=True)
        
        y_train_pred = softmax.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        
        y_val_pred = softmax.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax
```

### 5. Optimizing the Loss Function with SGD

The optimization is done using SGD, iterating over the training data and updating the weights to minimize the loss.

### 6. Visualizing the Learned Weights

Finally, we visualize the learned weights for each class to understand what the classifier has learned.

```python
w = best_softmax.W[:-1, :]  # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()
```

## Inline Questions

### Inline Question 1

**Why do we expect our loss to be close to -log(0.1)? Explain briefly.**

With random weights, the model's predictions are roughly uniform across all classes. For a classification problem with \( C \) classes, the probability for the correct class is approximately \( \frac{1}{C} \). For CIFAR-10, \( C = 10 \), so the initial loss should be close to \( -\log(\frac{1}{10}) = \log(10) \approx 2.3 \).

## To-Do Parts

1. **Naive Softmax Loss and Gradient:**
   - Implement `softmax_loss_naive` in `CV7062610/classifiers/softmax.py`.

2. **Vectorized Softmax Loss and Gradient:**
   - Implement `softmax_loss_vectorized` in `CV7062610/classifiers/softmax.py`.

3. **Gradient Checking:**
   - Use `grad_check_sparse` to ensure the gradients are correct.

4. **Hyperparameter Tuning:**
   - Experiment with different learning rates and regularization strengths to find the best hyperparameters.

5. **SGD Optimization:**
   - Optimize the classifier using SGD.

6. **Weight Visualization:**
   - Visualize the learned weights for each class.

## Conclusion

This exercise will help you understand the implementation details of the Softmax classifier, including loss and gradient computation, optimization, and hyperparameter tuning. By completing this task, you will gain a deeper understanding of how classification models are trained and evaluated.
