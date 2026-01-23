import numpy as np

def gradient_descent(X, y, learning_rate, num_iterations):
    """
    Implements the gradient descent algorithm for linear regression.

    Args:
        X (np.array): Feature data (independent variable).
        y (np.array): Target data (dependent variable).
        learning_rate (float): The step size for each update.
        num_iterations (int): The number of iterations to run the algorithm.

    Returns:
        tuple: (final slope 'm', final intercept 'c', history of loss values)
    """
    m = 0  # Initial value for slope
    c = 0  # Initial value for intercept
    N = float(len(y)) # Number of data points
    loss_history = []

    for i in range(num_iterations):
        y_pred = m * X + c  # Current prediction
        
        # Calculate gradients (partial derivatives of the loss function)
        d_m = (-2/N) * sum(X * (y - y_pred))
        d_c = (-2/N) * sum(y - y_pred)
        
        # Update parameters using the gradient descent formula
        m = m - learning_rate * d_m
        c = c - learning_rate * d_c
        
        # Optional: Track the loss for monitoring convergence
        loss = np.mean((y - y_pred)**2)
        loss_history.append(loss)
        
    return m, c, loss_history

# --- Example Usage ---
# 1. Generate sample data
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6]) # Expected: close to y = 0.8*x + 1.8

# 2. Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# 3. Run the gradient descent
final_m, final_c, loss_history = gradient_descent(X, y, learning_rate, num_iterations)

print(f"Final slope (m): {final_m:.4f}")
print(f"Final intercept (c): {final_c:.4f}")
print(f"Minimum loss: {loss_history[-1]:.4f}")
