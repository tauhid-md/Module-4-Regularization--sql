"""Module 4: Regularized Cost and Gradient - solved notebook ka script version.

Linear aur logistic regression ke liye L2 regularization ke saath
cost function aur gradient (b regularize nahi hota).
"""
import numpy as np

np.set_printoptions(precision=8)


# Sigmoid function for later use
def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g


def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        hx = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (hx - y[i])**2                               #scalar
    cost = cost / (2 * m)                                              #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar

    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar


def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        zx = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        hx = sigmoid(zx)                                          #scalar
        cost +=  -y[i]*np.log(hx) - (1-y[i])*np.log(1-hx)      #scalar

    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar

    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar


def compute_gradient_linear_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw


def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        err_i  = sigmoid(np.dot(X[i],w) + b) - y[i]   #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw


def main():
    # ---- Regularized costs
    np.random.seed(1)
    X_tmp = np.random.rand(5, 6)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,) - 0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    print("Linear   - Regularized cost:",
          compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp))   # ~0.0791723932021427
    print("Logistic - Regularized cost:",
          compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp))  # 0.6850849138741673

    # ---- Regularized gradients
    np.random.seed(1)
    X_tmp = np.random.rand(5, 3)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1])

    dj_db, dj_dw = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    print(f"\nLinear   - dj_db: {dj_db}\nRegularized dj_dw:\n {dj_dw.tolist()}")

    dj_db, dj_dw = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    print(f"\nLogistic - dj_db: {dj_db}\nRegularized dj_dw:\n {dj_dw.tolist()}")


if __name__ == "__main__":
    main()
