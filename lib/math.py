import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad

def erlang_pdf(x, k, lambd):
    """
    Compute the Erlang PDF at x using logarithms to avoid overflow.
    
    Parameters:
    x (float): The point at which to evaluate the PDF.
    k (int): The shape parameter of the Erlang distribution.
    lambd (float): The rate parameter of the Erlang distribution.
    
    Returns:
    float: The value of the Erlang PDF at x.
    """
    if x < 0:
        return 0
    log_pdf = (k * np.log(lambd) + (k - 1) * np.log(x) - lambd * x - gammaln(k))
    return np.exp(log_pdf)

def poisson_pmf(k, nL, lambda_):
    return (nL*lambda_)**k * np.exp(-nL*lambda_) / np.math.factorial(k)