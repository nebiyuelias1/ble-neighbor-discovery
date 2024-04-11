import numpy as np

# Define the PDF function
def erlang_pdf(x, k, lambda_):
    return lambda_**k * x**(k-1) * np.exp(-lambda_ * x) / np.math.factorial(k-1)

def poisson_pmf(k, nL, lambda_):
    return (nL*lambda_)**k * np.exp(-nL*lambda_) / np.math.factorial(k)