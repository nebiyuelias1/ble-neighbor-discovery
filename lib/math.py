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


def analytical_latency_result(n_limit, k_limit, interval, omega, rate):
    P_n = []
    
    for n in range(1, n_limit + 1):
        lower_bound = n * interval - omega/2
        upper_bound = n * interval + omega/2

        sum_k = 0
        for k in range(1, k_limit + 1):
            erlang_pdf_res, error = quad(erlang_pdf, lower_bound, upper_bound, args=(k, rate))
            sum_k += erlang_pdf_res
        P_n.append(sum_k)

    latency = 0
    bernouli_probabilities = []
    for n in range(n_limit):
        time_duration = (n + 1) * interval
        probability_of_no_match = np.prod([1 - P_n[i] for i in range(n)])
        bernouli_prob =  P_n[n] * probability_of_no_match
        bernouli_probabilities.append(bernouli_prob)
        latency += time_duration * bernouli_prob
    print(f'latency: {latency}')
    print(f'sum (P_n): {sum(bernouli_probabilities)}')
            
    return latency

