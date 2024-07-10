import math
import numpy as np
from scipy.special import gammaln, gammainc
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

def erlang_k_cdf(k, lam, t):
    """Calculate the CDF of the Erlang-k distribution at time t."""
    return gammainc(k, lam * t)

def erlang_k_interval_probability(k, lam, nL, delta):
    """Calculate the probability that the kth event happens in the interval [nL - delta, nL + delta]."""
    cdf_upper = erlang_k_cdf(k, lam, nL + delta)
    cdf_lower = erlang_k_cdf(k, lam, nL - delta)
    return cdf_upper - cdf_lower

def poisson_pmf(k, nL, lambda_):
    # Use logarithms to compute the PMF for large k
    log_pmf = k * np.log(nL * lambda_) - (nL * lambda_) - gammaln(k + 1)
    return np.exp(log_pmf)

def analytical_latency_result(n_limit, k_limit, interval, omega, rate):
    P_n = []
    
    for n in range(1, n_limit + 1):
        lower_bound = n * interval - omega/2
        upper_bound = n * interval + omega/2

        sum_k = 0
        for k in range(1, k_limit + 1):
            # erlang_pdf_res, error = quad(erlang_pdf, lower_bound, upper_bound, args=(k, rate))
            erlang_pdf_res = erlang_k_interval_probability(k, rate, n * interval, omega/2)
            sum_k +=  erlang_pdf_res 

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

