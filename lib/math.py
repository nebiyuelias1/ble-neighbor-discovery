import numpy as np
from scipy.special import gammaln, gammainc
from scipy.integrate import quad
from scipy.optimize import minimize

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
    # TODO: Make sure the range is correct, I feel like
    # the range should be [nL - delta/2, nL + delta/2]
    cdf_upper = erlang_k_cdf(k, lam, nL + delta)
    cdf_lower = erlang_k_cdf(k, lam, nL - delta)
    return cdf_upper - cdf_lower

def poisson_pmf(k, nL, lambda_):
    # Use logarithms to compute the PMF for large k
    log_pmf = k * np.log(nL * lambda_) - (nL * lambda_) - gammaln(k + 1)
    return np.exp(log_pmf)

def probability_of_matching_with_beacon_n(k_limit, interval, omega, rate, n):
    sum_k = 0
    for k in range(1, k_limit + 1):
        # erlang_pdf_res, error = quad(erlang_pdf, lower_bound, upper_bound, args=(k, rate))
        erlang_pdf_res = erlang_k_interval_probability(k, rate, n * interval, omega/2)
        sum_k +=  erlang_pdf_res
    return sum_k

def analytical_latency_result(params, n_limit, k_limit):
    interval, omega, rate = params
    
    P_n = []
    
    for n in range(1, n_limit + 1):
        lower_bound = n * interval - omega/2
        upper_bound = n * interval + omega/2

        sum_k = probability_of_matching_with_beacon_n(k_limit, interval, omega, rate, n) 

        P_n.append(sum_k)

    latency = 0
    bernouli_probabilities = []
    for n in range(n_limit):
        time_duration = (n + 1) * interval
        probability_of_no_match = np.prod([1 - P_n[i] for i in range(n)])
        bernouli_prob =  P_n[n] * probability_of_no_match
        bernouli_probabilities.append(bernouli_prob)
        latency += time_duration * bernouli_prob
    # print(f'latency: {latency}')
    # print(f'sum (P_n): {sum(bernouli_probabilities)}')
    
    # Print params if latency is nan
    if np.isnan(latency):
        print('nan found:')
        print(params)
    return latency

def average_energy_consumption(params):
    L, omega, lambda_ = params
    # Constants
    c1 = 1
    c2 = 0.5
    c3 = 20
    c4 = 0.1
    c5 = 0.05
    c6 = 5
    c7 = 0.15
    c8 = 0.25
    c9 = 0.1
    energy_used = (c1 * L + c2 * omega + c3 * lambda_ + 
                      c4 * L**2 + c5 * omega**2 + c6 * lambda_**2 + 
                      c7 * L * omega + c8 * L * lambda_ + c9 * omega * lambda_)
    return energy_used

# Define the energy constraint function
def energy_constraint(params, E_t, E_budget, n_max, k_max):
    L, omega, lambda_ = params
    # energy_used = E_t * int(analytical_latency_result(params, n_max, k_max) / L)
    # print(f'energy used: {energy_used} for params: {params}')
    # return E_budget - energy_used
    
    return E_budget - average_energy_consumption(params)

def minimize_latency(n_limit, k_limit):
    # Parameters and bounds
    E_t = 1.0 
    E_budget = 20.0
    L_bounds = (2.0, 10.0)
    omega_bounds = (0.01, 1.0)
    lambda_bounds = (0.01, 1.0)

    # Define the constraints in the form required by scipy.optimize.minimize
    constraints = ({
        'type': 'ineq',
        'fun': energy_constraint,
        'args': (E_t, E_budget, n_limit, k_limit)
    })

    initial_guess = [1.0, 0.1, 0.5]
    # Define the bounds in the form required by scipy.optimize.minimize
    bounds = [L_bounds, omega_bounds, lambda_bounds]
    
    result = minimize(analytical_latency_result, 
                      x0=initial_guess,
                      args=(n_limit, k_limit),
                      bounds=bounds,
                      constraints=constraints,
                      method='SLSQP')
    
    # Output the results
    if result.success:
        optimal_params = result.x
        print(f"Optimal L: {optimal_params[0]}")
        print(f"Optimal omega: {optimal_params[1]}")
        print(f"Optimal lambda: {optimal_params[2]}")
    else:
        print("Optimization failed:", result.message)
