import numpy as np
from scipy.special import gammaln, gammainc
from scipy.integrate import quad
from scipy.optimize import      

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

def P_n(k_limit, interval, omega, rate, n):
    sum_k = 0
    for k in range(1, k_limit + 1):
        # erlang_pdf_res, error = quad(erlang_pdf, lower_bound, upper_bound, args=(k, rate))
        erlang_pdf_res = erlang_k_interval_probability(k, rate, n * interval, omega/2)
        sum_k +=  erlang_pdf_res
    return sum_k

def analytical_latency_result(n_limit, k_limit, interval, omega, rate):
    P_n = []
    
    for n in range(1, n_limit + 1):
        lower_bound = n * interval - omega/2
        upper_bound = n * interval + omega/2

        sum_k = P_n(k_limit, interval, omega, rate, n) 

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

# Define the energy constraint function
def energy_constraint(params, E_t, E_budget, n_max=100, k_max=100):
    L, omega, lambd = params
    energy_used = E_t * sum(n * L * P(n, L, omega, lambd, k_max) for n in range(1, n_max + 1))
    return E_budget - energy_used


def minimize_latency():
    # Parameters and bounds
    E_t = 1.0 
    E_budget = 100.0
    L_bounds = (0.1, 10.0)
    omega_bounds = (0.01, 1.0)
    lambda_bounds = (0.01, 1.0)

    # Define the constraints in the form required by scipy.optimize.minimize
    constraints = ({
        'type': 'ineq',
        'fun': energy_constraint,
        'args': (E_t, E_budget)
    })

    initial_guess = [1.0, 0.1, 0.5]
    # Define the bounds in the form required by scipy.optimize.minimize
    bounds = [L_bounds, omega_bounds, lambda_bounds]
    
    result = minimize(analytical_latency_result, 
                      x0=initial_guess, 
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
