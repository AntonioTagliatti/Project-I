import numpy as np
from math import exp, sqrt
from typing import Literal

def binomial_tree_european(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int = 100, 
                           option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Price a European option using the Cox-Ross-Rubinstein (CRR) binomial tree model.
    
    Parameters:
        S0 : float
            Initial stock price.
        K : float
            Strike price.
        T : float
            Time to maturity (years).
        r : float
            Risk-free rate (annualized).
        sigma : float
            Volatility (annualized).
        n_steps : int, optional
            Number of time steps in the tree (default=100).
        option_type : str, optional
            Type of option ('call' or 'put', default='call').
    Returns:
        float
            Option price.
    """
    dt = T / n_steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    discount = exp(-r * dt)
    
    # Asset prices at maturity
    S = S0 * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps + 1))
    
    # Payoff at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Backward induction
    for _ in range(n_steps):
        V = discount * (p * V[:-1] + (1 - p) * V[1:])
    
    return float(V[0])

print(binomial_tree_european(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, n_steps=100, option_type='call'))
#8.0219