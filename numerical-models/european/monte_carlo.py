import numpy as np
from math import exp, sqrt
from typing import Literal, Optional, Tuple
from scipy.stats import norm

def monte_carlo_european(S0: float, K: float, T: float, r: float, sigma: float,
                          option_type: Literal['call', 'put'] = 'call', 
                          n_sims: int = 100_000,dividend_yield: float = 0.0,
                          antithetic: bool = True,seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Price a European option using Monte Carlo simulation with antithetic variance reduction.
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
        option_type : str, optional
            'call' or 'put' (default='call').
        n_sims : int, optional
            Number of simulations (default=100,000).
        dividend_yield : float, optional
            Continuous dividend yield (default=0.0).
        antithetic : bool, optional
            Use antithetic variates (default=True).
        seed : int, optional
            Random seed for reproducibility.
    
    Returns:
        Tuple[float, float]
            - Option price (float)
            - Standard error (float)
    Price: 8.0219, Std Error: 0.0283
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate terminal stock prices
    if antithetic:
        rand = np.random.standard_normal(n_sims // 2)
        rand = np.concatenate([rand, -rand])  # Antithetic pairs
    else:
        rand = np.random.standard_normal(n_sims)
    
    # Geometric Brownian Motion: S_T = S0 * exp((r - q - 0.5*σ²)T + σ√T * Z)
    drift = (r - dividend_yield - 0.5 * sigma**2) * T
    diffusion = sigma * sqrt(T)
    S_T = S0 * np.exp(drift + diffusion * rand)
    
    # Compute payoffs
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    
    # Discount and compute statistics
    discount = exp(-r * T)
    price = discount * np.mean(payoffs)
    stderr = discount * np.std(payoffs) / sqrt(n_sims)
    
    return price, stderr

price, stderr = monte_carlo_european(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type='call', n_sims=100000)
print(f"Price: {price:.4f}, Std Error: {stderr:.6f}")
#Price: 8.0219, Std Error: 0.0283