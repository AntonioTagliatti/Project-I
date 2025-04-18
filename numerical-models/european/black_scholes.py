import numpy as np
from scipy.stats import norm
from typing import Literal

def black_scholes(S0: float,K: float,T: float,r: float,sigma: float,
                  option_type: Literal['call', 'put'] = 'call',dividend_yield: float = 0.0) -> float:
    """
    Compute the Black-Scholes price of a European option.
    
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
            Type of option ('call' or 'put', default='call').
        dividend_yield : float, optional
            Continuous dividend yield (default=0.0).
    Returns:
        float
            Option price.
    
    Example:
    --------
    >>> black_scholes(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type='call')
    8.0219
    """
    # Calculate d1 and d2
    d1 = (np.log(S0 / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Compute option price
    if option_type == 'call':
        price = S0 * np.exp(-dividend_yield * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend_yield * T) * norm.cdf(-d1)
    
    return float(price)

print(black_scholes(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type='call'))
#8.0219