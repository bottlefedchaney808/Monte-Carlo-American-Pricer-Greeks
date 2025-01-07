import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Dict

@staticmethod
def init_monte_carlo_with_greeks(**kwargs) -> Union[float, Dict[str, float]]:
    """
    Standard Monte Carlo with Greeks
    Parameters
  **kwargs : Dict #dictionary of parameters supplied to the function.
    S = kwargs.get('S', 100)  # Stock Price. The default is 100.
    K = kwargs.get('K', 100)  # Strike Price. The default is 100.
    T = kwargs.get('T', 0.25) # Time to Maturity. The default is 0.25 (3 Months).
    r = kwargs.get('r', 0.005) # Interest Rate. The default is 0.005 (50bps)
    q = kwargs.get('q', 0) # Dividend Yield. The default is 0.
    sigma = kwargs.get('sigma', 0.2) # Implied Volatility. The default is 0.2 (20%).
    simulations = kwargs.get('simulations', 10000) # Number of Monte Carlo runs. The default is 10000.
    option = kwargs.get('option', 'call') # Type of option. 'put' or 'call'. The default is 'call'.
    output_flag = kwargs.get('output_flag', 'all') # Whether to return 'price', 'delta', 'gamma', 'theta', 'vega', 'vanna', 'vomma', 'rho', 'charm', 'color', 'speed' or 'all'. The default is 'all'.
    default = kwargs.get('default', True) # Whether the function is being called directly (in which case values that are not supplied are set to default values) or called from another function where they have already been updated.

    Returns
    -------
    result : Various
        Depending on output flag:
            'price' : Float; Option Price
            'delta' : Float; Option Delta
            'gamma' : Float; Option Gamma
            'theta' : Float; Option Theta
            'vega' : Float; Option Vega
            'vanna' : Float; Option Vanna
            'vomma' : Float; Option Vomma
            'rho' : Float; Option Rho
            'charm' : Float; Option Charm
            'color' : Float; Option Color
            'color' : Float; Option Color
            'speed' : Float; Option Speed
            'all': Dict; All of the above {default}
    """

def init_monte_carlo_with_greeks_para(
    S: float = 141.37,  # Stock Price. The default is 100.
    K: float = 150,  # Strike Price. The default is 100.
    T: float = 0.2028,  # Time to Maturity. The default is 0.25 (360 days).
    r: float = 0.045,  # Interest Rate. The default is 0.005 (50bps)
    q: float = 0.0,  # Dividend Yield. The default is 0.
    sigma: float = 0.5231,  # Implied Volatility. The default is 0.2 (20%).
    simulations: int = 20000,  # Number of Monte Carlo runs. The default is 10000.
    option: str = 'call',  # Type of option. 'put' or 'call'. The default is 'call'.
    output_flag: str = 'all',  # Whether to return 'price', 'delta', 'gamma', 'theta', 'vega', 'vanna', 'vomma', 'rho', 'charm', 'color', 'speed' or 'all'. The default is 'all'.
    default: bool = True  # Whether the function is being called directly (in which case values that are not supplied are set to default values) or called from another function where they have already been updated.
) -> Union[float, Dict[str, float]]:
    if default:
        if option == 'call':
            z = 1
        else:
            z = -1
        b = r - q
        Drift = (b - (sigma ** 2) / 2) * T
        sigmarT = sigma * np.sqrt(T)
        val = 0
        deltasum = 0
        gammasum = 0
        output = np.zeros((13))
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        random_numbers = norm.ppf(np.random.random(simulations), loc=0, scale=1)
        for i in range(simulations):
            St = S * np.exp(Drift + sigmarT * random_numbers[i])
            val = val + max(z * (St - K), 0)
            if z == 1 and St > K:
                deltasum = deltasum + St
            if z == -1 and St < K:
                deltasum = deltasum + St
            if abs(St - K) < 2:
                gammasum = gammasum + 1

        # Price
        output[0] = np.exp(-r * T) * val / simulations

        # Calculate Option Delta
        output[1] = np.exp(-r * T) * deltasum / (S * simulations)

        # Gamma
        output[2] = (np.exp(-r * T) * gammasum / simulations) / (S * sigma * np.sqrt(T))

        # Theta
        output[3] = (-(S * norm.pdf(d1) * sigma * np.exp(-r * T)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(z * d2) + z * r * S * np.exp(-q * T) * norm.cdf(z * d1)) / 360

        # Vega
        output[4] = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-r * T) / 100

        # Vanna
        output[5] = -np.exp(-r * T) * norm.pdf(d1) * d2 * np.sign(z) / sigma

        # Vomma
        output[6] = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-r * T) * np.sqrt(T) * d1 * d2 / sigma

        # Rho
        output[7] = K * T * np.exp(-r * T) * norm.cdf(z * d2)

        # Charm
        output[8] = np.exp(-r * T) * (norm.pdf(d1) * (2 * (r - b) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma))

        # Color
        output[9] = -np.exp(-r * T) * norm.pdf(d1) * (d2 / sigma) * (2 * (r - b) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma)

        # Volga
        output[10] = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-r * T) * np.sqrt(T) * d1 * d2 / sigma

        # Speed
        output[11] = -np.exp(-r * T) * norm.pdf(d1) * (d1 * d2 / sigma) * (1 - d1 / (sigma * np.sqrt(T))) / (S * sigma * T) 


        output_mapping = {
            'price': 0,
            'delta': 1,
            'gamma': 2,
            'theta': 3,
            'vega': 4,
            'vanna': 5,
            'vomma': 6,
            'rho': 7,
            'charm': 8,
            'volga': 9,
            'color': 10,
            'speed': 11
        }

        if output_flag == 'all':
            result = {
                    'price': output[0],
                    'delta': output[1],
                    'gamma': output[2],
                    'theta': output[3],
                    'vega': output[4],
                    'vanna': output[5],
                    'vomma': output[6],
                    'rho': output[7],
                    'charm': output[8],
                    'volga': output[9],
                    'color': output[10],
                    'speed': output[11]
                }
        else:
            result = output[output_mapping[output_flag]]
        return result
    
    # Add a return statement for when default is False
    else:
        return None
result = init_monte_carlo_with_greeks_para()

# Convert result to a DataFrame for spreadsheet style output
result_df = pd.DataFrame([result])
print(result_df)
