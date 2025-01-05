import random
import numpy as np
import pandas as pd
from scipy.stats import norm
try:
    from Utils import Utils
except ImportError:
    import sys
    sys.path.append('e:/Repositories/Monte-Carlo-American-Pricer-Greeks/path_to_utils_directory')
    from Utils import Utils
from models_params import models_params_dict
# pylint: disable=invalid-name

if not hasattr(Utils, 'init_params'):
    raise ImportError("The module 'Utils' does not have the 'init_params' function.")


def european_monte_carlo_with_greeks(**kwargs):
    """
    Standard Monte Carlo with Greeks

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        sigma : Float
            Implied Volatility.  The default is 0.2 (20%).
        simulations : Int
            Number of Monte Carlo runs. The default is 10000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        output_flag : Str
            Whether to return 'price', 'delta', 'gamma', 'theta',
            'vega' or 'all'. The default is 'price'.
        refresh : Bool
            Whether to refresh the parameters using the 'init_params' function from Utils. The default is False.
        default : Bool
            Whether the function is being called directly (in which
            case values that are not supplied are set to default
            values) or called from another function where they have
            already been updated.

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'vega' : Float; Option Vega
                'all' : Dict; Option Price, Option Delta, Option
                               Gamma, Option Theta, Option Vega

        """

    # Update pricing input parameters to default if not supplied
    if 'refresh' in kwargs and kwargs['refresh']:
        params = Utils.init_params(kwargs)
        S = params['S']
        K = params['K']
        T = params['T']
        r = params['r']
        q = params['q']
        sigma = params['sigma']
        simulations = params['simulations']
        option = params['option']
    else:
        S = kwargs.get('S', 100)
        K = kwargs.get('K', 100)
        T = kwargs.get('T', 0.25)
        r = kwargs.get('r', 0.005)
        q = kwargs.get('q', 0)
        sigma = kwargs.get('sigma', 0.2)
        option = kwargs.get('option', 'call')
        
    if option == 'call':
        z = 1
    else:
        z = -1

    b = r - q
    Drift = (b - (sigma ** 2) / 2) * T
    sigmarT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d1_cdf = norm.cdf(d1)
    d1_term = (2 * (r - q) * T - sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))

    output = [0] * 12
    val = 0
    deltasum = 0
    gammasum = 0
    counter = 0

    simulations = kwargs.get('simulations', 10000)

    for _ in range(simulations):
        St = S * np.exp(
            Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
        val = val + max(z * (St - K), 0)
        if z == 1 and St > K:
            deltasum = deltasum + St
        if z == -1 and St < K:
            deltasum = deltasum + St
        if abs(St - K) < 2:
            gammasum = gammasum + 1
        counter += 1

    # Option Value
    output[0] = np.exp(-r * T) * val / simulations

    # Delta
    output[1] = np.exp(-r * T) * deltasum / (simulations * S)

    # Gamma
    output[2] = (np.exp(-r * T) * ((K / S) ** 2)
                 * gammasum / (4 * simulations))

    # Theta
    output[3] = ((r * output[0]
                  - b * S * output[1]
                  - (0.5 * (sigma ** 2) * (S ** 2) * output[2]))
                 / 365)

    # Vega
    output[4] = output[2] * sigma * (S ** 2) * T / 100

    # Vanna
    output[5] = output[4] * (1 - (np.log(S / K) / sigma ** 2)) / S

    # Vomma
    output[6] = output[4] * (1 - (np.log(S / K) / sigma ** 2)) / sigma

    # Rho
    output[7] = -np.exp(-r * T) * (norm.cdf(d1) * (K * T) / 100)

    # Charm
    output[8] = -np.exp(-r * T) * (norm.cdf(d1) * (2 * (r - q) * T - sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)) - (2 * b * T - sigma * np.sqrt(T)) * norm.cdf(d1) / (2 * T))

    # Color
    output[10] = -np.exp(-r * T) * (norm.cdf(d1) * (2 * (r - q) * T - sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))

    # Volga
    output[9] = output[2] * sigma * np.sqrt(T)

    # Speed
    output[11] = -np.exp(-r * T) * (norm.cdf(d1) * (2 * (r - q) * T - sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))

    output_flag = kwargs.get('output_flag', 'price')

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
        result = {output_flag: output[{'price': 0, 'delta': 1, 'gamma': 2, 'theta': 3, 'vega': 4, 'vanna': 5, 'vomma': 6, 'rho': 7, 'charm': 8, 'volga': 9, 'color': 10, 'speed': 11}[output_flag]]}

    # Print the result dictionary directly
    print(result)
    return result