"""
SABR Volatility calibration

"""

import copy
import math
import numpy as np
import MC
import copy
import time
from typing import Callable
from functools import wraps
from models_params import models_params_dict, sabr_params_dict
import random
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from models_params import models_params_dict
import MC as MC

# Define sabr_params_dict
sabr_params_dict = {
    'F': 100,
    'K': 70,
    'T': 0.5,
    'r': 0.05,
    'atmvol': 0.3,
    'beta': 0.9999,
    'volvol': 0.5,
    'rho': -0.4,
    'option': 'put',
    'timing': False,
    'output_flag': 'price'
}

# Define params using models_params_dict
params = models_params_dict.copy()
params.update(sabr_params_dict)

class SABRVolatility:
    """
    Stochastic, Alpha, Beta, Rho model

    Extension of Black 76 model to include an easily implementable
    stochastic volatility model

    Beta will typically be chosen a priori according to how traders
    observe market prices:
        e.g. In FX markets, standard to assume lognormal terms, Beta = 1
             In some Fixed Income markets traders prefer to assume
             normal terms, Beta = 0

    Alpha will need to be calibrated to ATM volatility

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
    default : Bool
        Whether the function is being called directly (in which
        case values that are not supplied are set to default
        values) or called from another function where they have
        already been updated.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @staticmethod
    def calibrate(F, T, atmvol, beta, volvol, rho):
        """
        Calibrate the SABR model parameters.

        Parameters
        ----------
        F : Float
            Forward price.
        T : Float
            Time to maturity.
        atmvol : Float
            At-the-money volatility.
        beta : Float
            Beta parameter.
        volvol : Float
            Volatility of volatility.
        rho : Float
            Correlation between the asset price and its volatility.

        Returns
        -------
        alpha : Float
            Calibrated alpha parameter.
        """
        # Example calibration logic
        alpha = SABRVolatility._find_alpha(F, T, atmvol, beta, volvol, rho)
        return alpha

    @classmethod
    def _find_alpha(cls, F, T, atmvol, beta, volvol, rho):
        """
        Find alpha feeding values to _cube_root method.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        # Alpha is a function of atm vol etc
        alpha = cls._cube_root(
            ((1 - beta) ** 2 * T / (24 * F ** (2 - 2 * beta))),
            (0.25 * rho * volvol * beta * T / (F ** (1 - beta))),
            (1 + (2 - 3 * rho ** 2) / 24 * volvol ** 2 * T),
            (-atmvol * F ** (1 - beta))
        )
        return alpha

    @classmethod
    def _cube_root(cls, cubic, quadratic, linear, constant):
        """
        Finds the smallest positive root of the input cubic polynomial
        algorithm from Numerical Recipes

        Parameters
        ----------
        cubic : Float
            3rd order term of input polynomial.
        quadratic : Float
            2nd order term of input polynomial.
        linear : Float
            Linear term of input polynomial.
        constant : Float
            Constant term of input polynomial.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        a = quadratic / cubic
        b = linear / cubic
        C = constant / cubic
        Q = (a ** 2 - 3 * b) / 9
        r = (2 * a ** 3 - 9 * a * b + 27 * C) / 54
        D = Q ** 3 - r ** 2

        if D >= 0:
            theta = np.arccos(r / Q ** 1.5)
            root1 = -2 * Q ** 0.5 * np.cos(theta / 3) - a / 3
            root2 = -2 * Q ** 0.5 * np.cos((theta + 2 * np.pi) / 3) - a / 3
            root3 = -2 * Q ** 0.5 * np.cos((theta - 2 * np.pi) / 3) - a / 3
            roots = [root1, root2, root3]
        else:
            A = -np.sign(r) * (np.abs(r) + np.sqrt(-D)) ** (1 / 3)
            B = Q / A
            root1 = (A + B) - a / 3
            roots = [root1]

        positive_roots = [root for root in roots if root > 0]
        if not positive_roots:
            return None
        return min(positive_roots)

# Example usage of the calibrate function
alpha = SABRVolatility.calibrate(
    F=params['F'],
    T=params['T'],
    atmvol=params['atmvol'],
    beta=params['beta'],
    volvol=params['volvol'],
    rho=params['rho']
)
print(f"Calibrated alpha: {alpha}")

# Create an instance of SABRVolatility using params
sabr = SABRVolatility(**params)

# Example usage of the instance
print(sabr.params)

class Utils():
    """
    Utility functions for refreshing parameters and timing

    """
    @staticmethod
    def timer(func: Callable) -> Callable:
        """
        Add timing to a function

        Parameters
        ----------
        func : Function
            The function to be timed.

        Returns
        -------
        Function
            The function with runtime.

        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            function = func(*args, **kwargs)
            end = time.perf_counter()
            for key, value in kwargs.items():
                if key == 'timing' and bool(value):
                    print('{}.{} : {} milliseconds'.format(
                        func.__module__, func.__name__, round((
                            end - start)*1e3, 2)))
            return function
        return wrapper
    
    @staticmethod
    def init_sabr_params(inputs: dict) -> dict:
        """
        Initialise parameter dictionary
        Parameters
        ----------
        inputs : Dict
            Dictionary of parameters supplied to the function.
        Returns
        -------
        params : Dict
            Dictionary of parameters.
        """
        # Copy the default parameters
        params = copy.deepcopy(sabr_params_dict)

        # For all the supplied arguments
        for key, value in inputs.items():

            # Replace the default parameter with that provided
            params[key] = value

        return params

    def __init__(self, **kwargs):

        # Import dictionary of default parameters
        self.default_dict = copy.deepcopy(sabr_params_dict)

        # Store initial inputs
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value

        # Initialise system parameters
        params = Utils.init_sabr_params(inputs)

        self.params = params


    def calibrate(self, **kwargs):
        """
        Run the SABR calibration

        Returns
        -------
        Float
            Black-76 equivalent SABR volatility and price.

        """
        if 'refresh' not in kwargs:
            kwargs['refresh'] = True

        # Update pricing input parameters to default if not supplied
        if 'refresh' in kwargs and kwargs['refresh']:
            params = Utils.init_sabr_params(kwargs)
            F = params['F']
            K = params['K']
            T = params['T']
            r = params['r']
            atmvol = params['atmvol']
            beta = params['beta']
            volvol = params['volvol']
            rho = params['rho']
            output_flag = params['output_flag']
            option = params['option']

        black_vol = self._alpha_sabr(
            F, K, T, beta, volvol, rho, self._find_alpha(
                F=F, T=T, atmvol=atmvol, beta=beta, volvol=volvol, rho=rho,))

        black_price = MC.european_monte_carlo_with_greeks(
            F=F, K=K, T=T, r=r, sigma=black_vol, option=option, refresh=True)

        output_dict = {
            'vol':black_vol,
            'price':black_price,
            'all':{
                'Price':black_price,
                'Vol':black_vol
                }
            }

        return output_dict.get(output_flag, "Please enter a valid output flag")

#        if output_flag == 'vol':
#            return black_vol

#        elif output_flag == 'price':
#            return black_price

#        elif output_flag == 'all':
#            return {'Price':black_price,
#                    'Vol':black_vol}


    @staticmethod
    def _alpha_sabr(F, K, T, beta, volvol, rho, alpha):
        """
        The SABR skew vol function

        Parameters
        ----------
        Alpha : Float
            Alpha value.

        Returns
        -------
        result : Float
            Black-76 equivalent SABR volatility.

        """

        dSABR = np.zeros(4)
        dSABR[1] = (
            alpha
            / ((F * K) ** ((1 - beta) / 2)
               * (1
                  + (((1 - beta) ** 2) / 24)
                  * (np.log(F / K) ** 2)
                  + ((1 - beta) ** 4 / 1920)
                  * (np.log(F / K) ** 4))))

        if abs(F - K) > 10 ** -8:
            sabrz = (volvol / alpha) * (F * K) ** (
                (1 - beta) / 2) * np.log(F / K)
            y = (np.sqrt(1 - 2 * rho * sabrz + (
                sabrz ** 2)) + sabrz - rho) / (1 - rho)
            if abs(y - 1) < 10 ** -8:
                dSABR[2] = 1
            elif y > 0:
                dSABR[2] = sabrz / np.log(y)
            else:
                dSABR[2] = 1
        else:
            dSABR[2] = 1

        dSABR[3] = (1 + ((((1 - beta) ** 2 / 24) * alpha ** 2 / (
            (F * K) ** (1 - beta))) + (
                0.25 * rho * beta * volvol * alpha) / (
                    (F * K) ** ((1 - beta) / 2)) + (
                        2 - 3 * rho ** 2) * volvol ** 2 / 24) * T)

        result = dSABR[1] * dSABR[2] * dSABR[3]

        return sabr_params_dict


    @classmethod
    def _find_alpha(cls, F, T, atmvol, beta, volvol, rho):
        """
        Find alpha feeding values to _cube_root method.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        # Alpha is a function of atm vol etc

        alpha = cls._cube_root(
            ((1 - beta) ** 2 * T / (24 * F ** (2 - 2 * beta))),
            (0.25 * rho * volvol * beta * T / (F ** (1 - beta))),
            (1 + (2 - 3 * rho ** 2) / 24 * volvol ** 2 * T),
            (-atmvol * F ** (1 - beta)))

        return alpha


    @classmethod
    def _cube_root(cls, cubic, quadratic, linear, constant):
        """
        Finds the smallest positive root of the input cubic polynomial
        algorithm from Numerical Recipes

        Parameters
        ----------
        cubic : Float
            3rd order term of input polynomial.
        quadratic : Float
            2nd order term of input polynomial.
        linear : Float
            Linear term of input polynomial.
        constant : Float
            Constant term of input polynomial.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        a = quadratic / cubic
        b = linear / cubic
        C = constant / cubic
        Q = (a ** 2 - 3 * b) / 9
        r = (2 * a ** 3 - 9 * a * b + 27 * C) / 54
        roots = np.zeros(4)

        if r ** 2 - Q ** 3 >= 0:
            cap_A = -np.sign(r) * (abs(r) + np.sqrt(
                r ** 2 - Q ** 3)) ** (1 / 3)
            if cap_A == 0:
                cap_B = 0
            else:
                cap_B = Q / cap_A
            result = cap_A + cap_B - a / 3
        else:
            theta = cls._arccos(r / Q ** 1.5)

            # The three roots
            roots[1] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3) - a / 3
            roots[2] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3 + 2.0943951023932) - a / 3
            roots[3] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3 - 2.0943951023932) - a / 3

            # locate that one which is the smallest positive root
            # assumes there is such a root (true for SABR model)
            # there is always a small positive root

            if roots[1] > 0:
                result = roots[1]
            elif roots[2] > 0:
                result = roots[2]
            elif roots[3] > 0:
                result = roots[3]

            if roots[2] > 0 and roots[2] < result:
                result = roots[2]

            if roots[3] > 0 and roots[3] < result:
                result = roots[3]

        return sabr_params_dict


    @staticmethod
    def _arccos(y):
        """
        Inverse Cosine method

        Parameters
        ----------
        y : Float
            Input value.

        Returns
        -------
        result : Float
            Arc Cosine of input value.

        """
        result = np.arctan(-y / np.sqrt(-y * y + 1)) + 2 * np.arctan(1)

        return sabr_params_dict

def calibrate_sabr_params():
    sabr_params_dict = {
        'F': 100,
        'K': 70,
        'T': 0.5,
        'r': 0.05,
        'atmvol': 0.3,
        'beta': 0.9999,
        'volvol': 0.5,
        'rho': -0.4,
        'option': 'put',
        'timing': False,
        'output_flag':'price'
    }
    return sabr_params_dict

# Example usage of the function
alpha = SABRVolatility.calibrate(
    F=params['F'],
    T=params['T'],
    atmvol=params['atmvol'],
    beta=params['beta'],
    volvol=params['volvol'],
    rho=params['rho']
)

# Create an instance of SABRVolatility using params
sabr = SABRVolatility(**params)




