import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

class YieldCurve:
    """
    Zero-Coupon Yield Curve with Bootstrapping capabilities.
    """
    def __init__(self, times: np.ndarray = None, rates: np.ndarray = None):
        """
        :param times: 1D array of maturities (years) for the ZC rates.
        :param rates: 1D array of corresponding ZC rates (continuous).
        """
        if times is not None and rates is not None:
            self.times = np.array(times)
            self.rates = np.array(rates)
            self._update_interpolation()
        else:
            # Default to flat 3% if nothing provided
            self.times = np.array([0.0, 30.0])
            self.rates = np.array([0.03, 0.03])
            self._update_interpolation()

    def _update_interpolation(self):
        # Linear interpolation on ZC rates
        self.interp_func = interp1d(self.times, self.rates, kind='linear', fill_value="extrapolate")

    def get_zc_rate(self, t: float) -> float:
        return float(self.interp_func(t))

    def discount_factor(self, t: float) -> float:
        r = self.get_zc_rate(t)
        return np.exp(-r * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            return self.get_zc_rate(t1)
        
        # Continuous forward rate F(t1, t2) = (r2*t2 - r1*t1) / (t2 - t1)
        r1 = self.get_zc_rate(t1)
        r2 = self.get_zc_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    @classmethod
    def bootstrap(cls, instrument_tenors: np.ndarray, market_rates: np.ndarray):
        """
        Simplistic Bootstrap:
        Assuming instruments are:
        - Short term (< 1y): Zero-coupon deposits (Market Rate is ZC rate).
        - Long term (>= 1y): Annual Swaps paying semi-annual or annual coupons.
        """
        times = [0.0]
        rates = [market_rates[0]] # Assume overnight / very short rate starts here
        
        # Placeholder for a more complex bootstrap if needed.
        # For this demonstrator, we'll map tenors directly to ZC rates for simplicity,
        # but the structure is ready for fsolve-based stripping.
        
        return cls(times=instrument_tenors, rates=market_rates)
