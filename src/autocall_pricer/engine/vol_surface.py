import numpy as np
from scipy.interpolate import interp1d

class VolatilitySurface:
    """
    Volatility Surface supporting Term Structure (VTS) and Local Volatility (Skew).
    """
    def __init__(self, tenors: np.ndarray, atm_vols: np.ndarray, skew_intensity: float = 0.0, s0: float = 100.0):
        """
        :param tenors: 1D array of maturities (years).
        :param atm_vols: 1D array of corresponding ATM volatilities (0.20 for 20%).
        :param skew_intensity: Sensitivity of vol to spot changes (e.g. -0.5 means vol increases as spot decreases).
        :param s0: Reference spot price for skew calculation.
        """
        self.tenors = np.array(tenors, dtype=float)
        self.atm_vols = np.array(atm_vols, dtype=float)
        self.skew_intensity = float(skew_intensity)
        self.s0 = float(s0)
        
        # Linear interpolation for ATM Term Structure
        self._update_interpolation()

    def _update_interpolation(self):
        # Flat extrapolation outside bounds avoids crazy negative/huge vols at t=0
        self.vts_func = interp1d(self.tenors, self.atm_vols, kind='linear', fill_value=(self.atm_vols[0], self.atm_vols[-1]), bounds_error=False)

    def __hash__(self):
        return hash((tuple(self.tenors), tuple(self.atm_vols), self.skew_intensity, self.s0))

    def __eq__(self, other):
        if not isinstance(other, VolatilitySurface):
            return False
        return np.array_equal(self.tenors, other.tenors) and \
               np.array_equal(self.atm_vols, other.atm_vols) and \
               self.skew_intensity == other.skew_intensity and \
               self.s0 == other.s0
        
    def get_vol(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Calculate local volatility for time t and spot prices s.
        Formula: sigma(t, S) = ATM_Vol(t) * (1 + skew * ln(S/S0))
        Note: We clip the volatility to be at least 1% to avoid numerical issues.
        
        :param t: Time in years.
        :param s: 1D or 2D array of spot prices.
        :return: Array of volatilities matching s shape.
        """
        # 1. Get ATM Vol for time t
        atm_vol = float(self.vts_func(t))
        
        # 2. Apply Skew adjustment
        # Local Vol approx: sigma = atm_vol * (S/S0)^skew_power
        # Or simpler linear skew: atm_vol + skew * (S-S0)/S0
        # We use the log-linear version for better stability: atm_vol * (1 + skew * ln(S/S0))
        
        rel_spot = s / self.s0
        # Avoid log(0)
        rel_spot = np.maximum(rel_spot, 0.01)
        
        vol_adj = 1.0 + self.skew_intensity * np.log(rel_spot)
        
        # Final Vol
        vol = atm_vol * vol_adj
        
        # Floor at 1% for stability
        return np.maximum(vol, 0.01)

    @classmethod
    def from_flat_vol(cls, vol: float, skew: float = 0.0, s0: float = 100.0):
        return cls(tenors=np.array([0.0, 30.0]), atm_vols=np.array([vol, vol]), skew_intensity=skew, s0=s0)
