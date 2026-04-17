import numpy as np
from copy import deepcopy

class GreeksCalculator:
    """
    Calculates numerical Greeks for an Autocall product using finite differences
    and Common Random Numbers (CRN) for maximum stability.
    """
    def __init__(self, simulator_class, product, yield_curve, spots, vol_surfaces, corr_matrix, divs):
        self.simulator_class = simulator_class
        self.product = product
        self.yield_curve = yield_curve
        self.spots = np.array(spots, dtype=float)
        self.vol_surfaces = vol_surfaces # List of VolatilitySurface
        self.corr_matrix = np.array(corr_matrix, dtype=float)
        self.divs = np.array(divs, dtype=float)

    def calculate_greeks(self, obs_times, num_paths, seed=42):
        """
        Compute Delta, Gamma, Vega, Theta, Rho using FD + CRN.
        """
        base_price = self._price_at(self.spots, self.vol_surfaces, self.yield_curve, obs_times, num_paths, seed)
        
        # --- DELTA & GAMMA (Shock Spot by +/- 1%) ---
        ds = 0.01 * self.spots
        p_plus = self._price_at(self.spots + ds, self.vol_surfaces, self.yield_curve, obs_times, num_paths, seed)
        p_minus = self._price_at(self.spots - ds, self.vol_surfaces, self.yield_curve, obs_times, num_paths, seed)
        
        delta = (p_plus - p_minus) / (2 * ds)
        gamma = (p_plus - 2 * base_price + p_minus) / (ds**2)
        
        # --- VEGA (Shock ATM Vol Curve by +1% absolute) ---
        dv = 0.01
        
        v_plus = []
        v_minus = []
        for v in self.vol_surfaces:
            v_p = deepcopy(v)
            v_m = deepcopy(v)
            v_p.atm_vols += dv
            v_m.atm_vols -= dv
            v_p._update_interpolation() # Ensure interpolation is refreshed
            v_m._update_interpolation()
            v_plus.append(v_p)
            v_minus.append(v_m)
            
        p_vol_plus = self._price_at(self.spots, v_plus, self.yield_curve, obs_times, num_paths, seed)
        p_vol_minus = self._price_at(self.spots, v_minus, self.yield_curve, obs_times, num_paths, seed)
        vega = (p_vol_plus - p_vol_minus) / (2 * dv)
        
        # --- RHO (Shock Rates by +10 bps) ---
        dr = 0.001 
        shifted_yc = deepcopy(self.yield_curve)
        shifted_yc.rates += dr
        shifted_yc._update_interpolation()
        p_rho = self._price_at(self.spots, self.vol_surfaces, shifted_yc, obs_times, num_paths, seed)
        rho = (p_rho - base_price) / dr
        
        # --- THETA (Shock Time by 1 day) ---
        dt = 1/365.0
        # Only shock if time remains
        if obs_times[0] > dt:
            shifted_times = obs_times - dt
            p_theta = self._price_at(self.spots, self.vol_surfaces, self.yield_curve, shifted_times, num_paths, seed)
            theta = (p_theta - base_price) / (dt * 365.0) # Daily Theta
        else:
            theta = 0.0
        
        return {
            "delta": float(delta[0] if hasattr(delta, "__len__") else delta),
            "gamma": float(gamma[0] if hasattr(gamma, "__len__") else gamma),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
            "base_price": float(base_price)
        }

    def calculate_profiles(self, obs_times, spot_range, num_paths=2000, seed=42):
        """
        Calculates a range of Greeks across a spot price spectrum.
        """
        profiles = {"spot": [], "delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
        
        for s in spot_range:
            # Temporarily override spot
            old_spots = self.spots
            self.spots = np.array([s])
            g = self.calculate_greeks(obs_times, num_paths, seed)
            self.spots = old_spots # Restore
            
            profiles["spot"].append(s)
            profiles["delta"].append(g["delta"])
            profiles["gamma"].append(g["gamma"])
            profiles["vega"].append(g["vega"])
            profiles["theta"].append(g["theta"])
            profiles["rho"].append(g["rho"])
            
        return profiles

    def _price_at(self, s, v_surfaces, yc, times, n, seed):
        sim = self.simulator_class(s, v_surfaces, self.corr_matrix, yc, self.divs)
        paths = sim.generate_paths(times, num_paths=n, seed=seed)
        res = self.product.price(paths, s, yc)
        return res['fair_value']
