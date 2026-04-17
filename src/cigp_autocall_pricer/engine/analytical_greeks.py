import numpy as np
from scipy.stats import norm

class AnalyticalGreeksCalculator:
    """
    Calculates analytical Greeks by replicating the Autocall structure 
    using Black-Scholes formulas for Digital Options and Vanilla Puts.
    This ensures smooth, stable profiles for risk management.
    """
    def __init__(self, spot, vol, r, q, nominal=100.0):
        self.s = float(spot)
        self.vol = float(vol)
        self.r = float(r)
        self.q = float(q)
        self.nominal = float(nominal)

    def _bs_params(self, s, k, t, vol, r, q):
        if t <= 1e-6: return 0, 0, 0, 0
        d1 = (np.log(s/k) + (r - q + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
        d2 = d1 - vol * np.sqrt(t)
        return d1, d2, norm.pdf(d1), norm.cdf(d1), norm.pdf(d2), norm.cdf(d2)

    def vanilla_put_greeks(self, s, k, t, vol, r, q):
        if t <= 1e-6:
            return {"price": max(k-s, 0.0), "delta": -1.0 if s < k else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        d1, d2, nd1, Nd1, nd2, Nd2 = self._bs_params(s, k, t, vol, r, q)
        
        price = k * np.exp(-r*t) * norm.cdf(-d2) - s * np.exp(-q*t) * norm.cdf(-d1)
        delta = -np.exp(-q*t) * norm.cdf(-d1)
        gamma = np.exp(-q*t) * nd1 / (s * vol * np.sqrt(t))
        vega = s * np.exp(-q*t) * nd1 * np.sqrt(t)
        rho = -k * t * np.exp(-r*t) * norm.cdf(-d2)
        theta = -(s * vol * np.exp(-q*t) * nd1) / (2 * np.sqrt(t)) + q * s * np.exp(-q*t) * norm.cdf(-d1) - r * k * np.exp(-r*t) * norm.cdf(-d2)
        
        return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def digital_call_greeks(self, s, k, t, vol, r, q, payout):
        """Cash-or-Binary Call Greeks"""
        if t <= 1e-6:
            return {"price": payout if s > k else 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        d1, d2, nd1, Nd1, nd2, Nd2 = self._bs_params(s, k, t, vol, r, q)
        
        df = np.exp(-r*t)
        price = payout * df * Nd2
        delta = payout * df * nd2 / (s * vol * np.sqrt(t))
        gamma = -delta * d1 / (s * vol * np.sqrt(t))
        vega = -payout * df * nd2 * d1 / vol
        rho = -payout * t * df * Nd2 + payout * df * nd2 * (np.sqrt(t)/vol - d2/(s*vol*np.sqrt(t))) # Approx for Rho
        # Simplifying Theta for stability in the profile
        theta = -(payout * df * nd2 * d1) / (2 * t)
        
        return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def calculate_autocall_greeks(self, product_params, spot_override=None):
        """
        Sums up replicated Greeks for the Athena structure.
        product_params: dict with obs_times, autocall_barrier, pdi_barrier, coupon_pa, etc.
        """
        s = spot_override if spot_override is not None else self.s
        total_greeks = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        # 1. Bond Part (Principal)
        bond_t = product_params['maturity']
        bond_price = self.nominal * np.exp(-self.r * bond_t)
        total_greeks["price"] += bond_price
        total_greeks["rho"] += -bond_t * bond_price

        # 2. Coupon Parts (Digital Stack)
        # We model each autocall date as a digital call. 
        # Probability of being called is simplified to a single digital for the profile's shape.
        for i, t in enumerate(product_params['obs_times']):
            c_val = product_params['coupon_per_period'] * (i + 1 if product_params['memory'] else 1)
            # Principal repayment (100) at autocall
            payout = self.nominal + c_val
            
            # Autocall Digital
            g = self.digital_call_greeks(s, product_params['autocall_barrier'] * self.s, t, self.vol, self.r, self.q, payout)
            # Subtract previous digital to simulate "only if not called before" (simplified)
            # For the risk profile, the dominant digital is the next one.
            for k in total_greeks:
                total_greeks[k] += g[k]

        # 3. PDI Part (Short Put)
        # We are LONG the bond, but SHORT the PDI risk (Short Put)
        pdi_g = self.vanilla_put_greeks(s, product_params['pdi_barrier'] * self.s, product_params['maturity'], self.vol, self.r, self.q)
        for k in total_greeks:
            if k == "price":
                total_greeks[k] -= pdi_g[k]
            else:
                total_greeks[k] -= pdi_g[k] # Greeks of a short position

        return total_greeks

    def calculate_profiles(self, product_params, spot_range):
        profiles = {"spot": [], "delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
        for s in spot_range:
            g = self.calculate_autocall_greeks(product_params, spot_override=s)
            profiles["spot"].append(s)
            profiles["delta"].append(g["delta"])
            profiles["gamma"].append(g["gamma"])
            profiles["vega"].append(g["vega"])
            profiles["theta"].append(g["theta"])
            profiles["rho"].append(g["rho"])
        return profiles
