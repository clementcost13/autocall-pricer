import numpy as np

class MonteCarloSimulator:
    """
    Monte-Carlo Simulator for correlated assets following Geometric Brownian Motion.
    """
    def __init__(self, spots: np.ndarray, vol_surfaces, corr_matrix: np.ndarray, 
                 yield_curve, div_yields: np.ndarray):
        """
        :param spots: 1D array of initial spot prices.
        :param vol_surfaces: List of VolatilitySurface objects (one per asset).
        :param corr_matrix: 2D correlation matrix.
        :param yield_curve: An instance of YieldCurve (e.g. flat rate mapping)
        :param div_yields: 1D array of continuous dividend yields.
        """
        self.spots = np.array(spots, dtype=float)
        self.vol_surfaces = vol_surfaces # List of VolatilitySurface
        self.corr_matrix = np.array(corr_matrix, dtype=float)
        self.yield_curve = yield_curve
        self.div_yields = np.array(div_yields, dtype=float)
        
        self.num_assets = len(spots)
        # Compatibility check: if vol_surfaces is a 1D array of floats, convert to flat surfaces
        from .vol_surface import VolatilitySurface
        if isinstance(self.vol_surfaces, (np.ndarray, list)) and not isinstance(self.vol_surfaces[0], VolatilitySurface):
            self.vol_surfaces = [VolatilitySurface.from_flat_vol(v, s0=s) for v, s in zip(self.vol_surfaces, self.spots)]
            
        assert len(self.vol_surfaces) == self.num_assets
        assert self.corr_matrix.shape == (self.num_assets, self.num_assets)
        assert self.div_yields.shape == (self.num_assets,)
        
        # Cholesky decomposition of the correlation matrix for generating correlated normals
        # We add a tiny epsilon to the diagonal to ensure it is positive definite (handling extreme correlations)
        safe_corr = self.corr_matrix + np.eye(self.num_assets) * 1e-9
        self.cholesky_lb = np.linalg.cholesky(safe_corr)
        
    def generate_paths(self, obs_times: np.ndarray, num_paths: int, seed: int = 42, 
                       antithetic: bool = True, steps_per_year: int = 252) -> np.ndarray:
        """
        Generate scenarios for the underlying assets with high precision.
        
        :param obs_times: Observation dates in years.
        :param num_paths: Number of MC paths.
        :param seed: Random seed.
        :param antithetic: If True, uses antithetic variates for variance reduction.
        :param steps_per_year: Number of discretization steps per year (252 for daily).
        :return: 3D array (num_paths, num_assets, len(obs_times)).
        """
        rng = np.random.default_rng(seed)
        
        # If antithetic, we generate half the paths and mirror them
        sim_paths = num_paths // 2 if antithetic else num_paths
        
        # Create a unified time grid including all obs_times and intermediate steps
        final_t = obs_times[-1]
        dt = 1.0 / steps_per_year
        
        # We need to simulate step by step to capture barriers properly (if applicable)
        # and to improve convergence.
        
        # results will store only values at obs_times
        results = np.zeros((num_paths, self.num_assets, len(obs_times)))
        
        # Current state for paths and their antithetic pairs
        s_base = np.tile(self.spots, (sim_paths, 1)).T # (N_assets, sim_paths)
        if antithetic:
            s_anti = np.tile(self.spots, (sim_paths, 1)).T
            
        t_curr = 0.0
        obs_idx = 0
        
        # Total steps to cover all observation dates
        total_steps = int(np.ceil(final_t * steps_per_year))
        
        for i in range(1, total_steps + 1):
            t_next = min(i * dt, final_t)
            actual_dt = t_next - t_curr
            
            if actual_dt <= 0: continue
            
            r = self.yield_curve.forward_rate(t_curr, t_next)
            
            # --- LOCAL VOLATILITY CALCULATION ---
            # sigma(t, S) for each asset and each path
            sigmas_base = np.zeros((self.num_assets, sim_paths))
            for a in range(self.num_assets):
                sigmas_base[a, :] = self.vol_surfaces[a].get_vol(t_curr, s_base[a, :])
            
            # Draw randoms
            z = rng.standard_normal((self.num_assets, sim_paths))
            
            # Apply correlation
            z = self.cholesky_lb @ z
            
            # Drift & Diffusion (Path-dependent due to LocVol)
            # We align shapes: sigmas_base is (N_assets, sim_paths)
            drift_base = (r - self.div_yields[:, np.newaxis] - 0.5 * sigmas_base**2) * actual_dt
            diffusion_base = sigmas_base * np.sqrt(actual_dt) * z
            
            # Update s_base
            s_base *= np.exp(drift_base + diffusion_base)
            
            if antithetic:
                sigmas_anti = np.zeros((self.num_assets, sim_paths))
                for a in range(self.num_assets):
                    sigmas_anti[a, :] = self.vol_surfaces[a].get_vol(t_curr, s_anti[a, :])
                
                drift_anti = (r - self.div_yields[:, np.newaxis] - 0.5 * sigmas_anti**2) * actual_dt
                diffusion_anti = sigmas_anti * np.sqrt(actual_dt) * z # z is already correlated
                
                # Update s_anti with -z (since Z ~ N(0,1), its antithesis is -Z, thus diffusion is negative relative to the base drift)
                s_anti *= np.exp(drift_anti - diffusion_anti)
            
            t_curr = t_next
            
            # Check if we hit an observation date
            while obs_idx < len(obs_times) and abs(t_curr - obs_times[obs_idx]) < 1e-7:
                # Store results
                results[:sim_paths, :, obs_idx] = s_base.T
                if antithetic:
                    results[sim_paths:2*sim_paths, :, obs_idx] = s_anti.T
                obs_idx += 1
                
        return results
