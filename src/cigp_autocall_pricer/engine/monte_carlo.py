import numpy as np

class MonteCarloSimulator:
    """
    Monte-Carlo Simulator for correlated assets following Geometric Brownian Motion.
    """
    def __init__(self, spots: np.ndarray, vols: np.ndarray, corr_matrix: np.ndarray, 
                 yield_curve, div_yields: np.ndarray):
        """
        :param spots: 1D array of initial spot prices.
        :param vols: 1D array of volatilities.
        :param corr_matrix: 2D correlation matrix.
        :param yield_curve: An instance of YieldCurve (e.g. flat rate mapping)
        :param div_yields: 1D array of continuous dividend yields.
        """
        self.spots = np.array(spots, dtype=float)
        self.vols = np.array(vols, dtype=float)
        self.corr_matrix = np.array(corr_matrix, dtype=float)
        self.yield_curve = yield_curve
        self.div_yields = np.array(div_yields, dtype=float)
        
        self.num_assets = len(spots)
        assert self.vols.shape == (self.num_assets,)
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
            
            # Draw randoms
            z = rng.standard_normal((self.num_assets, sim_paths))
            
            # Apply correlation
            z = self.cholesky_lb @ z
            
            # Drift & Diffusion
            # (r - q - 0.5 * sigma^2) * dt
            drift = (r - self.div_yields - 0.5 * self.vols**2) * actual_dt
            diffusion = self.vols[:, np.newaxis] * np.sqrt(actual_dt) * z
            
            # Update s_base
            s_base *= np.exp(drift[:, np.newaxis] + diffusion)
            
            if antithetic:
                # Update s_anti with -z
                s_anti *= np.exp(drift[:, np.newaxis] - diffusion)
            
            t_curr = t_next
            
            # Check if we hit an observation date
            while obs_idx < len(obs_times) and abs(t_curr - obs_times[obs_idx]) < 1e-7:
                # Store results
                results[:sim_paths, :, obs_idx] = s_base.T
                if antithetic:
                    results[sim_paths:2*sim_paths, :, obs_idx] = s_anti.T
                obs_idx += 1
                
        return results
