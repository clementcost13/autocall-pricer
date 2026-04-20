import numpy as np

class AutocallAthena:
    """
    Autocall Athena Pitch (Worst-Of, path-dependent coupons, PDI at maturity).
    """
    def __init__(self, obs_times: np.ndarray, autocall_levels: np.ndarray, 
                 coupon_levels: np.ndarray, coupon_rates: np.ndarray, 
                 pdi_barrier: float, nominal: float = 100.0,
                 memory_feature: bool = True):
        """
        :param obs_times: 1D array of observation dates (e.g. [1.0, 2.0, 3.0]).
        :param autocall_levels: 1D array of early redemption triggers (e.g. 1.0 = 100%).
        :param coupon_levels: 1D array of coupon barriers (e.g. 0.80 = 80%).
        :param coupon_rates: 1D array of absolute coupon values or rates per observing period.
        :param pdi_barrier: Put Down-and-In barrier at final maturity (e.g. 0.60 = 60%).
        :param nominal: Notional amount.
        :param memory_feature: True if memory coupon effect applies.
        """
        self.obs_times = np.array(obs_times, dtype=float)
        self.autocall_levels = np.array(autocall_levels, dtype=float)
        self.coupon_levels = np.array(coupon_levels, dtype=float)
        self.coupon_rates = np.array(coupon_rates, dtype=float)
        self.pdi_barrier = pdi_barrier
        self.nominal = nominal
        self.memory_feature = memory_feature
        self.num_obs = len(obs_times)

    def price(self, paths: np.ndarray, initial_spots: np.ndarray, yield_curve) -> dict:
        """
        Prices the Autocall Athena product using simulated paths.
        
        :param paths: 3D array of shape (num_paths, num_assets, num_obs)
        :param initial_spots: 1D array of initial spot prices.
        :param yield_curve: An instance of YieldCurve to discount payoffs.
        :return: Dictionary containing the fair value and other metrics.
        """
        num_paths, num_assets, num_obs = paths.shape
        
        # Calculate performance relative to initial spot
        # performance shape: (num_paths, num_assets, num_obs)
        perf_tensor = paths / initial_spots[np.newaxis, :, np.newaxis]
        
        # Calculate Worst-Of performance for each path at each observation date
        # Calculate Worst-Of performance for each path at each observation date
        # worst_of_perf shape: (num_paths, num_obs)
        worst_of_perf = np.min(perf_tensor, axis=1)
        
        # Initialize cashflow sub-matrices for breakdown
        cf_pure_bond = np.zeros((num_paths, num_obs))  # Always 100 if called or at maturity protected
        cf_coupons = np.zeros((num_paths, num_obs))
        cf_pdi_risk = np.zeros((num_paths, num_obs))   # Negative adjustment at maturity
        
        # Track which paths are still alive
        is_alive = np.ones(num_paths, dtype=bool)
        accumulated_coupons = np.zeros(num_paths)
        exit_times = np.full(num_paths, self.obs_times[-1]) # Default to maturity

        for j in range(num_obs):
            active_mask = is_alive.copy()
            if not np.any(active_mask): break
            
            # Current performance for all paths at this observation
            p = worst_of_perf[:, j]
            
            # 1. Coupon Logic
            c_mask = (p >= self.coupon_levels[j]) & active_mask
            
            if self.memory_feature:
                accumulated_coupons += self.coupon_rates[j] * active_mask
            else:
                accumulated_coupons = np.where(c_mask, self.coupon_rates[j], 0.0)

            # 2. Autocall / Redemption Logic
            is_autocalled = (p >= self.autocall_levels[j]) & active_mask
            is_final_obs = (j == num_obs - 1)
            
            # --- Coupon Payment Rule (Athena: Only pay at redemption) ---
            pay_now = (is_autocalled | is_final_obs) & c_mask
            
            cf_coupons[pay_now, j] += accumulated_coupons[pay_now]
            accumulated_coupons[pay_now] = 0.0
            
            # --- Redemption Logic ---
            if not is_final_obs:
                cf_pure_bond[is_autocalled, j] += self.nominal
                exit_times[is_autocalled] = self.obs_times[j]
                is_alive[is_autocalled] = False
            else:
                # Maturity Payoff
                protected = (p >= self.pdi_barrier) & is_alive
                at_risk = (p < self.pdi_barrier) & is_alive
                
                cf_pure_bond[is_alive, j] += self.nominal # Pure bond part is 100
                # PDI part is the loss: Nominal * (P - 1)
                cf_pdi_risk[at_risk, j] += self.nominal * (p[at_risk] - 1.0)
                
        # Discounting
        dfs = np.array([yield_curve.discount_factor(t) for t in self.obs_times])
        
        pv_pure_bond = np.sum(cf_pure_bond * dfs, axis=1)
        pv_coupons = np.sum(cf_coupons * dfs, axis=1)
        pv_pdi_risk = np.sum(cf_pdi_risk * dfs, axis=1)
        
        path_pvs = pv_pure_bond + pv_coupons + pv_pdi_risk
        
        fair_value = np.mean(path_pvs)
        std_err = np.std(path_pvs) / np.sqrt(num_paths)
        
        # Final analytics computation
        redemption_counts = np.sum(cf_pure_bond[:, :-1] > 0, axis=0)
        prob_call_per_period = redemption_counts / num_paths
        
        # Prob of reaching maturity
        prob_maturity = np.mean(is_alive)
        prob_pdi = np.mean(at_risk) if 'at_risk' in locals() else 0.0
        
        expected_maturity = np.mean(exit_times)

        # Audit: Select representative paths
        audit_paths = []
        
        # 1. Path that was Autocalled around 1 Year
        # Find the observation index closest to 1.0 Year
        t_1yr_idx = np.argmin(np.abs(self.obs_times - 1.0))
        t_1yr = self.obs_times[t_1yr_idx]
        
        autocall_1y = np.where(exit_times == t_1yr)[0]
        if len(autocall_1y) == 0:
            autocall_1y = np.where(exit_times < self.obs_times[-1])[0] # Fallback
            
        if len(autocall_1y) > 0:
            p_idx = int(autocall_1y[0])
            abs_payoff = float(np.sum(cf_pure_bond[p_idx]) + np.sum(cf_coupons[p_idx]) + np.sum(cf_pdi_risk[p_idx]))
            audit_paths.append({
                "path_id": p_idx, 
                "category": "Autocall_1Y",
                "event": f"Early Redemption (T={float(exit_times[p_idx]):.2f}Y)",
                "exit_time": float(exit_times[p_idx]),
                "total_pv": float(path_pvs[p_idx]),
                "absolute_payoff": abs_payoff
            })
            
        # 1b. Path that was Autocalled at Maturity (Max Coupons)
        mat_autocall = np.where((exit_times == self.obs_times[-1]) & (worst_of_perf[:, -1] >= self.autocall_levels[-1]))[0]
        if len(mat_autocall) > 0:
            p_idx = int(mat_autocall[0])
            abs_payoff = float(np.sum(cf_pure_bond[p_idx]) + np.sum(cf_coupons[p_idx]) + np.sum(cf_pdi_risk[p_idx]))
            audit_paths.append({
                "path_id": p_idx, 
                "category": "Autocall_Mat",
                "event": "Maturity Autocall (Max Memory)",
                "exit_time": float(exit_times[p_idx]),
                "total_pv": float(path_pvs[p_idx]),
                "absolute_payoff": abs_payoff
            })
            
        # 2. Path that went to maturity and was protected (True Median = No coupons, full capital)
        # We ensure it didn't trigger a massive end-of-life memory coupon, to avoid 'better than autocall' anomalies
        mat_protected = np.where((exit_times == self.obs_times[-1]) & (cf_pdi_risk[:, -1] == 0) & (cf_coupons[:, -1] == 0))[0]
        if len(mat_protected) == 0:
            # Fallback if no such path exists
            mat_protected = np.where((exit_times == self.obs_times[-1]) & (cf_pdi_risk[:, -1] == 0))[0]
            
        if len(mat_protected) > 0:
            p_idx = int(mat_protected[0])
            abs_payoff = float(np.sum(cf_pure_bond[p_idx]) + np.sum(cf_coupons[p_idx]) + np.sum(cf_pdi_risk[p_idx]))
            audit_paths.append({
                "path_id": p_idx, 
                "category": "Protected",
                "event": "Maturity Protected (Capital Safe, No Return)", 
                "exit_time": float(exit_times[p_idx]),
                "total_pv": float(path_pvs[p_idx]),
                "absolute_payoff": abs_payoff
            })
            
        # 3. Path that suffered a PDI Loss (Target specifically PDI - 10% for pedagogy)
        pdi_all = np.where((exit_times == self.obs_times[-1]) & (cf_pdi_risk[:, -1] < 0))[0]
        if len(pdi_all) > 0:
            # If multi-asset, prioritize a path with high dispersion (Best > Barrier, Worst < Barrier)
            if num_assets > 1:
                best_perf_at_mat = np.max(perf_tensor[pdi_all, :, -1], axis=1)
                high_dispersion_indices = np.where(best_perf_at_mat >= self.pdi_barrier)[0]
                if len(high_dispersion_indices) > 0:
                    pdi_candidate_pool = pdi_all[high_dispersion_indices]
                else:
                    pdi_candidate_pool = pdi_all
            else:
                pdi_candidate_pool = pdi_all

            target_perf = self.pdi_barrier - 0.10
            # Find the path in the candidate set that is closest to our target (PDI - 10%)
            dist_to_target = np.abs(worst_of_perf[pdi_candidate_pool, -1] - target_perf)
            closest_idx = np.argmin(dist_to_target)
            p_idx = int(pdi_candidate_pool[closest_idx])
            
            final_perf = worst_of_perf[p_idx, -1]
            event_str = f"Typical Capital Breach (Perf: {final_perf*100:.1f}%)"
                
            abs_payoff = float(np.sum(cf_pure_bond[p_idx]) + np.sum(cf_coupons[p_idx]) + np.sum(cf_pdi_risk[p_idx]))
            audit_paths.append({
                "path_id": p_idx, 
                "category": "PDI",
                "event": event_str, 
                "exit_time": float(exit_times[p_idx]),
                "total_pv": float(path_pvs[p_idx]),
                "absolute_payoff": abs_payoff
            })


        return {
            "fair_value": fair_value,
            "std_error": std_err,
            "path_pvs": path_pvs,
            "expected_maturity": expected_maturity,
            "call_probs": prob_call_per_period.tolist(),
            "prob_maturity": prob_maturity,
            "prob_pdi": prob_pdi,
            "audit_paths": audit_paths,
            "breakdown": {
                "Pure Bond Part": np.mean(pv_pure_bond),
                "Coupons Part": np.mean(pv_coupons),
                "PDI Risk (Put)": np.mean(pv_pdi_risk)
            }
        }
