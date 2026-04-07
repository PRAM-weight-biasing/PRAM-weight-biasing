import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

class DriftMappingOptimizer():
    def __init__(self,
                 g_min      = 1e-1,
                 g_max      = 25,
                 N          = 500,
                 tau_target = 1e6,
                 nu_amp     = None,
                 distortion = 0.1,
                 sigma_prog = 0.0,
                 sigma_read = 0.0,
                 sigma_nu   = 0.0,
                 ):
        self.g_min = g_min
        self.g_max = g_max
        self.N = N
        self.tau_target = tau_target
        self.nu_amp = nu_amp if nu_amp is not None else self.nu(self.g_max)
        self.distortion = distortion
        self.sigma_prog = sigma_prog
        self.sigma_read = sigma_read
        self.sigma_nu = sigma_nu
        self.beta = 1 / (g_max - g_min)  # differential conductance to weight conversion scale

        self.opt_gp = None
        self.opt_gm = None

    def nu(self, g):
        g_rel = (g - self.g_min) / (self.g_max - self.g_min)
        return -0.0155 * np.log(g_rel ** 1.0 + 0.00762) + 0.0244
        # return -0.0757 * g_rel + 0.1
    
    def drift(self, g: np.ndarray, tau, noisy: bool=True) -> np.ndarray:
        """
        Apply drift from programming to inference:
          g(t_inf) = g * (τ_inf / τ_prog)^(-ν(g) + δ)
        with a small random jitter δ ~ N(0, sigma_nu).
        """
        # add random noise to nu(g)
        nu_noisy = self.apply_nu_noise(g, noisy=noisy)
        return g * (tau ** (-nu_noisy))
    
    def brute_force_mapping(self, n_interp=1000, extrapolate_fit=5):
        """
        Brute‐force search + 1D interpolation along each constant‐W0 line,
        then enforce gp=gm at W0=0 by extrapolation.
        """
        # 1) build the GP/GM grid
        gp = np.linspace(self.g_min, self.g_max, self.N)
        gm = np.linspace(self.g_min, self.g_max, self.N)
        GP, GM = np.meshgrid(gp, gm)

        # 2) initial weights and drift-error
        W0 = self.beta * (GP - GM)
        GP_inf = self.drift(GP, self.tau_target, noisy=False)
        GM_inf = self.drift(GM, self.tau_target, noisy=False)
        err = np.abs(W0 - self.beta * (GP_inf - GM_inf) * self.tau_target ** self.nu_amp)

        # 3) set up interpolator and unique positive W0s
        interp_err = RegularGridInterpolator((gp, gm), err, method='linear',
                                             bounds_error=False, fill_value=np.nan)
        W0_vals = np.unique(W0[W0 >= 0])

        # 4) for each W0, do 1D search along gp−gm=const
        gp_opt = np.zeros_like(W0_vals)
        gm_opt = np.zeros_like(W0_vals)
        for i, w in enumerate(W0_vals):
            delta = w / self.beta
            gm_line = np.linspace(self.g_min, self.g_max - delta, n_interp)
            gp_line = gm_line + delta
            pts = np.vstack((gp_line, gm_line)).T
            errs = interp_err(pts)
            idx = np.nanargmin(errs)
            gp_opt[i], gm_opt[i] = gp_line[idx], gm_line[idx]

        # 5) enforce gp=gm at W0=0 via linear extrapolation
        zero_idx = np.argmin(np.abs(W0_vals))
        pos_idx = np.where(W0_vals > 0)[0][:extrapolate_fit]
        if len(pos_idx) < extrapolate_fit:
            raise RuntimeError(f"Need at least {extrapolate_fit} positive‐W0 points to extrapolate")

        # fit gp vs W0 and gm vs W0
        coeff_gp = np.polyfit(W0_vals[pos_idx], gp_opt[pos_idx], 1)
        coeff_gm = np.polyfit(W0_vals[pos_idx], gm_opt[pos_idx], 1)
        intercept_gp = coeff_gp[1]
        intercept_gm = coeff_gm[1]
        intercept_mean = 0.5 * (intercept_gp + intercept_gm)

        # overwrite the zero-weight entry
        gp_opt[zero_idx] = intercept_mean
        gm_opt[zero_idx] = intercept_mean

        # store and return
        self.opt_gp, self.opt_gm = gp_opt, gm_opt
        return gp_opt, gm_opt

#==================================================================
# compensation analysis methods
#==================================================================
    def eval_compensation(self, 
                          tau_inf_list: np.ndarray,
                          n_iters: int = 1):
        if self.opt_gm is None or self.opt_gp is None:
            raise ValueError("Run brute_force_mapping() first to compute optimal conductances.")
        gp_line = np.asarray(self.opt_gp)
        gm_line = np.asarray(self.opt_gm)

        W0_vals = self.beta * (gp_line - gm_line)
        M = len(W0_vals)
        delta_avgs = np.zeros((len(tau_inf_list), M))

        for ti, tau_inf in enumerate(tau_inf_list):
            delta_sum = np.zeros(M)
            
            for _ in range(n_iters):
                # programming noise
                gp0 = self.apply_programming_noise(gp_line)
                gm0 = self.apply_programming_noise(gm_line)
                
                # apply drift
                gp_inf = self.drift(gp0, tau_inf)
                gm_inf = self.drift(gm0, tau_inf)

                gp_inf = self.apply_read_noise(gp_inf)
                gm_inf = self.apply_read_noise(gm_inf)
                
                # compute inferred weight and add read noise & distortion
                W_inf = self.beta * (gp_inf - gm_inf) * tau_inf ** self.nu_amp
                
                # accumulate absolute delta
                delta_sum += np.abs(W_inf - W0_vals)
            
            # average over iterations
            delta_avgs[ti, :] = delta_sum / n_iters

        return W0_vals, delta_avgs
    
    def analyze_compensation(self, tau_inf_list: np.ndarray, n_iters: int = 1):
        """
        Wraps eval_compensation: plots error vs. W0 for each tau_inf, then
        computes and returns for each:
          1) area under the error curve (AUC)
          2) peak error value *within* the W0 ∈ [0.6, 0.8] range
          3) position of that local peak (W0 at max error in that range)
        
        Returns a pandas DataFrame with columns:
          ['tau_inf', 'auc', 'peak_val', 'peak_pos']
        """
        W0_vals, delta_avgs = self.eval_compensation(tau_inf_list, n_iters)

        # Plotting
        plt.figure(figsize=(6, 4))
        for i, tau in enumerate(tau_inf_list):
            plt.plot(W0_vals, delta_avgs[i],
                     label=fr'$\tau_{{\rm inf}} = {tau:.0e}$')
        plt.xlabel('Initial Weight $W_0$', fontsize=14, fontweight='bold')
        plt.ylabel('$|\Delta W|$', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Weight vs. Drift Error over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        records = []
        for i, tau in enumerate(tau_inf_list):
            y = delta_avgs[i]
            # AUC via trapezoid
            dx = np.diff(W0_vals)
            auc = np.sum((y[:-1] + y[1:]) * dx / 2)

            # Local peak in the specified W0 window
            mask = (W0_vals >= 0.0) & (W0_vals <= 0.9)
            if np.any(mask):
                y_region = y[mask]
                w_region = W0_vals[mask]
                idx_local = np.argmax(y_region)
                peak_val = y_region[idx_local]
                peak_pos = w_region[idx_local]
            else:
                # fallback to global
                peak_val = y.max()
                peak_pos = W0_vals[np.argmax(y)]

            records.append({
                'tau_inf': tau,
                'auc': auc,
                'peak_val': peak_val,
                'peak_pos': peak_pos
            })

        return pd.DataFrame(records)
    
    def eval_compensation_rev(self, 
                          tau_inf_list: np.ndarray,
                          n_iters: int = 1):
        if self.opt_gm is None or self.opt_gp is None:
            raise ValueError("Run brute_force_mapping() first to compute optimal conductances.")
        gp_line = np.asarray(self.opt_gp)
        gm_line = np.asarray(self.opt_gm)

        W0_vals = self.beta * (gp_line - gm_line)
        M = len(W0_vals)
        delta_avgs = np.zeros((len(tau_inf_list), M))

        for ti, tau_inf in enumerate(tau_inf_list):
            delta_sum = np.zeros(M)
            
            for _ in range(n_iters):
                # programming noise
                gp0 = self.apply_programming_noise(gp_line)
                gm0 = self.apply_programming_noise(gm_line)
                
                # apply drift
                gp_inf = self.drift(gp0, tau_inf)
                gm_inf = self.drift(gm0, tau_inf)

                gp_inf = self.apply_read_noise(gp_inf)
                gm_inf = self.apply_read_noise(gm_inf)
                
                # compute inferred weight and add read noise & distortion
                W_inf = self.beta * (gp_inf - gm_inf) * tau_inf ** self.nu_amp
                
                # accumulate (X) absolute delta
                delta_sum += (W_inf - W0_vals)
            
            # average over iterations
            delta_avgs[ti, :] = delta_sum / n_iters

        return W0_vals, delta_avgs
    
    def analyze_compensation_rev(self, tau_inf_list: np.ndarray, n_iters: int = 1):
        """
        Wraps eval_compensation: plots error vs. W0 for each tau_inf, then
        computes and returns for each:
          1) area under the error curve (AUC)
          2) peak error value *within* the W0 ∈ [0.6, 0.8] range
          3) position of that local peak (W0 at max error in that range)
        
        Returns a pandas DataFrame with columns:
          ['tau_inf', 'auc', 'peak_val', 'peak_pos']
        """
        W0_vals, delta_avgs = self.eval_compensation_rev(tau_inf_list, n_iters)

        # Plotting
        plt.figure(figsize=(6, 4))
        for i, tau in enumerate(tau_inf_list):
            plt.plot(W0_vals, delta_avgs[i],
                     label=fr'$\tau_{{\rm inf}} = {tau:.0e}$')
        plt.xlabel('Initial Weight $W_0$')
        plt.ylabel('$\Delta W$')
        plt.title('Weight vs. Drift Error over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        records = []
        for i, tau in enumerate(tau_inf_list):
            y = delta_avgs[i]
            # AUC via trapezoid
            dx = np.diff(W0_vals)
            auc = np.sum((y[:-1] + y[1:]) * dx / 2)

            # Local peak in the specified W0 window
            mask = (W0_vals >= 0.0) & (W0_vals <= 0.9)
            if np.any(mask):
                y_region = y[mask]
                w_region = W0_vals[mask]
                idx_local = np.argmax(y_region)
                peak_val = y_region[idx_local]
                peak_pos = w_region[idx_local]
            else:
                # fallback to global
                peak_val = y.max()
                peak_pos = W0_vals[np.argmax(y)]

            records.append({
                'tau_inf': tau,
                'auc': auc,
                'peak_val': peak_val,
                'peak_pos': peak_pos
            })

        return pd.DataFrame(records)
    
    def save_mapping_to_csv(self, filename: str):
        """
        Save the optimal conductance pairs to a CSV file.
        """
        if self.opt_gp is None or self.opt_gm is None:
            raise ValueError("Run brute_force_mapping() first to compute optimal conductances.")
        
        import pandas as pd
        df = pd.DataFrame({
            'W0': self.beta * (self.opt_gp - self.opt_gm),
            'g_p_opt': self.opt_gp,
            'g_m_opt': self.opt_gm
        })
        df.to_csv(filename, index=False)

    def fit_quadratic_mapping(self):
        """
        Fit quadratic functions to the optimal mapping:
           g_p(W0) ≈ a2*W0^2 + a1*W0 + a0
           g_m(W0) ≈ b2*W0^2 + b1*W0 + b0
        Returns:
            gp_coeffs: array [a2, a1, a0]
            gm_coeffs: array [b2, b1, b0]
        """
        if self.opt_gp is None or self.opt_gm is None:
            raise ValueError("Call brute_force_mapping() before fitting.")

        # compute target weights
        W0 = self.beta * (self.opt_gp - self.opt_gm)

        # fit quadratic models
        gp_coeffs = np.polyfit(W0, self.opt_gp, 2)
        gm_coeffs = np.polyfit(W0, self.opt_gm, 2)

        return gp_coeffs, gm_coeffs


#==================================================================
# noise application methods
#==================================================================
    def apply_programming_noise(self, g: np.ndarray) -> np.ndarray:
        """
        Apply programming noise to conductance g. change equation later.
        """
        return g + np.random.normal(0, self.sigma_prog, size=g.shape)
    
    def apply_read_noise(self, g: np.ndarray) -> np.ndarray:
        """
        Apply read noise to conductance g. change equation later.
        """
        return g + np.random.normal(0, self.sigma_read, size=g.shape)
    
    def apply_nu_noise(self, g: np.ndarray, noisy: bool=True) -> np.ndarray:
        """
        Apply noise to the drift coefficient nu(g).
        """
        nu_nom = self.nu(g)
        if not noisy:
            return nu_nom
        
        noise = np.random.normal(0, self.sigma_nu, size=nu_nom.shape)
        return nu_nom + noise