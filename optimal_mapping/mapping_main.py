from mapping_optimizer import DriftMappingOptimizer
import matplotlib.pyplot as plt

opt = DriftMappingOptimizer(
    g_min=1e-1,
    g_max=25,
    N=500,
    tau_target=1e6,
    tau_inf=1e6,
    nu_amp=None,
    distortion=0.1,
    sigma_prog=0.0,
    sigma_read=0.0,
    sigma_nu=0.0,
    )

gp_opt, gm_opt = opt.brute_force_mapping()

W0_vals = opt.beta * (gp_opt - gm_opt)

fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(W0_vals, gp_opt, 'g-', label='$g_p^{opt}$')
ax2.plot(W0_vals, gm_opt, 'b-', label='$g_m^{opt}$')
ax1.set_xlabel('Initial Weight $W_0$')
ax1.set_ylabel('$g_p^{opt}$', color='g')
ax2.set_ylabel('$g_m^{opt}$', color='b')
ax1.set_ylim(0, opt.g_max)
ax2.set_ylim(0, opt.g_max)
plt.title('Weight vs. Optimal Conductance Pair')
ax1.grid(True)
fig.tight_layout()
plt.show()