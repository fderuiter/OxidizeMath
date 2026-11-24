Zero-allocation 2D Langevin `B-A-O-A-B` stochastic momentum integrator splitting the underdamped SDE
$dq = \partial_p K(p)\,dt$, $dp = -\nabla V(q)\,dt - \gamma p\,dt + \sqrt{2 m \gamma k_B T}\,dW_t$
into half-kick (`B`), half-drift (`A`), exact Ornstein-Uhlenbeck thermostat (`O`), half-drift (`A`), and half-kick (`B`).

Reference: Leimkuhler & Matthews (2013), *Appl. Math. Res. Express* 2013(1), DOI: 10.1093/amrx/abs010;
[`docs/math_specs/engine_diffeq.md`](../engine_diffeq.md) and [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Preserves exact configurational Gibbs-Boltzmann variance on harmonic potentials across $0 < h \omega_{\max} < 2$ via unconditionally contractive Ornstein-Uhlenbeck factors $c_1 = e^{-\gamma h} \in (0, 1]$ and `sqrt(m k_B T (1 - c_1^2))`.

# Domain of Validity
Requires finite phase-space coordinates `(q, p)`, finite gradients $\nabla V(q)$, strictly positive `dt > 0` (clamped to `[1e-4, 0.1]`), mass in `[0.05, 50.0]`, damping in `[0.0, 20.0]`, and temperature $k_B T \in [0.0, 50.0]$ (`EvolutionError::InvalidState` otherwise).
