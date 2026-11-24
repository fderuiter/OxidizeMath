Pre-allocated equatorial Schwarzschild geodesic and Newtonian comparison bundle (`<= 64` rays) integrating
$$\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta}(x)\frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda} = 0, \qquad \frac{d^2 \mathbf{x}_{\text{N}}}{dt^2} = -\frac{R_s}{2|\mathbf{x}_{\text{N}}|^3}\mathbf{x}_{\text{N}}$$
via fourth-order Runge-Kutta over closed-form [`pure_math::pure_math::tensor::SchwarzschildConnection4D`] Christoffel symbols.

Reference: Misner, Thorne, & Wheeler (1973), *Gravitation*, [`misner1973gravitation`](../../../papers/references.bib#misner1973gravitation);
Schwarzschild (1916), [`schwarzschild1916`](../../../papers/references.bib#schwarzschild1916); [`docs/math_specs/domain_physics.md`](../domain_physics.md).

# Stability
Fourth-order RK4 achieves $O(\Delta\lambda^4)$ Killing-energy drift outside horizon capture ($r \le 1.02\,R_s$) and escape ($r \ge r_{\text{escape}}$) guards, clamping Newtonian radial denominators at $r_{\min} = 0.25\,R_s$.

# Domain of Validity
Requires strictly positive finite Schwarzschild radius $R_s > 0$ (`HighEnergyError::InvalidMass`), equatorial initial states ($\theta = \pi/2, u^\theta = 0$) with $r_0 > R_s$, and finite affine step `d_lambda > 0`.
