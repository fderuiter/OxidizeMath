//! Zero-allocation 2D Allen-Cahn (Model A) and Cahn-Hilliard (Model B) phase-field solver
//! using Eyre convex-splitting spectral stepping and [`engine_pde::spde_noise`] staggered conservative noise.

// [cite:phase_field]

use crate::physics::stat_mech::edwards_wilkinson::mean_field;
use engine_pde::{BoundaryCondition, inject_additive_noise_2d, inject_conserved_flux_noise_2d};
use math_commons::math_kernel::types::{SolverWorkspace, fill_standard_normal, flatten_2d_index};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::pde::spde::{
    ConservedBiharmonicParams, RadialSpectrumSummary, ReactionBiharmonicParams,
    SemiImplicitSpectral2D,
};

use super::telemetry::PhaseFieldTelemetry;

/// Lifshitz-Slyozov-Wagner domain coarsening exponent $1/3$ ($k_{\max}(t) \propto t^{-1/3}$) for Model B.
pub const LIFSHITZ_SLYOZOV_EXPONENT: f64 = 1.0 / 3.0;
/// Allen-Cahn curvature-driven coarsening exponent $1/2$ ($k_{\max}(t) \propto t^{-1/2}$) for Model A.
pub const MODEL_A_COARSENING_EXPONENT: f64 = 0.5;

/// Hohenberg-Halperin universality class for 2D stochastic phase-field kinetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseFieldModel {
    /// Non-conserved order parameter relaxation (Model A): $\partial_t \phi = -M \frac{\delta F}{\delta \phi} + \eta$.
    AllenCahn,
    /// Conserved diffusive phase separation (Model B): $\partial_t \phi = M \nabla^2 \frac{\delta F}{\delta \phi} + \nabla \cdot \boldsymbol{\xi}$.
    CahnHilliard,
}

/// Physical and numerical parameters for [`PhaseFieldSolver2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseFieldParams {
    /// Selected Hohenberg-Halperin dynamics (`AllenCahn` vs `CahnHilliard`).
    pub model: PhaseFieldModel,
    /// Bath quench temperature $T \ge 0$ ($T < T_c$ enters the two-phase spinodal/binodal regime).
    pub temperature: f64,
    /// Critical spinodal temperature $T_c > 0$ defining $f(\phi) = \frac{1}{4}\phi^4 + \frac{T - T_c}{2}\phi^2 + c$.
    pub critical_temperature: f64,
    /// Kinetic mobility $M > 0$.
    pub mobility: f64,
    /// Interfacial gradient-energy penalty $\kappa > 0$.
    pub kappa: f64,
    /// Thermal fluctuation strength $\sigma \ge 0$.
    pub noise_sigma: f64,
    /// Isotropic spatial lattice spacing $\Delta x = \Delta y > 0$.
    pub dx: f64,
    /// Semi-implicit time step $\Delta t > 0$.
    pub dt: f64,
}

impl Default for PhaseFieldParams {
    fn default() -> Self {
        Self {
            model: PhaseFieldModel::CahnHilliard,
            temperature: 0.25,
            critical_temperature: 1.0,
            mobility: 1.0,
            kappa: 0.5,
            noise_sigma: 0.04,
            dx: 1.0,
            dt: 0.05,
        }
    }
}

/// Zero-allocation 2D periodic stochastic phase-field solver with radial $S(k, t)$ tracking.
#[derive(Clone, Debug)]
pub struct PhaseFieldSolver2D {
    width: usize,
    height: usize,
    params: PhaseFieldParams,
    phi: Vec<f64>,
    rhs_scratch: Vec<f64>,
    spectral: SemiImplicitSpectral2D,
    noise_field: Vec<f64>,
    noise_scratch: Vec<f64>,
    telemetry: PhaseFieldTelemetry,
    time: f64,
    steps: u64,
}

impl PhaseFieldSolver2D {
    /// Constructs a uniform phase field $\phi(x, y) = \text{mean\_phi}$ with pre-allocated FFT and noise workspaces.
    pub fn new(width: usize, height: usize, mean_phi: f64, params: PhaseFieldParams) -> Self {
        assert!(width >= 4 && height >= 4, "Phase-field grid must be >= 4x4");
        assert!(
            mean_phi.is_finite(),
            "Initial mean concentration must be finite"
        );
        validate_phase_field_params(&params);

        let spectral = SemiImplicitSpectral2D::new(width, height, params.dx, params.dx);
        let num_bins = spectral.radial_bin_count();
        let total = width * height;

        Self {
            width,
            height,
            params,
            phi: vec![mean_phi; total],
            rhs_scratch: vec![0.0; total],
            spectral,
            noise_field: vec![0.0; total],
            noise_scratch: vec![0.0; 2 * total],
            telemetry: PhaseFieldTelemetry::new(num_bins, 256),
            time: 0.0,
            steps: 0,
        }
    }

    /// Constructs a quenched spinodal initial state $\phi = \bar{\phi} + \delta\phi$ with exact mean $\bar{\phi}$.
    pub fn new_spinodal(
        width: usize,
        height: usize,
        mean_phi: f64,
        perturbation_std: f64,
        rng: &mut OxidizeRng,
        params: PhaseFieldParams,
    ) -> Self {
        let mut solver = Self::new(width, height, mean_phi, params);
        solver.reseed_spinodal(mean_phi, perturbation_std, rng);
        solver
    }

    /// Re-initializes $\phi(x, y)$ around `mean_phi` with zero-sum Gaussian fluctuations and refreshes $S(k, 0)$.
    #[verified_engine::verified]
    pub fn reseed_spinodal(&mut self, mean_phi: f64, perturbation_std: f64, rng: &mut OxidizeRng) {
        assert!(
            mean_phi.is_finite() && perturbation_std.is_finite() && perturbation_std >= 0.0,
            "Spinodal mean and perturbation amplitude must be finite"
        );
        fill_standard_normal(&mut self.phi, rng);
        let sample_mean = mean_field(&self.phi);
        for v in self.phi.iter_mut() {
            *v = mean_phi + perturbation_std * (*v - sample_mean);
        }
        self.time = 0.0;
        self.steps = 0;
        self.telemetry.clear();
        let _ = self.refresh_telemetry();
    }

    /// Updates physical and numerical parameters in place without heap allocation.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: PhaseFieldParams) {
        validate_phase_field_params(&params);
        if (params.dx - self.params.dx).abs() > f64::EPSILON {
            self.spectral.set_grid_spacing(params.dx, params.dx);
        }
        self.params = params;
    }

    /// Grid width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Current elapsed simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Completed semi-implicit time steps.
    #[verified_engine::verified]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Active phase-field parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> PhaseFieldParams {
        self.params
    }

    /// Immutable slice of the 2D order parameter $\phi(x, y, t)$.
    #[verified_engine::verified]
    pub fn field(&self) -> &[f64] {
        &self.phi
    }

    /// Reference to the pre-allocated radial $S(k, t)$ and coarsening telemetry store.
    #[verified_engine::verified]
    pub fn telemetry(&self) -> &PhaseFieldTelemetry {
        &self.telemetry
    }

    /// Spatial mean concentration $\langle \phi \rangle = \frac{1}{WH}\sum_{x,y} \phi_{x,y}$.
    #[verified_engine::verified]
    pub fn mean_concentration(&self) -> f64 {
        mean_field(&self.phi)
    }

    /// Discrete Ginzburg-Landau free energy $F_h[\phi] = \sum_{x,y} \Delta x^2 [f(\phi) + \frac{\kappa}{2}|\nabla_h \phi|^2]$.
    #[verified_engine::verified]
    pub fn free_energy(&self) -> f64 {
        compute_discrete_free_energy(self.width, self.height, &self.phi, &self.params)
    }

    /// Paints a smooth circular droplet perturbation; optionally compensates background to preserve $\langle\phi\rangle$.
    #[verified_engine::verified]
    pub fn paint_droplet(
        &mut self,
        cx: usize,
        cy: usize,
        radius: usize,
        delta: f64,
        conserve_mean: bool,
    ) {
        assert!(delta.is_finite(), "Droplet delta must be finite");
        let before_mean = self.mean_concentration();
        apply_periodic_droplet(
            self.width,
            self.height,
            &mut self.phi,
            cx,
            cy,
            radius,
            delta,
        );
        if conserve_mean {
            let drift = self.mean_concentration() - before_mean;
            for v in self.phi.iter_mut() {
                *v -= drift;
            }
        }
    }

    /// Advances $\phi(x, y, t)$ by one semi-implicit step with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, rng: &mut OxidizeRng) {
        let stabilizer_s = fill_eyre_nonlinear_rhs(&self.phi, &mut self.rhs_scratch, &self.params);
        let p = self.params;
        let has_noise = sample_phase_field_noise(
            self.width,
            self.height,
            &p,
            &mut self.noise_field,
            &mut self.noise_scratch,
            rng,
        );
        let noise = has_noise.then_some(self.noise_field.as_slice());

        match p.model {
            PhaseFieldModel::AllenCahn => {
                for v in self.rhs_scratch.iter_mut() {
                    *v *= -p.mobility;
                }
                self.spectral.step_reaction_biharmonic(
                    &mut self.phi,
                    &self.rhs_scratch,
                    noise,
                    ReactionBiharmonicParams {
                        dt: p.dt,
                        alpha_0: p.mobility * stabilizer_s,
                        alpha_2: p.mobility * p.kappa,
                        alpha_4: 0.0,
                    },
                );
            }
            PhaseFieldModel::CahnHilliard => {
                self.spectral.step_conserved_biharmonic(
                    &mut self.phi,
                    &self.rhs_scratch,
                    noise,
                    ConservedBiharmonicParams {
                        dt: p.dt,
                        mobility: p.mobility,
                        stabilizer_s,
                        kappa: p.kappa,
                    },
                );
            }
        }
        self.time += p.dt;
        self.steps += 1;
    }

    /// Advances $\phi(x, y, t)$ by `n_steps` semi-implicit steps with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_n(&mut self, rng: &mut OxidizeRng, n_steps: usize) {
        for _ in 0..n_steps {
            self.step(rng);
        }
    }

    /// Evaluates radial $S(k, t)$, $k_{\max}(t)$, and $F_h[\phi]$ into pre-allocated telemetry buffers.
    #[verified_engine::verified]
    pub fn refresh_telemetry(&mut self) -> RadialSpectrumSummary {
        let fe = self.free_energy();
        self.telemetry
            .update_from_field(&mut self.spectral, &self.phi, self.time, fe)
    }
}

fn sample_phase_field_noise(
    width: usize,
    height: usize,
    p: &PhaseFieldParams,
    noise_field: &mut [f64],
    noise_scratch: &mut [f64],
    rng: &mut OxidizeRng,
) -> bool {
    let mobility_temp = p.mobility * p.noise_sigma * p.noise_sigma;
    if mobility_temp <= 0.0 {
        return false;
    }
    noise_field.fill(0.0);
    let mut ws = SolverWorkspace::new(noise_scratch);
    match p.model {
        PhaseFieldModel::AllenCahn => inject_additive_noise_2d(
            noise_field,
            width,
            height,
            p.dx,
            p.dx,
            p.dt,
            mobility_temp,
            BoundaryCondition::Periodic,
            &mut ws,
            rng,
        ),
        PhaseFieldModel::CahnHilliard => inject_conserved_flux_noise_2d(
            noise_field,
            width,
            height,
            p.dx,
            p.dx,
            p.dt,
            mobility_temp,
            BoundaryCondition::Periodic,
            &mut ws,
            rng,
        ),
    }
    true
}

fn validate_phase_field_params(p: &PhaseFieldParams) {
    assert!(
        p.temperature.is_finite() && p.temperature >= 0.0,
        "Temperature must be finite and non-negative"
    );
    assert!(
        p.critical_temperature.is_finite() && p.critical_temperature > 0.0,
        "Critical temperature Tc must be finite and positive"
    );
    assert!(
        p.mobility.is_finite() && p.mobility > 0.0,
        "Mobility M must be finite and positive"
    );
    assert!(
        p.kappa.is_finite() && p.kappa >= 0.0,
        "Interfacial kappa must be finite and non-negative"
    );
    assert!(
        p.noise_sigma.is_finite() && p.noise_sigma >= 0.0,
        "Noise amplitude must be finite and non-negative"
    );
    assert!(
        p.dx.is_finite() && p.dx > 0.0 && p.dt.is_finite() && p.dt > 0.0,
        "dx and dt must be finite and positive"
    );
}

fn fill_eyre_nonlinear_rhs(phi: &[f64], mu_nl_out: &mut [f64], p: &PhaseFieldParams) -> f64 {
    let a = p.temperature - p.critical_temperature;
    let mut max_sq = 1.0_f64;
    for &v in phi {
        let sq = v * v;
        if sq > max_sq {
            max_sq = sq;
        }
    }
    let stabilizer_s = (3.0 * max_sq + a.max(0.0)).max(2.0);
    let linear_shift = a - stabilizer_s;
    for (dst, &v) in mu_nl_out.iter_mut().zip(phi.iter()) {
        *dst = v * v * v + linear_shift * v;
    }
    stabilizer_s
}

fn compute_discrete_free_energy(
    width: usize,
    height: usize,
    phi: &[f64],
    p: &PhaseFieldParams,
) -> f64 {
    let a = p.temperature - p.critical_temperature;
    let tau_pos = (-a).max(0.0);
    let well_offset = 0.25 * tau_pos * tau_pos;
    let inv_dx_sq = 1.0 / (p.dx * p.dx);
    let half_kappa = 0.5 * p.kappa;
    let cell_area = p.dx * p.dx;
    let mut total = 0.0;

    for y in 0..height {
        let yp = if y + 1 == height { 0 } else { y + 1 };
        for x in 0..width {
            let xp = if x + 1 == width { 0 } else { x + 1 };
            let idx = flatten_2d_index(x, y, width);
            let u = phi[idx];
            let ux = phi[flatten_2d_index(xp, y, width)] - u;
            let uy = phi[flatten_2d_index(x, yp, width)] - u;
            let u2 = u * u;
            let bulk = 0.25 * u2 * u2 + 0.5 * a * u2 + well_offset;
            let grad = half_kappa * (ux * ux + uy * uy) * inv_dx_sq;
            total += (bulk + grad) * cell_area;
        }
    }
    total
}

fn apply_periodic_droplet(
    width: usize,
    height: usize,
    phi: &mut [f64],
    cx: usize,
    cy: usize,
    radius: usize,
    delta: f64,
) {
    let tx = cx % width;
    let ty = cy % height;
    let r = radius.max(1) as f64;
    let r_sq = r * r;
    for y in 0..height {
        let ry = (y as isize - ty as isize).unsigned_abs();
        let dy = ry.min(height - ry) as f64;
        for x in 0..width {
            let rx = (x as isize - tx as isize).unsigned_abs();
            let dx = rx.min(width - rx) as f64;
            let d2 = dx * dx + dy * dy;
            if d2 <= r_sq {
                let weight = 1.0 - (d2 / r_sq);
                let idx = flatten_2d_index(x, y, width);
                phi[idx] = (phi[idx] + delta * weight).clamp(-1.8, 1.8);
            }
        }
    }
}
