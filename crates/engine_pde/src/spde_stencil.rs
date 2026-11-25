// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::spde_noise::{SpdeNoiseKind, apply_noise_1d, apply_noise_2d};
use crate::stencil::StencilOperators;
use math_commons::math_kernel::types::{SolverWorkspace, flatten_2d_index};
use rand_core::RngCore;

/// Explicit Euler-Maruyama finite-difference SPDE stepper evolving
/// $\partial_t u = \mathcal{N}(u, \nabla_h u, \nabla_h^2 u) + \xi$ across Periodic, Neumann, and Dirichlet
/// boundaries with additive or staggered-face conserved-flux noise.
///
/// Reference: LeVeque (2007), DOI: 10.1137/1.9780898717839;
/// [`docs/math_specs/engine_pde.md`](../../../docs/math_specs/engine_pde.md).
///
/// # Stability
/// Second-order central diffusion requires $D\,\Delta t \sum_k \Delta x_k^{-2} \le \frac{1}{2}$, while
/// explicit 2D Biharmonic Cahn-Hilliard stepping requires $M\kappa\,\Delta t(\Delta x^{-2} + \Delta y^{-2})^2 \le \frac{1}{8}$.
///
/// # Domain of Validity
/// Requires positive finite stencil spacings (`dx, dy > 0`), `dt >= 0`, non-negative noise intensity,
/// and `SolverWorkspace` capacity [`Self::scratch_len_1d`] / [`Self::scratch_len_2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplicitSpdeStencilStepper {
    /// Spatial finite-difference stencil operators.
    pub ops: StencilOperators,
    /// Boundary condition enforced on all spatial axes.
    pub boundary: BoundaryCondition,
    /// Stochastic noise structure.
    pub noise: SpdeNoiseKind,
}

impl ExplicitSpdeStencilStepper {
    /// Creates an [`ExplicitSpdeStencilStepper`] from prevalidated [`StencilOperators`].
    #[inline]
    pub const fn new(
        ops: StencilOperators,
        boundary: BoundaryCondition,
        noise: SpdeNoiseKind,
    ) -> Self {
        Self {
            ops,
            boundary,
            noise,
        }
    }

    /// Returns the scratch buffer length required for 1D SPDE steps (`2 * n`).
    #[inline]
    pub const fn scratch_len_1d(n: usize) -> usize {
        2 * n
    }

    /// Returns the scratch buffer length required for 2D SPDE steps (`5 * width * height`).
    #[inline]
    pub const fn scratch_len_2d(width: usize, height: usize) -> usize {
        5 * width * height
    }

    /// Advances a 1D SPDE $\partial_t u = \mathcal{N}(x, u, \partial_x u, \partial_{xx} u) + \xi$ in-place.
    #[verified_engine::verified]
    pub fn step_1d<F, R>(
        &self,
        u: &mut [f64],
        dt: f64,
        mut deterministic_rhs: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) where
        F: FnMut(usize, f64, f64, f64) -> f64,
        R: RngCore + ?Sized,
    {
        let n = u.len();
        assert!(dt >= 0.0);
        if n == 0 || dt == 0.0 {
            return;
        }
        let mut ws = workspace.reborrow();
        let src = ws.take_slice(n);
        src.copy_from_slice(u);
        for x in 0..n {
            if let BoundaryCondition::Dirichlet(val) = self.boundary
                && (x == 0 || x + 1 == n)
            {
                u[x] = val;
                continue;
            }
            let grad = self.ops.gradient_1d_at(src, n, x, self.boundary);
            let lap = self.ops.laplacian_1d_at(src, n, x, self.boundary);
            u[x] = src[x] + dt * deterministic_rhs(x, src[x], grad, lap);
        }
        apply_noise_1d(u, *self.ops.dx, dt, self.boundary, self.noise, &mut ws, rng);
    }

    /// Advances a 2D SPDE $\partial_t u = \mathcal{N}(x, y, u, \nabla u, \nabla^2 u) + \xi$ in-place.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        dt: f64,
        mut deterministic_rhs: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) where
        F: FnMut(usize, usize, f64, (f64, f64), f64) -> f64,
        R: RngCore + ?Sized,
    {
        let total = width * height;
        assert!(u.len() >= total && dt >= 0.0);
        if total == 0 || dt == 0.0 {
            return;
        }
        let mut ws = workspace.reborrow();
        let src = ws.take_slice(total);
        src.copy_from_slice(&u[..total]);
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                if let BoundaryCondition::Dirichlet(val) = self.boundary
                    && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                {
                    u[idx] = val;
                    continue;
                }
                let grad = self
                    .ops
                    .gradient_2d_at(src, width, height, x, y, self.boundary);
                let lap = self
                    .ops
                    .laplacian_2d_at(src, width, height, x, y, self.boundary);
                u[idx] = src[idx] + dt * deterministic_rhs(x, y, src[idx], grad, lap);
            }
        }
        apply_noise_2d(
            &mut u[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut ws,
            rng,
        );
    }

    /// Advances the 2D conserved Cahn-Hilliard SPDE
    /// $\partial_t \phi = M \left[\nabla^2(\phi^3 - \phi) - \kappa \nabla^4 \phi\right] + \nabla \cdot \vec{\eta}$
    /// using [`StencilOperators::eval_biharmonic_2d_from_laplacian`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) {
        let total = width * height;
        assert!(phi.len() >= total && mobility >= 0.0 && kappa >= 0.0 && dt >= 0.0);
        if total == 0 || dt == 0.0 {
            return;
        }
        let mut ws = workspace.reborrow();
        let src = ws.take_slice(total);
        let bulk_mu = ws.take_slice(total);
        let lap_phi = ws.take_slice(total);
        src.copy_from_slice(&phi[..total]);
        for i in 0..total {
            let p = src[i];
            bulk_mu[i] = p * p * p - p;
        }
        self.ops
            .compute_laplacian_2d(src, width, height, self.boundary, lap_phi);

        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                if let BoundaryCondition::Dirichlet(val) = self.boundary
                    && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                {
                    phi[idx] = val;
                    continue;
                }
                let lap_bulk =
                    self.ops
                        .laplacian_2d_at(bulk_mu, width, height, x, y, self.boundary);
                let biharm = self.ops.eval_biharmonic_2d_from_laplacian(
                    lap_phi,
                    width,
                    height,
                    x,
                    y,
                    self.boundary,
                );
                phi[idx] = src[idx] + dt * mobility * (lap_bulk - kappa * biharm);
            }
        }
        apply_noise_2d(
            &mut phi[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut ws,
            rng,
        );
    }
}

/// Operator-splitting Strang SPDE stencil stepper advancing $\partial_t u = D \nabla_h^2 u + R(u) + \xi$
/// via half-reaction ($\Delta t / 2$), `diffusion_substeps` diffusive substeps, half-reaction, and noise.
///
/// Reference: Strang (1968), DOI: 10.1137/0705041;
/// [`docs/math_specs/engine_pde.md`](../../../docs/math_specs/engine_pde.md).
///
/// # Stability
/// Partitioning diffusion into $S = \text{diffusion\_substeps} \ge 1$ substeps relaxes the parabolic CFL
/// constraint to $D(\Delta t / S)(\Delta x^{-2} + \Delta y^{-2}) \le \frac{1}{2}$ while retaining second-order deterministic splitting.
///
/// # Domain of Validity
/// Requires positive grid spacings (`dx, dy > 0`), `diffusivity >= 0`, `dt >= 0`, `u.len() >= width * height`,
/// and `SolverWorkspace` capacity at least [`Self::scratch_len_2d`]`(width, height)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitStepSpdeStencilStepper {
    /// Spatial finite-difference stencil operators.
    pub ops: StencilOperators,
    /// Boundary condition enforced during diffusive substeps.
    pub boundary: BoundaryCondition,
    /// Stochastic noise structure applied after operator splitting.
    pub noise: SpdeNoiseKind,
    /// Number of diffusive substeps executed between half-reaction kicks (`>= 1`).
    pub diffusion_substeps: usize,
}

impl SplitStepSpdeStencilStepper {
    /// Creates a [`SplitStepSpdeStencilStepper`] with `diffusion_substeps.max(1)` diffusive substeps.
    #[inline]
    pub const fn new(
        ops: StencilOperators,
        boundary: BoundaryCondition,
        noise: SpdeNoiseKind,
        diffusion_substeps: usize,
    ) -> Self {
        let subs = if diffusion_substeps == 0 {
            1
        } else {
            diffusion_substeps
        };
        Self {
            ops,
            boundary,
            noise,
            diffusion_substeps: subs,
        }
    }

    /// Returns the scratch buffer length required for 2D split-step integration (`3 * width * height`).
    #[inline]
    pub const fn scratch_len_2d(width: usize, height: usize) -> usize {
        3 * width * height
    }

    /// Advances a 2D reaction-diffusion SPDE $\partial_t u = D \nabla^2 u + R(u) + \xi$ via Strang splitting.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_reaction_diffusion_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        diffusivity: f64,
        dt: f64,
        mut reaction: F,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) where
        F: FnMut(f64) -> f64,
        R: RngCore + ?Sized,
    {
        let total = width * height;
        assert!(u.len() >= total && diffusivity >= 0.0 && dt >= 0.0);
        if total == 0 || dt == 0.0 {
            return;
        }
        let half_dt = 0.5 * dt;
        apply_local_reaction_2d(u, width, height, half_dt, self.boundary, &mut reaction);

        let sub_dt = dt / (self.diffusion_substeps as f64);
        let mut ws = workspace.reborrow();
        let src = ws.take_slice(total);
        for _ in 0..self.diffusion_substeps {
            src.copy_from_slice(&u[..total]);
            for y in 0..height {
                for x in 0..width {
                    let idx = flatten_2d_index(x, y, width);
                    if let BoundaryCondition::Dirichlet(val) = self.boundary
                        && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                    {
                        u[idx] = val;
                        continue;
                    }
                    let lap = self
                        .ops
                        .laplacian_2d_at(src, width, height, x, y, self.boundary);
                    u[idx] = src[idx] + sub_dt * diffusivity * lap;
                }
            }
        }

        apply_local_reaction_2d(u, width, height, half_dt, self.boundary, &mut reaction);
        apply_noise_2d(
            &mut u[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut ws,
            rng,
        );
    }
}

/// Checked `#![no_std]` companion steppers returning [`MathError`](math_commons::error::MathError).
pub mod checked;

fn apply_local_reaction_2d<F: FnMut(f64) -> f64>(
    u: &mut [f64],
    width: usize,
    height: usize,
    dt_react: f64,
    bc: BoundaryCondition,
    reaction: &mut F,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if let BoundaryCondition::Dirichlet(val) = bc
                && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
            {
                u[idx] = val;
                continue;
            }
            u[idx] += dt_react * reaction(u[idx]);
        }
    }
}
