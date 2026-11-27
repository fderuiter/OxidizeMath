// [cite:engine_opt]
//! Monotone 1D Mean-Field Game (HJB / Fokker-Planck) Mann fixed-point solver over `SolverWorkspace`.

mod fp;

use crate::error::OptError;
use engine_pde::StencilOperators;
use fp::{compute_substeps, forward_fp_sweep, mann_blend_and_normalize};
use math_scalars::workspace::SolverWorkspace;

const CFL_SAFETY: f64 = 0.45;
const MAX_SUBSTEPS: usize = 65_536;
const MIN_MASS_SUM: f64 = 1.0e-15;

/// Convex differentiable Hamiltonian $H(p)$ governing optimal agent drift $v = -H'(p)$.
///
/// $$
/// H(p) = \sup_{\alpha \in \mathbb{R}} \bigl(-\alpha p - L(\alpha)\bigr)
/// $$
///
/// Reference: Achdou & Capuzzo-Dolcetta (2010), DOI: 10.1137/090758477.
///
/// # Stability
/// Convexity $H''(p) \ge 0$ with minimum at $p = 0$ guarantees monotonicity of the Engquist-Osher flux.
///
/// # Domain of Validity
/// Defined for all finite momentum gradients $p \in \mathbb{R}$.
pub trait MfgHamiltonian {
    /// Evaluates $H(p)$.
    fn eval(&self, p: f64) -> f64;
    /// Evaluates the derivative $H'(p)$.
    fn deriv(&self, p: f64) -> f64;
}

/// Standard quadratic Hamiltonian $H(p) = \frac{1}{2} c p^2$ with $H'(p) = c p$.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuadraticMfgHamiltonian {
    /// Positive mobility coefficient $c > 0$ (defaults to `1.0`).
    pub coeff: f64,
}

impl Default for QuadraticMfgHamiltonian {
    fn default() -> Self {
        Self { coeff: 1.0 }
    }
}

impl MfgHamiltonian for QuadraticMfgHamiltonian {
    #[inline]
    fn eval(&self, p: f64) -> f64 {
        0.5 * self.coeff * p * p
    }

    #[inline]
    fn deriv(&self, p: f64) -> f64 {
        self.coeff * p
    }
}

/// Numerical and physical configuration for 1D Mann-damped Mean-Field Games.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfgConfig1D {
    /// Left spatial boundary $x_{\min}$.
    pub space_min: f64,
    /// Uniform spatial grid spacing $\Delta x > 0$.
    pub dx: f64,
    /// Macro time step $\Delta t > 0$.
    pub dt: f64,
    /// Diffusion viscosity $\nu \ge 0$ (`0.0` selects the deterministic inviscid limit).
    pub viscosity: f64,
    /// Number of spatial grid nodes $N_x \ge 2$.
    pub nx: usize,
    /// Number of macro time intervals $N_t \ge 1$ (`nt + 1` time slices).
    pub nt: usize,
    /// Maximum Mann fixed-point sweeps.
    pub max_iterations: usize,
    /// Mann damping weight $\alpha \in (0, 1]$ (`1.0` is undamped Picard).
    pub mann_alpha: f64,
    /// Stopping threshold on the $L^\infty_t L^1_x$ density residual.
    pub l1_tolerance: f64,
}

/// Convergence telemetry returned by [`solve_mfg_1d`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfgStats {
    /// Number of Mann sweeps executed.
    pub iterations: usize,
    /// Final $\max_n \sum_i |\tilde{m}_i^n - m_i^n| \Delta x$ residual.
    pub l1_residual: f64,
    /// Largest adaptive CFL substep count used in backward HJB sweeps.
    pub max_hjb_substeps: usize,
    /// Largest adaptive CFL substep count used in forward Fokker-Planck sweeps.
    pub max_fp_substeps: usize,
    /// Whether `l1_residual <= l1_tolerance` was reached.
    pub converged: bool,
}

/// Returns the exact `SolverWorkspace<f64>` length required by [`solve_mfg_1d`].
#[inline]
#[must_use]
pub const fn mfg_1d_workspace_len(nx: usize, nt: usize) -> usize {
    nx.saturating_mul(nt.saturating_add(1))
        .saturating_add(nx.saturating_mul(4))
        .saturating_add(1)
}

/// Solves a coupled 1D HJB / Fokker-Planck Mean-Field Game into column-major `(u_grid, m_grid)` slices.
///
/// $$
/// -\partial_t u - \nu \partial_{xx} u + H_{\text{EO}}(D^+ u_i, D^- u_i) = F(x_i, m_i), \qquad
/// \partial_t m - \nu \partial_{xx} m + \frac{F_{i+1/2} - F_{i-1/2}}{\Delta x} = 0
/// $$
///
/// Reference: Achdou & Capuzzo-Dolcetta (2010), DOI: 10.1137/090758477.
///
/// # Stability
/// Pairs the monotone Engquist-Osher Hamiltonian with its adjoint donor-cell Fokker-Planck flux,
/// adaptive CFL substepping down to $\nu = 0$, unit integral normalization $\sum_i m_i \Delta x = 1$,
/// and Mann fixed-point damping $m^{(k+1)} = (1 - \alpha) m^{(k)} + \alpha \tilde{m}$.
///
/// # Domain of Validity
/// Requires `nx >= 2`, `nt >= 1`, `dx > 0`, `dt > 0`, `viscosity >= 0`, `mann_alpha` in `(0, 1]`,
/// `u_grid.len() == m_grid.len() == nx * (nt + 1)`, and `ws` length at least [`mfg_1d_workspace_len`].
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn solve_mfg_1d<H, C, G, M0>(
    cfg: &MfgConfig1D,
    hamiltonian: &H,
    running_cost: &C,
    terminal_cost: &G,
    initial_density: &M0,
    u_grid: &mut [f64],
    m_grid: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<MfgStats, OptError>
where
    H: MfgHamiltonian,
    C: Fn(f64, f64) -> f64,
    G: Fn(f64, f64) -> f64,
    M0: Fn(f64) -> f64,
{
    validate_mfg_inputs(cfg, u_grid.len(), m_grid.len())?;
    let cols = cfg.nt + 1;
    let total = cfg.nx * cols;
    let req = mfg_1d_workspace_len(cfg.nx, cfg.nt);
    if ws.len() < req {
        return Err(OptError::WorkspaceTooSmall {
            required: req,
            available: ws.len(),
        });
    }
    let (m_cand, mut r1) = ws.try_split(total)?;
    let (xs, mut r2) = r1.try_split(cfg.nx)?;
    let (work_a, mut r3) = r2.try_split(cfg.nx)?;
    let (work_b, mut r4) = r3.try_split(cfg.nx)?;
    let (v_faces, _) = r4.try_split(cfg.nx + 1)?;
    debug_assert_eq!(m_cand.len(), total);
    debug_assert_eq!(xs.len(), cfg.nx);
    debug_assert_eq!(v_faces.len(), cfg.nx + 1);

    init_grid_and_m0(cfg, initial_density, xs, m_grid)?;
    let stencil = StencilOperators::try_new_1d_f64(cfg.dx).ok_or(OptError::InvalidParameter)?;
    let mut stats = MfgStats {
        iterations: 0,
        l1_residual: 0.0,
        max_hjb_substeps: 1,
        max_fp_substeps: 1,
        converged: cfg.max_iterations == 0,
    };
    if cfg.max_iterations == 0 {
        fill_terminal_u(cfg, terminal_cost, xs, m_grid, u_grid)?;
        return Ok(stats);
    }

    for iter in 0..cfg.max_iterations {
        let hjb_sub = backward_hjb_sweep(
            cfg,
            &stencil,
            hamiltonian,
            running_cost,
            terminal_cost,
            xs,
            m_grid,
            u_grid,
            work_a,
            work_b,
        )?;
        let fp_sub = forward_fp_sweep(
            cfg,
            &stencil,
            hamiltonian,
            u_grid,
            m_grid,
            m_cand,
            work_a,
            work_b,
            v_faces,
        )?;
        stats.max_hjb_substeps = stats.max_hjb_substeps.max(hjb_sub);
        stats.max_fp_substeps = stats.max_fp_substeps.max(fp_sub);
        stats.l1_residual = mann_blend_and_normalize(cfg, m_grid, m_cand)?;
        debug_assert!(stats.l1_residual.is_finite() && stats.l1_residual >= 0.0);
        debug_assert!(stats.max_hjb_substeps >= 1 && stats.max_fp_substeps >= 1);
        stats.iterations = iter + 1;
        if stats.l1_residual <= cfg.l1_tolerance {
            let _ = backward_hjb_sweep(
                cfg,
                &stencil,
                hamiltonian,
                running_cost,
                terminal_cost,
                xs,
                m_grid,
                u_grid,
                work_a,
                work_b,
            )?;
            stats.converged = true;
            break;
        }
    }
    debug_assert!(stats.iterations <= cfg.max_iterations);
    Ok(stats)
}

#[verified_engine::verified]
fn validate_mfg_inputs(cfg: &MfgConfig1D, u_len: usize, m_len: usize) -> Result<(), OptError> {
    if cfg.nx < 2 || cfg.nt < 1 {
        return Err(OptError::InvalidDimension);
    }
    let expected = cfg
        .nx
        .checked_mul(cfg.nt.saturating_add(1))
        .ok_or(OptError::InvalidDimension)?;
    if u_len != expected || m_len != expected {
        return Err(OptError::InvalidDimension);
    }
    if !cfg.space_min.is_finite()
        || !cfg.dx.is_finite()
        || !cfg.dt.is_finite()
        || !cfg.viscosity.is_finite()
        || !cfg.mann_alpha.is_finite()
        || !cfg.l1_tolerance.is_finite()
    {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.dx <= 0.0
        || cfg.dt <= 0.0
        || cfg.viscosity < 0.0
        || cfg.mann_alpha <= 0.0
        || cfg.mann_alpha > 1.0
        || cfg.l1_tolerance < 0.0
    {
        return Err(OptError::InvalidParameter);
    }
    Ok(())
}

#[verified_engine::verified]
fn normalize_density_slice(col: &mut [f64], dx: f64) -> Result<(), OptError> {
    let mut sum = 0.0;
    for slot in col.iter_mut() {
        if !slot.is_finite() {
            return Err(OptError::NumericalInstability);
        }
        *slot = (*slot).max(0.0);
        sum += *slot;
    }
    let integral = sum * dx;
    if !integral.is_finite() || integral <= MIN_MASS_SUM {
        return Err(OptError::InvalidParameter);
    }
    let scale = 1.0 / integral;
    for slot in col.iter_mut() {
        *slot *= scale;
    }
    Ok(())
}

#[verified_engine::verified]
fn init_grid_and_m0<M0: Fn(f64) -> f64>(
    cfg: &MfgConfig1D,
    initial_density: &M0,
    xs: &mut [f64],
    m_grid: &mut [f64],
) -> Result<(), OptError> {
    for (i, x_slot) in xs.iter_mut().enumerate() {
        let x = cfg.space_min + (i as f64) * cfg.dx;
        *x_slot = x;
        let m0 = initial_density(x);
        if !m0.is_finite() || m0 < 0.0 {
            return Err(OptError::InvalidParameter);
        }
        m_grid[i] = m0;
    }
    normalize_density_slice(&mut m_grid[..cfg.nx], cfg.dx)?;
    for n in 1..=cfg.nt {
        let (first, rest) = m_grid.split_at_mut(n * cfg.nx);
        rest[..cfg.nx].copy_from_slice(&first[..cfg.nx]);
    }
    Ok(())
}

#[verified_engine::verified]
fn fill_terminal_u<G: Fn(f64, f64) -> f64>(
    cfg: &MfgConfig1D,
    terminal_cost: &G,
    xs: &[f64],
    m_grid: &[f64],
    u_grid: &mut [f64],
) -> Result<(), OptError> {
    let base = cfg.nt * cfg.nx;
    for i in 0..cfg.nx {
        let val = terminal_cost(xs[i], m_grid[base + i]);
        if !val.is_finite() {
            return Err(OptError::NumericalInstability);
        }
        u_grid[base + i] = val;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn backward_hjb_sweep<H: MfgHamiltonian, C: Fn(f64, f64) -> f64, G: Fn(f64, f64) -> f64>(
    cfg: &MfgConfig1D,
    stencil: &StencilOperators,
    hamiltonian: &H,
    running_cost: &C,
    terminal_cost: &G,
    xs: &[f64],
    m_grid: &[f64],
    u_grid: &mut [f64],
    work_a: &mut [f64],
    work_b: &mut [f64],
) -> Result<usize, OptError> {
    fill_terminal_u(cfg, terminal_cost, xs, m_grid, u_grid)?;
    let h0 = hamiltonian.eval(0.0);
    let mut max_sub = 1usize;

    for n in (0..cfg.nt).rev() {
        let next_off = (n + 1) * cfg.nx;
        work_a.copy_from_slice(&u_grid[next_off..next_off + cfg.nx]);
        let c_max = max_hjb_wave_speed(work_a, stencil.inv_dx, hamiltonian)?;
        let rate = 2.0 * cfg.viscosity * stencil.inv_dx_sq + c_max * stencil.inv_dx;
        let n_sub = compute_substeps(cfg.dt, rate)?;
        max_sub = max_sub.max(n_sub);
        let dt_sub = cfg.dt / (n_sub as f64);

        for _ in 0..n_sub {
            step_hjb_once(
                cfg,
                stencil,
                hamiltonian,
                h0,
                running_cost,
                xs,
                &m_grid[next_off..next_off + cfg.nx],
                dt_sub,
                work_a,
                work_b,
            )?;
            work_a.copy_from_slice(work_b);
        }
        u_grid[n * cfg.nx..(n + 1) * cfg.nx].copy_from_slice(work_a);
    }
    Ok(max_sub)
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn step_hjb_once<H: MfgHamiltonian, C: Fn(f64, f64) -> f64>(
    cfg: &MfgConfig1D,
    stencil: &StencilOperators,
    hamiltonian: &H,
    h0: f64,
    running_cost: &C,
    xs: &[f64],
    m_col: &[f64],
    dt_sub: f64,
    u_in: &[f64],
    u_out: &mut [f64],
) -> Result<(), OptError> {
    let nx = cfg.nx;
    for i in 0..nx {
        let u_prev = if i == 0 { u_in[0] } else { u_in[i - 1] };
        let u_curr = u_in[i];
        let u_next = if i + 1 == nx {
            u_in[nx - 1]
        } else {
            u_in[i + 1]
        };
        let d_minus = (u_curr - u_prev) * stencil.inv_dx;
        let d_plus = (u_next - u_curr) * stencil.inv_dx;
        let h_eo = hamiltonian.eval(d_minus.max(0.0)) + hamiltonian.eval(d_plus.min(0.0)) - h0;
        let lap = stencil.central_diff_2nd(u_prev, u_curr, u_next);
        let f_cost = running_cost(xs[i], m_col[i]);
        let val = u_curr + dt_sub * (cfg.viscosity * lap - h_eo + f_cost);
        if !val.is_finite() {
            return Err(OptError::NumericalInstability);
        }
        u_out[i] = val;
    }
    Ok(())
}

#[verified_engine::verified]
fn max_hjb_wave_speed<H: MfgHamiltonian>(
    u_col: &[f64],
    inv_dx: f64,
    hamiltonian: &H,
) -> Result<f64, OptError> {
    let mut c_max = 0.0_f64;
    for i in 0..(u_col.len() - 1) {
        let grad = (u_col[i + 1] - u_col[i]) * inv_dx;
        let speed = hamiltonian.deriv(grad).abs();
        if !speed.is_finite() {
            return Err(OptError::NumericalInstability);
        }
        c_max = c_max.max(speed);
    }
    Ok(c_max)
}
