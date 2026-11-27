use super::physics::{Hamiltonian, QuadraticHamiltonian};
use super::types::{Density, MFGConfig, Position};
use engine_opt::mfg::{MfgConfig1D, MfgHamiltonian, MfgStats, mfg_1d_workspace_len, solve_mfg_1d};
use engine_opt::{OptError, SolverWorkspace};
use nalgebra::DMatrix;

const DEFAULT_MANN_DAMPING: f64 = 0.5;
const DEFAULT_L1_TOLERANCE: f64 = 1.0e-8;

struct DomainHamiltonianAdapter<'a, H: Hamiltonian>(&'a H);

impl<H: Hamiltonian> MfgHamiltonian for DomainHamiltonianAdapter<'_, H> {
    #[inline]
    fn eval(&self, p: f64) -> f64 {
        self.0.evaluate(p)
    }

    #[inline]
    fn deriv(&self, p: f64) -> f64 {
        self.0.derivative(p)
    }
}

/// Returns the `SolverWorkspace<f64>` capacity required by [`FixedPointSolver::solve_into`].
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn mfg_workspace_len(config: &MFGConfig) -> usize {
    mfg_1d_workspace_len(config.grid_points.get(), config.time_steps.get())
}

/// Strategy trait for solving coupled Hamilton-Jacobi-Bellman / Fokker-Planck Mean Field Games.
///
/// $$
/// -\partial_t u + H(x, \nabla u) - \nu \Delta u = F(x, m), \qquad \partial_t m - \nabla \cdot (m \nabla_p H) - \nu \Delta m = 0
/// $$
///
/// Reference: Lasry & Lions (2007), DOI: 10.1007/s11537-007-0657-8;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Implementations must preserve non-negativity and unit mass $\int m(x, t)\,dx = 1$ across forward sweeps
/// while respecting parabolic/hyperbolic CFL bounds via monotone flux splitting.
///
/// # Domain of Validity
/// Valid for viscosity $\nu \ge 0$, positive horizon $T > 0$, and $N_x \ge 2$ spatial nodes.
pub trait MFGSolver {
    /// Solves the Mean Field Game coupled system.
    ///
    /// # Parameters
    /// - `config`: The physical and grid configuration.
    /// - `cost_function`: $F(x, m)$ - Running cost.
    /// - `terminal_cost`: $G(x, m)$ - Terminal cost.
    /// - `initial_distribution`: $m_0(x)$ - Initial population distribution.
    ///
    /// # Returns
    /// Tuple `(u, m)` containing the value function and distribution matrices.
    #[verified_engine::verified]
    fn solve(
        &self,
        config: &MFGConfig,
        cost_function: &impl Fn(Position, Density) -> f64,
        terminal_cost: &impl Fn(Position, Density) -> f64,
        initial_distribution: &impl Fn(Position) -> f64,
    ) -> (DMatrix<f64>, DMatrix<f64>);
}

/// Mann-damped monotone fixed-point solver for 1D Mean Field Games routed through [`engine_opt::mfg`].
///
/// Alternates backward Engquist-Osher HJB sweeps ($u^n$ from $u^{n+1}$) and forward conservative
/// donor-cell Fokker-Planck sweeps ($m^{n+1}$ from $m^n$) with Mann damping $m^{(k+1)} = (1 - \alpha)m^{(k)} + \alpha \tilde{m}$.
///
/// Reference: Lasry & Lions (2007), DOI: 10.1007/s11537-007-0657-8; Achdou & Capuzzo-Dolcetta (2010),
/// DOI: 10.1137/090758477; [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Adaptive CFL substepping paired with `H(D^+ u_i, D^- u_i)` and donor-cell fluxes preserves non-negativity,
/// continuous unit mass $\sum_i m_i \Delta x = 1$, and inviscid stability down to $\nu = 0$.
///
/// # Domain of Validity
/// Valid on bounded 1D intervals $x_{\min} < x_{\max}$ with $\nu \ge 0$, $\alpha \in (0, 1]$, and convex Hamiltonians $H(p)$.
pub struct FixedPointSolver<H: Hamiltonian> {
    /// Maximum number of backward-HJB / forward-FP Mann iterations to execute.
    pub iterations: usize,
    /// Hamiltonian strategy $H(p)$ governing optimal agent drift $-H'(p)$.
    pub hamiltonian: H,
    /// Mann fixed-point damping weight $\alpha \in (0, 1]$.
    pub mann_alpha: f64,
    /// $L^1$ density residual convergence threshold.
    pub l1_tolerance: f64,
}

impl FixedPointSolver<QuadraticHamiltonian> {
    /// Creates a new solver with a default Quadratic Hamiltonian ($H = p^2/2$) and Mann damping $\alpha = 0.5$.
    #[verified_engine::verified]
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            hamiltonian: QuadraticHamiltonian::default(),
            mann_alpha: DEFAULT_MANN_DAMPING,
            l1_tolerance: DEFAULT_L1_TOLERANCE,
        }
    }
}

impl<H: Hamiltonian> FixedPointSolver<H> {
    /// Creates a new solver with a custom Hamiltonian strategy and Mann damping $\alpha = 0.5$.
    #[verified_engine::verified]
    pub fn new_with_hamiltonian(iterations: usize, hamiltonian: H) -> Self {
        Self {
            iterations,
            hamiltonian,
            mann_alpha: DEFAULT_MANN_DAMPING,
            l1_tolerance: DEFAULT_L1_TOLERANCE,
        }
    }

    /// Configures Mann fixed-point damping $\alpha \in (0, 1]$ and $L^1$ residual stopping tolerance.
    #[verified_engine::verified]
    pub fn with_mann_damping(mut self, alpha: f64, l1_tolerance: f64) -> Self {
        self.mann_alpha = alpha;
        self.l1_tolerance = l1_tolerance;
        self
    }

    /// Solves the 1D Mean Field Game with zero heap allocations into column-major `u_grid` and `m_grid` slices.
    ///
    /// $$
    /// m_i^{(k+1)} = (1 - \alpha)m_i^{(k)} + \alpha \tilde{m}_i^{(k)}, \qquad \sum_{i=0}^{N_x - 1} m_i^n \Delta x = 1
    /// $$
    ///
    /// Reference: Achdou & Capuzzo-Dolcetta (2010), DOI: 10.1137/090758477;
    /// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
    ///
    /// # Stability
    /// Slices all HJB and Fokker-Planck scratch arrays from `ws` without touching the global allocator.
    ///
    /// # Domain of Validity
    /// Requires `u_grid.len() == m_grid.len() == nx * (nt + 1)` and `ws.len() >= mfg_workspace_len(config)`.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn solve_into(
        &self,
        config: &MFGConfig,
        cost_function: &impl Fn(Position, Density) -> f64,
        terminal_cost: &impl Fn(Position, Density) -> f64,
        initial_distribution: &impl Fn(Position) -> f64,
        u_grid: &mut [f64],
        m_grid: &mut [f64],
        ws: &mut SolverWorkspace<'_, f64>,
    ) -> Result<MfgStats, OptError> {
        let mfg_cfg = MfgConfig1D {
            space_min: config.space_min,
            dx: config.dx,
            dt: config.dt,
            viscosity: config.viscosity,
            nx: config.grid_points.get(),
            nt: config.time_steps.get(),
            max_iterations: self.iterations,
            mann_alpha: self.mann_alpha,
            l1_tolerance: self.l1_tolerance,
        };
        let adapter = DomainHamiltonianAdapter(&self.hamiltonian);
        let run_fn = |x: f64, m: f64| cost_function(Position(x), Density(m));
        let term_fn = |x: f64, m: f64| terminal_cost(Position(x), Density(m));
        let init_fn = |x: f64| initial_distribution(Position(x));
        solve_mfg_1d(
            &mfg_cfg, &adapter, &run_fn, &term_fn, &init_fn, u_grid, m_grid, ws,
        )
    }
}

impl<H: Hamiltonian> MFGSolver for FixedPointSolver<H> {
    #[verified_engine::verified]
    fn solve(
        &self,
        config: &MFGConfig,
        cost_function: &impl Fn(Position, Density) -> f64,
        terminal_cost: &impl Fn(Position, Density) -> f64,
        initial_distribution: &impl Fn(Position) -> f64,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nx = config.grid_points.get();
        let cols = config.time_steps.get() + 1;
        let mut u = DMatrix::zeros(nx, cols);
        let mut m = DMatrix::zeros(nx, cols);
        let mut ws_buf = vec![0.0_f64; mfg_workspace_len(config)];
        let mut ws = SolverWorkspace::new(&mut ws_buf);
        let _ = self.solve_into(
            config,
            cost_function,
            terminal_cost,
            initial_distribution,
            u.as_mut_slice(),
            m.as_mut_slice(),
            &mut ws,
        );
        (u, m)
    }
}
