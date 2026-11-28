//! # Mean Field Games (MFG)
//!
//! MFGs model the limit of $N \to \infty$ players, where individual interactions are replaced by
//! an interaction with the "mean field" (the aggregate population). This leads to a system of two coupled PDEs:
//!
//! 1.  **Hamilton-Jacobi-Bellman (HJB)**: Backward equation. Determines the optimal value function $u(x,t)$ for a representative agent.
//! 2.  **Fokker-Planck (FP)**: Forward equation. Describes the evolution of the population density $m(x,t)$ under the optimal control.
//!
//! ##  Quick Start
//!
//! Solve a standard 1D MFG where agents try to minimize travel cost while avoiding congestion.
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::game_theory::mean_field::{MFGConfig, FixedPointSolver, MFGSolver};
//!
//! // 1. Configuration
//! // Viscosity (noise) = 0.1, Time Horizon = 1.0
//! // Grid: 50 spatial points, 100 time steps. Range: [-2.0, 2.0]
//! use std::num::NonZeroUsize;
//! use math_explorer::applied::game_theory::mean_field::types::MFGConfigBuilder;
//! let config = MFGConfigBuilder::new()
//!     .viscosity(0.1)
//!     .time_horizon(1.0)
//!     .grid_points(NonZeroUsize::new(50).unwrap())
//!     .time_steps(NonZeroUsize::new(100).unwrap())
//!     .space_bounds(-2.0, 2.0)
//!     .unwrap()
//!     .build()
//!     .unwrap();
//!
//! // 2. Define Costs and Initial Distribution
//! // Running Cost: F(x, m) = m + x^2 (Agents dislike crowds + prefer origin)
//! use math_explorer::applied::game_theory::mean_field::types::{Density, Position};
//! let cost_fn = |p: Position, d: Density| -> f64 { d.0 + p.0 * p.0 };
//!
//! // Terminal Cost: G(x) = x^2 (Agents want to be at origin at T)
//! let term_fn = |p: Position, _d: Density| -> f64 { p.0 * p.0 };
//!
//! // Initial Distribution: Gaussian centered at origin
//! let init_dist = |p: Position| -> f64 { (-p.0 * p.0 * 5.0).exp() };
//!
//! // 3. Solve using Fixed Point Iteration
//! let solver = FixedPointSolver::new(5); // 5 Iterations
//! let (u, m) = solver.solve(&config, &cost_fn, &term_fn, &init_dist);
//!
//! println!("Value at origin at t=0: {:.4}", u[(25, 0)]);
//! ```

pub mod physics;
/// Coupled backward-HJB and forward-Fokker-Planck fixed-point PDE solvers.
pub mod solver;
/// Grid discretization, viscosity, and boundary configuration types for 1D Mean Field Games.
pub mod types;

pub use physics::{Hamiltonian, QuadraticHamiltonian};
pub use solver::{FixedPointSolver, MFGSolver, mfg_workspace_len};
pub use types::{Density, MFGConfig, MFGConfigBuilder, Position};

#[cfg(test)]
mod tests {
    use super::*;
    use engine_opt::ScalarWorkspace;
    use std::num::NonZeroUsize;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_mfg_with_custom_hamiltonian() {
        let config = MFGConfigBuilder::new()
            .viscosity(0.1)
            .time_horizon(1.0)
            .grid_points(NonZeroUsize::new(50).unwrap())
            .time_steps(NonZeroUsize::new(100).unwrap())
            .space_bounds(-2.0, 2.0)
            .unwrap()
            .build()
            .unwrap();
        let hamiltonian = QuadraticHamiltonian::new(2.0);
        let solver = FixedPointSolver::new_with_hamiltonian(5, hamiltonian);

        let cost_fn = |p: Position, d: Density| -> f64 { d.0 + p.0 * p.0 };
        let term_fn = |p: Position, _d: Density| -> f64 { p.0 * p.0 };
        let init_dist = |p: Position| -> f64 { (-p.0 * p.0 * 5.0).exp() };

        let (u, m) = solver.solve(&config, &cost_fn, &term_fn, &init_dist);

        assert_eq!(u.nrows(), 50);
        assert_eq!(u.ncols(), 101);
        assert!((m.column(100).sum() * config.dx - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_mfg_zero_alloc_uniform_compression_dx_mass_and_inviscid_limit() {
        // 1. Verify continuous mass sum(m_i) * dx == 1.0 within 1e-12 across Nx in 17..=65,
        //    uniform m_0(x) = 0.5 compression, zero heap allocations, and viscosity = 0.0 stability.
        for nx in (17..=65).step_by(8) {
            let nt = 32usize;
            let cfg = MFGConfigBuilder::new()
                .viscosity(0.0)
                .time_horizon(0.8)
                .grid_points(NonZeroUsize::new(nx).unwrap())
                .time_steps(NonZeroUsize::new(nt).unwrap())
                .space_bounds(-1.0, 1.0)
                .unwrap()
                .build()
                .unwrap();

            let mut u_buf = vec![0.0_f64; nx * (nt + 1)];
            let mut m_buf = vec![0.0_f64; nx * (nt + 1)];
            let mut ws_buf = vec![0.0_f64; mfg_workspace_len(&cfg)];
            let solver = FixedPointSolver::new(12).with_mann_damping(0.5, 1.0e-6);

            let running = |p: Position, d: Density| 0.5 * p.0 * p.0 + 0.15 * d.0;
            let terminal = |p: Position, _d: Density| 0.8 * p.0 * p.0;
            let uniform_m0 = |_p: Position| 0.5;

            let (stats, _) = {
                let _guard = crate::VERIFIED_TEST_LOCK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                VerifiedEngine::run_verified(|| {
                    let mut ws = ScalarWorkspace::new(&mut ws_buf);
                    solver
                        .solve_into(
                            &cfg,
                            &running,
                            &terminal,
                            &uniform_m0,
                            &mut u_buf,
                            &mut m_buf,
                            &mut ws,
                        )
                        .expect("inviscid zero-alloc MFG must succeed")
                })
            };

            assert!(stats.iterations >= 1);
            assert!(u_buf.iter().all(|v| v.is_finite()));
            assert!(m_buf.iter().all(|v| v.is_finite() && *v >= 0.0));
            for col in 0..=nt {
                let mass = m_buf[col * nx..(col + 1) * nx].iter().sum::<f64>() * cfg.dx;
                assert!(
                    (mass - 1.0).abs() < 1.0e-12,
                    "Nx={nx} col={col}: expected mass 1.0, got {mass}"
                );
            }
            let m0_center = m_buf[nx / 2];
            let m_end_center = m_buf[nt * nx + nx / 2];
            let m_end_edge = m_buf[nt * nx];
            assert!(
                m_end_center > m0_center * 1.25 && m_end_center > m_end_edge,
                "Nx={nx}: uniform m0={m0_center} failed to compress to center ({m_end_center} vs edge {m_end_edge})"
            );
        }
    }
}

// [cite:graph_parameters_rust]
