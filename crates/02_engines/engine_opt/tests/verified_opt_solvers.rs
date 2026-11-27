//! Zero-allocation verification tests for `engine_opt::{mfg, ilqr, mpc, dp, nash}`.

use engine_opt::dp::{
    DenseTabularMdp, HowardDpConfig, dp_finite_workspace_len, dp_infinite_workspace_len,
    solve_finite_horizon_dp, solve_infinite_horizon_dp,
};
use engine_opt::ilqr::{
    ControlModel, IlqrConfig, QuadraticBoxCost, ilqr_workspace_len, solve_ilqr,
};
use engine_opt::mfg::{MfgConfig1D, QuadraticMfgHamiltonian, mfg_1d_workspace_len, solve_mfg_1d};
use engine_opt::mpc::{MpcConfig, MpcProblem, mpc_workspace_len, solve_mpc};
use engine_opt::nash::{
    PolymatrixConfig, nash_bimatrix_workspace_len, nash_polymatrix_workspace_len,
    solve_bimatrix_lemke_howson, solve_polymatrix_regret_matching,
};
use engine_opt::{OptError, SolverWorkspace};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct DoubleIntegrator {
    dt: f64,
}

impl ControlModel for DoubleIntegrator {
    fn state_dim(&self) -> usize {
        2
    }
    fn control_dim(&self) -> usize {
        1
    }
    fn step_and_linearize(
        &self,
        _step: usize,
        x: &[f64],
        u: &[f64],
        x_next: &mut [f64],
        fx: &mut [f64],
        fu: &mut [f64],
    ) -> Result<(), OptError> {
        let dt = self.dt;
        x_next[0] = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
        x_next[1] = x[1] + dt * u[0];
        fx.copy_from_slice(&[1.0, dt, 0.0, 1.0]);
        fu.copy_from_slice(&[0.5 * dt * dt, dt]);
        Ok(())
    }
}

#[test]
fn verified_mfg_zero_alloc_mass_and_inviscid_stability() {
    const NX: usize = 33;
    const NT: usize = 20;
    const WS_LEN: usize = mfg_1d_workspace_len(NX, NT);
    let mut u_grid = vec![0.0; NX * (NT + 1)];
    let mut m_grid = vec![0.0; NX * (NT + 1)];
    let mut ws_buf = vec![0.0; WS_LEN];

    let cfg = MfgConfig1D {
        space_min: 0.0,
        dx: 2.0 / ((NX - 1) as f64),
        dt: 0.04,
        viscosity: 0.0,
        nx: NX,
        nt: NT,
        max_iterations: 15,
        mann_alpha: 0.5,
        l1_tolerance: 1.0e-5,
    };
    let ham = QuadraticMfgHamiltonian::default();

    let (stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut ws_buf);
        solve_mfg_1d(
            &cfg,
            &ham,
            &|x, m| 0.5 * (x - 1.0) * (x - 1.0) + 0.2 * m,
            &|x, _m| 0.5 * (x - 1.0) * (x - 1.0),
            &|_x| 0.5,
            &mut u_grid,
            &mut m_grid,
            &mut ws,
        )
        .expect("mfg must succeed")
    });

    assert!(stats.iterations >= 1);
    assert!(u_grid.iter().all(|v| v.is_finite()));
    assert!(m_grid.iter().all(|v| v.is_finite() && *v >= 0.0));
    for n in 0..=NT {
        let mass = m_grid[n * NX..(n + 1) * NX].iter().sum::<f64>() * cfg.dx;
        assert!((mass - 1.0).abs() < 1.0e-12, "mass at step {n} was {mass}");
    }
    let m_final = &m_grid[NT * NX..(NT + 1) * NX];
    assert!(
        m_final[NX / 2] > m_grid[0],
        "uniform m0 must compress toward center"
    );
}

#[test]
fn verified_mpc_zero_alloc_respects_bounds() {
    const H: usize = 15;
    const WS_LEN: usize = mpc_workspace_len(H, 2, 1);
    let model = DoubleIntegrator { dt: 0.2 };
    let mut x_traj = [0.0; (H + 1) * 2];
    let mut u_traj = [0.0; H];
    let mut u0 = [0.0; 1];
    let mut ws_buf = [0.0; WS_LEN];

    let prob = MpcProblem {
        q_diag: &[4.0, 1.0],
        r_diag: &[0.1],
        qf_diag: &[25.0, 8.0],
        x_ref: &[1.0, 0.0],
        u_ref: &[0.0],
        u_min: &[-1.5],
        u_max: &[1.5],
        x_min: &[-0.2, -1.2],
        x_max: &[1.15, 1.2],
        state_penalty_weight: 200.0,
        state_margin: 0.02,
    };
    let mpc_cfg = MpcConfig {
        horizon: H,
        max_iterations: 30,
        tolerance: 1.0e-6,
        reg_init: 1.0e-6,
    };

    let (stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut ws_buf);
        solve_mpc(
            &model,
            &prob,
            &mpc_cfg,
            &[0.0, 0.0],
            &mut x_traj,
            &mut u_traj,
            &mut u0,
            &mut ws,
        )
        .expect("mpc must succeed")
    });

    assert!(stats.optimal_cost.is_finite());
    assert!(
        u_traj
            .iter()
            .all(|&u| (-1.5 - 1e-12..=1.5 + 1e-12).contains(&u))
    );
    assert!(stats.max_state_violation < 0.05);
}

#[test]
fn verified_ilqr_zero_alloc_converges() {
    const H: usize = 15;
    let model = DoubleIntegrator { dt: 0.2 };
    let mut x_traj = [0.0; (H + 1) * 2];
    let mut u_traj = [0.0; H];
    let mut ilqr_ws_buf = [0.0; ilqr_workspace_len(H, 2, 1)];
    let cost = QuadraticBoxCost {
        q_diag: &[2.0, 0.5],
        r_diag: &[0.1],
        qf_diag: &[20.0, 5.0],
        x_ref: &[1.0, 0.0],
        u_ref: &[0.0],
        x_min: None,
        x_max: None,
        bound_weight: 0.0,
    };
    let (ilqr_stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut ilqr_ws_buf);
        solve_ilqr(
            &model,
            &cost,
            &IlqrConfig {
                horizon: H,
                ..Default::default()
            },
            &[0.0, 0.0],
            Some(&[-2.0]),
            Some(&[2.0]),
            &mut x_traj,
            &mut u_traj,
            &mut ws,
        )
        .expect("ilqr must succeed")
    });
    assert!(ilqr_stats.converged);
}

#[test]
fn verified_howard_and_finite_dp_zero_alloc_high_discount() {
    const N: usize = 6;
    const A: usize = 2;
    let mut trans = [0.0; N * A * N];
    let mut rewards = [0.0; N * A];
    let term = [false, false, false, false, false, true];
    for s in 0..(N - 1) {
        let left = s.saturating_sub(1);
        let right = (s + 1).min(N - 1);
        trans[(s * A) * N + left] = 1.0;
        rewards[s * A] = -0.1;
        trans[(s * A + 1) * N + right] = 1.0;
        rewards[s * A + 1] = if right == N - 1 { 10.0 } else { -0.1 };
    }
    let mdp = DenseTabularMdp {
        n_states: N,
        n_actions: A,
        transitions: &trans,
        rewards: &rewards,
        terminal_mask: Some(&term),
    };

    let mut values = [0.0; N];
    let mut q_table = [0.0; N * A];
    let mut policy = [0usize; N];
    let mut ws_buf = [0.0; dp_infinite_workspace_len(N, A)];
    let cfg = HowardDpConfig {
        gamma: 0.999,
        max_sweeps: 20,
        tolerance: 1.0e-10,
    };

    let (stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut ws_buf);
        solve_infinite_horizon_dp(&mdp, &cfg, &mut values, &mut q_table, &mut policy, &mut ws)
            .expect("howard dp must succeed")
    });
    assert!(stats.converged);
    assert!(stats.sweeps < 10);
    assert!(stats.bellman_residual < 1.0e-10);
    for &act in &policy[..(N - 1)] {
        assert_eq!(act, 1);
    }

    let mut fh_values = [0.0; 9 * N];
    let mut fh_policy = [0usize; 8 * N];
    let mut fh_ws_buf = [0.0; dp_finite_workspace_len(N, A)];
    let _ = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut fh_ws_buf);
        solve_finite_horizon_dp(
            &mdp,
            8,
            0.99,
            &[0.0; N],
            &mut fh_values,
            &mut fh_policy,
            &mut ws,
        )
        .expect("finite dp must succeed")
    });
    assert!(fh_values[0] > 9.0);
}

#[test]
fn verified_nash_lemke_howson_and_polymatrix_zero_alloc() {
    // Rock-Paper-Scissors bimatrix game (unique mixed Nash = [1/3, 1/3, 1/3]).
    let a = [0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0];
    let b = [0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0];
    let mut sa = [0.0; 3];
    let mut sb = [0.0; 3];
    let mut lh_ws_buf = [0.0; nash_bimatrix_workspace_len(3, 3)];

    let (lh_stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut lh_ws_buf);
        solve_bimatrix_lemke_howson(&a, &b, 3, 3, &mut sa, &mut sb, &mut ws)
            .expect("lemke-howson must succeed")
    });
    assert!(lh_stats.max_regret < 1.0e-12);
    for &p in sa.iter().chain(sb.iter()) {
        assert!((p - 1.0 / 3.0).abs() < 1.0e-12);
    }

    // 3-player coordination / matching polymatrix game (2 actions per player, M = 6).
    let counts = [2usize, 2, 2];
    let mut poly = [0.0; 36];
    // Each player wants to match the next player's action modulo 3 (coordination -> pure/mixed fixed point).
    for (p, next_p) in [(0usize, 1usize), (1, 2), (2, 0)] {
        for act in 0..2 {
            poly[(2 * p + act) * 6 + (2 * next_p + act)] = 1.0;
        }
    }
    let mut avg_strat = [0.0; 6];
    let mut poly_ws_buf = [0.0; nash_polymatrix_workspace_len(6)];
    let (poly_stats, _) = VerifiedEngine::run_verified(|| {
        let mut ws = SolverWorkspace::new(&mut poly_ws_buf);
        solve_polymatrix_regret_matching(
            &counts,
            &poly,
            &PolymatrixConfig {
                max_iterations: 2_000,
                tolerance: 1.0e-5,
                cfr_plus: true,
            },
            &mut avg_strat,
            &mut ws,
        )
        .expect("polymatrix regret matching must succeed")
    });
    assert!(poly_stats.exploitability < 1.0e-4);
}
