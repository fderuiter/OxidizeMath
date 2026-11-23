//! Zero-allocation ensemble seeding helpers for double-pendulum fans and 3-body perturbations.

// [cite:chaos_theory]

use super::DoublePendulum;
use crate::physics::astrophysics::nbody::FigureEightThreeBody;
use pure_math::pure_math::analysis::ode::ArrayState;

/// Populates `out` with a symmetric fan of nearby canonical double-pendulum states on the
/// exact constant-energy shell $H = \text{energy}$ around `(center_theta2, center_p2)`.
///
/// Returns the number of valid states written into `out` (`0..=out.len()`) with zero heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn seed_double_pendulum_fan(
    pendulum: &DoublePendulum,
    energy: f64,
    center_theta2: f64,
    center_p2: f64,
    angular_spread: f64,
    out: &mut [ArrayState<4>],
) -> usize {
    if out.is_empty()
        || !energy.is_finite()
        || !center_theta2.is_finite()
        || !center_p2.is_finite()
        || !angular_spread.is_finite()
    {
        return 0;
    }
    let spread = angular_spread.abs().clamp(0.0, 0.5);
    let total = out.len();
    let mut written = 0_usize;

    for idx in 0..total {
        let offset = if total == 1 {
            0.0
        } else {
            let frac = (idx as f64) / ((total - 1) as f64) - 0.5;
            2.0 * frac * spread
        };
        if let Some(state) =
            pendulum.seed_on_poincare_section(energy, center_theta2 + offset, center_p2)
        {
            out[written] = state;
            written += 1;
        }
    }
    written
}

/// Constructs a planar 3-body Chenciner-Montgomery Figure-8 state with a symmetric
/// zero-linear-momentum perturbation $(\delta p_x, \delta p_y)$.
///
/// Applies $+(\delta p_x, \delta p_y)$ to bodies 1 and 2 and $-2(\delta p_x, \delta p_y)$ to body 3,
/// keeping the center of mass strictly stationary at the origin.
#[must_use]
#[verified_engine::verified]
pub fn perturb_figure_eight_momentum(delta_px: f64, delta_py: f64) -> ArrayState<12> {
    let mut state = FigureEightThreeBody::figure_eight_initial_state();
    let dpx = if delta_px.is_finite() {
        delta_px.clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let dpy = if delta_py.is_finite() {
        delta_py.clamp(-0.5, 0.5)
    } else {
        0.0
    };
    state.0[6] += dpx;
    state.0[7] += dpy;
    state.0[8] += dpx;
    state.0[9] += dpy;
    state.0[10] -= 2.0 * dpx;
    state.0[11] -= 2.0 * dpy;
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::{
        HamiltonianSystem, ImplicitMidpoint, RungeKutta4, Solver,
    };

    #[test]
    #[verified_engine::verified]
    fn test_double_pendulum_fan_and_implicit_midpoint_vs_rk4() {
        let pendulum = DoublePendulum::default();
        let target_e = 30.0;
        let mut fan = [ArrayState::<4>::zeros(); 6];
        let count = seed_double_pendulum_fan(&pendulum, target_e, 0.9, 0.0, 1e-3, &mut fan);
        assert_eq!(count, 6);
        for s in &fan {
            assert!((pendulum.hamiltonian(s) - target_e).abs() < 1e-11);
        }

        let dt = 0.01;
        let mut s_mid = fan[0];
        let mut s_rk4 = fan[0];
        let mut midpoint = ImplicitMidpoint::new(&s_mid);
        let mut rk4 = RungeKutta4::new(&s_rk4);
        for step in 0..6_000 {
            let t = step as f64 * dt;
            midpoint.step(&pendulum, t, &mut s_mid, dt);
            rk4.step(&pendulum, t, &mut s_rk4, dt);
        }
        let err_mid = pendulum.relative_energy_error(&s_mid, target_e);
        let err_rk4 = pendulum.relative_energy_error(&s_rk4, target_e);
        assert!(
            err_mid < 2e-3,
            "ImplicitMidpoint bounded oscillation: {err_mid}"
        );
        assert!(
            err_mid < err_rk4 || err_mid < 1e-3,
            "ImplicitMidpoint ({err_mid}) must stay tightly bounded vs RK4 ({err_rk4})"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_perturbed_figure_eight_preserves_zero_total_momentum() {
        let sys = FigureEightThreeBody::default();
        let perturbed = perturb_figure_eight_momentum(0.03, -0.02);
        let p_tot = sys.total_linear_momentum(&perturbed);
        assert!(p_tot[0].abs() < 1e-14 && p_tot[1].abs() < 1e-14);
    }
}
