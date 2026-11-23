//! Zero-allocation Poincaré surface-of-section detector with cubic Hermite interpolation.
//!
//! Detects transverse hyperplane intersections $q_k = c$ between consecutive solver states
//! $(y_0, \dot{y}_0)$ and $(y_1, \dot{y}_1)$ using safeguarded Newton-bisection on the
//! $O(\Delta t^4)$ cubic Hermite interpolant without dynamic heap allocation.

// [cite:chaos_theory]

use pure_math::pure_math::analysis::ode::ArrayState;

/// Transverse directional filter for Poincaré hyperplane crossings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum CrossingDirection {
    /// Trigger when the section coordinate crosses from $\le c$ to $> c$ with positive slope.
    #[default]
    Positive,
    /// Trigger when the section coordinate crosses from $\ge c$ to $< c$ with negative slope.
    Negative,
    /// Trigger on transverse crossings in either direction.
    Both,
}

/// Interpolated phase-space state at an exact Poincaré hyperplane intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareCrossing<const N: usize> {
    /// Interpolated crossing timestamp $t^* = t_0 + s^* \Delta t$.
    pub time: f64,
    /// Normalized substep fraction $s^* \in [0, 1]$.
    pub fraction: f64,
    /// Projected horizontal section coordinate $q_{\text{plot}}(t^*)$.
    pub q_sec: f64,
    /// Projected conjugate momentum coordinate $p_{\text{plot}}(t^*)$.
    pub p_sec: f64,
    /// Full cubic-Hermite interpolated state vector $y(t^*)$.
    pub state: ArrayState<N>,
}

/// Zero-allocation Poincaré surface-of-section hyperplane specification.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::henon_heiles::HenonHeiles;
/// use math_explorer::physics::chaos::poincare::PoincareSection;
/// use math_explorer::pure_math::analysis::ode::{ArrayState, OdeSystem, Yoshida4};
///
/// let hh = HenonHeiles::default();
/// let section = PoincareSection::<4>::henon_heiles();
/// let mut y0 = hh.seed_on_poincare_section(0.125, 0.1, 0.0).unwrap();
/// let mut y1 = y0;
/// let mut dy0 = ArrayState::zeros();
/// let mut dy1 = ArrayState::zeros();
/// let mut solver = Yoshida4::new(&ArrayState([0.0, 0.0]));
/// let mut crossings = 0_usize;
/// for step in 0..800 {
///     hh.derivative_in_place(0.0, &y0, &mut dy0);
///     solver.step_separable(&hh, &mut y1, 0.02);
///     hh.derivative_in_place(0.0, &y1, &mut dy1);
///     if section.detect_crossing(step as f64 * 0.02, 0.02, &y0, &y1, &dy0, &dy1).is_some() {
///         crossings += 1;
///     }
///     y0 = y1;
/// }
/// assert!(crossings >= 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareSection<const N: usize> {
    /// Index of the state coordinate defining the hyperplane $y[\text{plane\_index}] = \text{plane\_value}$.
    pub plane_index: usize,
    /// Target hyperplane scalar value.
    pub plane_value: f64,
    /// Required crossing orientation.
    pub direction: CrossingDirection,
    /// State index plotted on the horizontal Poincaré axis $q$.
    pub q_plot_index: usize,
    /// State index plotted on the vertical Poincaré axis $p$.
    pub p_plot_index: usize,
    /// Whether the hyperplane and plotted $q$ angles wrap $2\pi$-periodically onto $(-\pi, \pi]$.
    pub periodic_angle: bool,
}

#[verified_engine::verified]
fn wrap_angle_pi(angle: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

#[verified_engine::verified]
fn cubic_hermite_scalar(s: f64, dt: f64, v0: f64, v1: f64, d0: f64, d1: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * v0 + h10 * (dt * d0) + h01 * v1 + h11 * (dt * d1)
}

#[verified_engine::verified]
fn cubic_hermite_slope(s: f64, dt: f64, v0: f64, v1: f64, d0: f64, d1: f64) -> f64 {
    let s2 = s * s;
    let dh00 = 6.0 * s2 - 6.0 * s;
    let dh10 = 3.0 * s2 - 4.0 * s + 1.0;
    let dh01 = -6.0 * s2 + 6.0 * s;
    let dh11 = 3.0 * s2 - 2.0 * s;
    dh00 * v0 + dh10 * (dt * d0) + dh01 * v1 + dh11 * (dt * d1)
}

#[verified_engine::verified]
fn solve_hermite_root(dt: f64, g0: f64, g1: f64, dg0: f64, dg1: f64) -> f64 {
    if (g1 - g0).abs() <= 1e-15 {
        return 0.5;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut g_lo = g0;
    let mut s = (-g0 / (g1 - g0)).clamp(0.0, 1.0);

    for _ in 0..16 {
        let f = cubic_hermite_scalar(s, dt, g0, g1, dg0, dg1);
        if f.abs() <= 1e-14 {
            break;
        }
        if (f > 0.0) == (g_lo > 0.0) {
            lo = s;
            g_lo = f;
        } else {
            hi = s;
        }
        let df = cubic_hermite_slope(s, dt, g0, g1, dg0, dg1);
        let candidate = if df.abs() > 1e-14 { s - f / df } else { -1.0 };
        s = if candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    s.clamp(0.0, 1.0)
}

impl PoincareSection<4> {
    /// Canonical Hénon-Heiles Poincaré section ($x = 0, p_x > 0$, plotting $(y, p_y)$).
    #[must_use]
    #[verified_engine::verified]
    pub fn henon_heiles() -> Self {
        Self {
            plane_index: 0,
            plane_value: 0.0,
            direction: CrossingDirection::Positive,
            q_plot_index: 1,
            p_plot_index: 3,
            periodic_angle: false,
        }
    }

    /// Canonical Double Pendulum Poincaré section ($\theta_1 \equiv 0, \dot{\theta}_1 > 0$, plotting $(\theta_2, p_2)$).
    #[must_use]
    #[verified_engine::verified]
    pub fn double_pendulum() -> Self {
        Self {
            plane_index: 0,
            plane_value: 0.0,
            direction: CrossingDirection::Positive,
            q_plot_index: 1,
            p_plot_index: 3,
            periodic_angle: true,
        }
    }
}

impl PoincareSection<12> {
    /// Planar 3-body Figure-8 Poincaré section ($y_3 = 0, \dot{y}_3 < 0$, plotting $(x_1, p_{x1})$).
    #[must_use]
    #[verified_engine::verified]
    pub fn figure_eight() -> Self {
        Self {
            plane_index: 5,
            plane_value: 0.0,
            direction: CrossingDirection::Negative,
            q_plot_index: 0,
            p_plot_index: 6,
            periodic_angle: false,
        }
    }
}

impl<const N: usize> PoincareSection<N> {
    /// Detects and cubic-Hermite interpolates a hyperplane crossing across $[t_0, t_0 + \Delta t]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn detect_crossing(
        &self,
        t0: f64,
        dt: f64,
        y0: &ArrayState<N>,
        y1: &ArrayState<N>,
        dy0: &ArrayState<N>,
        dy1: &ArrayState<N>,
    ) -> Option<PoincareCrossing<N>> {
        if self.plane_index >= N || self.q_plot_index >= N || self.p_plot_index >= N || dt <= 0.0 {
            return None;
        }
        let raw0 = y0.0[self.plane_index] - self.plane_value;
        let raw1 = y1.0[self.plane_index] - self.plane_value;
        let (g0, g1) = if self.periodic_angle {
            let w0 = wrap_angle_pi(raw0);
            let w1 = wrap_angle_pi(raw1);
            if (w1 - w0).abs() > std::f64::consts::PI {
                return None;
            }
            (w0, w1)
        } else {
            (raw0, raw1)
        };

        let pos_cross = g0 < 0.0 && g1 >= 0.0;
        let neg_cross = g0 > 0.0 && g1 <= 0.0;
        let matches_dir = match self.direction {
            CrossingDirection::Positive => pos_cross,
            CrossingDirection::Negative => neg_cross,
            CrossingDirection::Both => pos_cross || neg_cross,
        };
        if !matches_dir {
            return None;
        }

        let dg0 = dy0.0[self.plane_index];
        let dg1 = dy1.0[self.plane_index];
        let s = solve_hermite_root(dt, g0, g1, dg0, dg1);
        let state = self.interpolate_state(s, dt, y0, y1, dy0, dy1);
        let q_raw = state.0[self.q_plot_index];
        let q_sec = if self.periodic_angle {
            wrap_angle_pi(q_raw)
        } else {
            q_raw
        };
        Some(PoincareCrossing {
            time: t0 + s * dt,
            fraction: s,
            q_sec,
            p_sec: state.0[self.p_plot_index],
            state,
        })
    }

    #[verified_engine::verified]
    fn interpolate_state(
        &self,
        s: f64,
        dt: f64,
        y0: &ArrayState<N>,
        y1: &ArrayState<N>,
        dy0: &ArrayState<N>,
        dy1: &ArrayState<N>,
    ) -> ArrayState<N> {
        let mut interp = ArrayState::zeros();
        for i in 0..N {
            interp.0[i] = cubic_hermite_scalar(s, dt, y0.0[i], y1.0[i], dy0.0[i], dy1.0[i]);
        }
        interp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_cubic_hermite_harmonic_crossing_fourth_order_accuracy() {
        let section = PoincareSection::<2> {
            plane_index: 0,
            plane_value: 0.0,
            direction: CrossingDirection::Positive,
            q_plot_index: 0,
            p_plot_index: 1,
            periodic_angle: false,
        };
        // Exact trajectory: q(t) = sin(t), p(t) = cos(t), crossing q = 0 upward at t = 0
        let t0: f64 = -0.013;
        let dt: f64 = 0.04;
        let t1: f64 = t0 + dt;
        let y0 = ArrayState([t0.sin(), t0.cos()]);
        let y1 = ArrayState([t1.sin(), t1.cos()]);
        let dy0 = ArrayState([t0.cos(), -t0.sin()]);
        let dy1 = ArrayState([t1.cos(), -t1.sin()]);

        let crossing = section
            .detect_crossing(t0, dt, &y0, &y1, &dy0, &dy1)
            .expect("crossing detected");
        assert!(
            crossing.time.abs() < 1e-8,
            "Cubic Hermite crossing time error too large: {}",
            crossing.time
        );
        assert!(
            (crossing.p_sec - 1.0).abs() < 1e-7,
            "Interpolated momentum should equal 1.0, got {}",
            crossing.p_sec
        );
    }
}
