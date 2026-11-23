#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompiledBifurcationJob, CUSTOM_ESCAPE_GUARD, CUSTOM_R_BOUND};
use crate::framework::MathExpressionInputBar;
use math_explorer::physics::chaos::logistic::LogisticMap;

/// Number of discrete map iterations displayed in the cobweb and orbit inspectors.
pub const COBWEB_ORBIT_STEPS: usize = 50;

/// Synchronized curve, cobweb staircase, and $x_n$ orbit series for a chosen $r$ and $x_0$.
pub struct CobwebOrbitSeries {
    /// Sampled map curve $y = f_r(x)$.
    pub parabola: Vec<[f64; 2]>,
    /// Alternating vertical/horizontal cobweb staircase segments.
    pub cobweb: Vec<[f64; 2]>,
    /// Discrete iteration time series $(n, x_n)$.
    pub orbit: Vec<[f64; 2]>,
}

/// Builds synchronized parabola, cobweb staircase, and $x_n$ orbit series from [`LogisticMap`].
#[must_use]
pub fn build_cobweb_and_orbit(r: f64, x0: f64, n_steps: usize) -> CobwebOrbitSeries {
    let safe_r = if r.is_finite() {
        r.clamp(0.0, 4.0)
    } else {
        3.5
    };
    let safe_x0 = if x0.is_finite() {
        x0.clamp(0.001, 0.999)
    } else {
        0.1
    };
    let mut parabola = Vec::with_capacity(101);
    for idx in 0..=100 {
        let x = idx as f64 / 100.0;
        parabola.push([x, safe_r * x * (1.0 - x)]);
    }

    let mut map = LogisticMap::new(safe_r, safe_x0);
    let mut cobweb = Vec::with_capacity(1 + n_steps * 2);
    let mut orbit = Vec::with_capacity(n_steps + 1);
    let mut x_curr = safe_x0;
    cobweb.push([x_curr, 0.0]);
    orbit.push([0.0, x_curr]);

    for step in 1..=n_steps {
        let x_next = map.next().unwrap_or(0.0);
        if !x_next.is_finite() {
            break;
        }
        let clamped_next = x_next.clamp(0.0, 1.0);
        cobweb.push([x_curr, clamped_next]);
        cobweb.push([clamped_next, clamped_next]);
        orbit.push([step as f64, clamped_next]);
        x_curr = clamped_next;
    }

    CobwebOrbitSeries {
        parabola,
        cobweb,
        orbit,
    }
}

/// Builds synchronized map curve, cobweb staircase, and $x_n$ orbit series for a custom 1D map.
#[must_use]
pub fn build_custom_cobweb_and_orbit(
    r: f64,
    x0: f64,
    n_steps: usize,
    bar: &MathExpressionInputBar,
) -> CobwebOrbitSeries {
    let job = CompiledBifurcationJob::from_bar(bar, x0);
    let safe_r = if r.is_finite() {
        r.clamp(-CUSTOM_R_BOUND, CUSTOM_R_BOUND)
    } else {
        1.0
    };
    let safe_x0 = if x0.is_finite() {
        x0.clamp(-10.0, 10.0)
    } else {
        0.5
    };
    let mut cobweb = Vec::with_capacity(1 + n_steps * 2);
    let mut orbit = Vec::with_capacity(n_steps + 1);
    let mut x_curr = safe_x0;
    let mut lo = safe_x0.min(-1.0);
    let mut hi = safe_x0.max(1.0);
    cobweb.push([x_curr, 0.0]);
    orbit.push([0.0, x_curr]);

    for step in 1..=n_steps {
        let x_next = job.eval(safe_r, x_curr);
        if !x_next.is_finite() || x_next.abs() > CUSTOM_ESCAPE_GUARD {
            break;
        }
        lo = lo.min(x_next);
        hi = hi.max(x_next);
        cobweb.push([x_curr, x_next]);
        cobweb.push([x_next, x_next]);
        orbit.push([step as f64, x_next]);
        x_curr = x_next;
    }

    let mut parabola = Vec::with_capacity(101);
    let span = (hi - lo).max(0.1);
    for idx in 0..=100 {
        let x = lo + (idx as f64 / 100.0) * span;
        let y = job.eval(safe_r, x);
        if y.is_finite() && y.abs() <= CUSTOM_ESCAPE_GUARD {
            parabola.push([x, y]);
        }
    }
    CobwebOrbitSeries {
        parabola,
        cobweb,
        orbit,
    }
}
