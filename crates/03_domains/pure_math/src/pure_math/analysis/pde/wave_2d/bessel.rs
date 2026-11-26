//! Stack-only cylindrical Bessel $J_m(z)$ evaluators, zero tables, and drumhead antinode snapping.

// [cite:advanced_linear_algebra]

use std::f64::consts::PI;

use super::{Eigenmode2D, PinConstraint2D};

/// Maximum Bessel order $m$ in [`BESSEL_JN_ZEROS`].
pub(super) const MAX_BESSEL_ORDER: usize = 5;
/// Number of radial zeros $n$ tabulated per Bessel order in [`BESSEL_JN_ZEROS`].
pub(super) const MAX_BESSEL_RADIAL: usize = 4;

/// Tabulated roots $\alpha_{m, n}$ of $J_m(\alpha_{m, n}) = 0$ for $m \in 0..=5$, $n \in 1..=4$.
pub const BESSEL_JN_ZEROS: [[f64; MAX_BESSEL_RADIAL]; MAX_BESSEL_ORDER + 1] = [
    [
        2.404_825_557_695_773,
        5.520_078_110_286_311,
        8.653_727_912_911_012,
        11.791_534_439_014_282,
    ],
    [
        3.831_705_970_207_512_5,
        7.015_586_669_815_619,
        10.173_468_135_062_722,
        13.323_691_936_314_223,
    ],
    [
        5.135_622_301_840_683,
        8.417_244_140_399_864,
        11.619_841_172_149_059,
        14.795_951_782_351_262,
    ],
    [
        6.380_161_895_923_983,
        9.761_023_129_981_67,
        13.015_200_721_698_434,
        16.223_466_160_318_77,
    ],
    [
        7.588_342_434_503_804,
        11.064_709_488_501_185,
        14.372_536_671_617_59,
        17.615_966_049_804_832,
    ],
    [
        8.771_483_815_959_954,
        12.338_604_197_466_944,
        15.700_174_079_711_67,
        18.980_133_875_179_92,
    ],
];

/// First positive peak argument $j'_{m, 1}$ maximizing $|J_m(z)|$ for $m \in 0..=5$.
pub(super) const BESSEL_FIRST_PEAK: [f64; MAX_BESSEL_ORDER + 1] = [
    0.0,
    1.841_183_781_340_659_3,
    3.054_236_928_227_140_4,
    4.201_188_941_210_528,
    5.317_553_126_083_994,
    6.415_616_375_700_241,
];

/// Returns the $n$-th positive root $\alpha_{m, n}$ ($n \ge 1$) of $J_m(z) = 0$.
#[must_use]
#[verified_engine::verified]
pub fn bessel_jn_zero(order_m: usize, radial_n: usize) -> f64 {
    let n_idx = radial_n.max(1);
    if order_m <= MAX_BESSEL_ORDER && n_idx <= MAX_BESSEL_RADIAL {
        return BESSEL_JN_ZEROS[order_m][n_idx - 1];
    }
    let m_f = order_m as f64;
    let beta = ((n_idx as f64) + 0.5 * m_f - 0.25) * PI;
    let mu = 4.0 * m_f * m_f;
    beta - (mu - 1.0) / (8.0 * beta)
}

/// Evaluates the cylindrical Bessel function $J_m(z)$ ($m \ge 0$) on the stack with zero allocations.
#[must_use]
#[verified_engine::verified]
pub fn bessel_jn(order_m: usize, z: f64) -> f64 {
    if !z.is_finite() {
        return 0.0;
    }
    let x = z.abs();
    let sign = if z < 0.0 && (order_m % 2 == 1) {
        -1.0
    } else {
        1.0
    };
    if x < 1.0e-15 {
        return if order_m == 0 { 1.0 } else { 0.0 };
    }
    if x <= 14.0 {
        let half = 0.5 * x;
        let mut term = 1.0_f64;
        for idx in 1..=order_m {
            term *= half / (idx as f64);
        }
        let mut sum = term;
        let neg_half_sq = -(half * half);
        for k in 1..=36_usize {
            term *= neg_half_sq / ((k * (order_m + k)) as f64);
            sum += term;
            if term.abs() <= 1.0e-16 * sum.abs() {
                break;
            }
        }
        sign * sum
    } else {
        let m_f = order_m as f64;
        let mu = 4.0 * m_f * m_f;
        let eight_x = 8.0 * x;
        let a1 = (mu - 1.0) / eight_x;
        let a2 = a1 * (mu - 9.0) / (2.0 * eight_x);
        let p = 1.0 - a2;
        let q = a1;
        let chi = x - (0.5 * m_f + 0.25) * PI;
        let amp = (2.0 / (PI * x)).sqrt();
        sign * amp * (p * chi.cos() - q * chi.sin())
    }
}

#[verified_engine::verified]
pub(super) fn snap_bessel_antinode(
    mode: &Eigenmode2D,
    x_norm: f64,
    y_norm: f64,
    pins: PinConstraint2D,
    drumhead_radius: f64,
) -> (f64, f64) {
    let r_lim = drumhead_radius.clamp(0.05, 0.49);
    let alpha = bessel_jn_zero(mode.m, mode.n.max(1));
    let r_peak = if mode.m == 0 && pins != PinConstraint2D::CenterPin {
        if mode.n <= 1 {
            0.0
        } else {
            r_lim * (3.831_705_970_207_512_5 / alpha)
        }
    } else {
        let z_peak = BESSEL_FIRST_PEAK[mode.m.clamp(1, MAX_BESSEL_ORDER)];
        (r_lim * (z_peak / alpha)).min(0.88 * r_lim)
    };
    if r_peak <= 1.0e-6 {
        return (0.5, 0.5);
    }
    if mode.m == 0 {
        let dx = x_norm - 0.5;
        let dy = y_norm - 0.5;
        let theta = if dx * dx + dy * dy > 1.0e-8 {
            dy.atan2(dx)
        } else {
            0.0
        };
        return (0.5 + r_peak * theta.cos(), 0.5 + r_peak * theta.sin());
    }
    let lobes = 2 * mode.m;
    let mut best_pt = (0.5 + r_peak, 0.5);
    let mut best_dist = f64::INFINITY;
    for lobe in 0..lobes {
        let theta = (lobe as f64) * PI / (mode.m as f64);
        let cx = 0.5 + r_peak * theta.cos();
        let cy = 0.5 + r_peak * theta.sin();
        let dist = (cx - x_norm) * (cx - x_norm) + (cy - y_norm) * (cy - y_norm);
        if dist < best_dist {
            best_dist = dist;
            best_pt = (cx, cy);
        }
    }
    best_pt
}

#[verified_engine::verified]
pub(super) fn insert_sorted_mode(
    out: &mut [Eigenmode2D],
    count: &mut usize,
    candidate: Eigenmode2D,
) {
    if !candidate.frequency_hz.is_finite() || candidate.frequency_hz <= 1.0e-5 || out.is_empty() {
        return;
    }
    for existing in out.iter().take(*count) {
        if (existing.frequency_hz - candidate.frequency_hz).abs() < 1.0e-5
            && existing.kind == candidate.kind
        {
            return;
        }
    }
    let cap = out.len();
    let mut pos = *count;
    while pos > 0 && out[pos - 1].frequency_hz > candidate.frequency_hz {
        pos -= 1;
    }
    if *count < cap {
        let mut idx = *count;
        while idx > pos {
            out[idx] = out[idx - 1];
            idx -= 1;
        }
        out[pos] = candidate;
        *count += 1;
    } else if pos < cap {
        let mut idx = cap - 1;
        while idx > pos {
            out[idx] = out[idx - 1];
            idx -= 1;
        }
        out[pos] = candidate;
    }
}

#[verified_engine::verified]
pub(super) fn mode_survives_pins(
    mode: &Eigenmode2D,
    pins: PinConstraint2D,
    drumhead_radius: f64,
    tol: f64,
) -> bool {
    match pins {
        PinConstraint2D::None => true,
        PinConstraint2D::CenterPin => match mode.kind {
            super::PlateEigenmodeKind::SquareNeumannCosine
            | super::PlateEigenmodeKind::SquareChladniSymmetric => {
                !mode.m.is_multiple_of(2) && !mode.n.is_multiple_of(2)
            }
            super::PlateEigenmodeKind::SquareChladniAntisymmetric => {
                (!mode.m.is_multiple_of(2) && !mode.n.is_multiple_of(2))
                    || (mode.m.is_multiple_of(2) && mode.n.is_multiple_of(2))
            }
            super::PlateEigenmodeKind::SquareDirichletSine
            | super::PlateEigenmodeKind::SquareClampedBeam => {
                mode.m.is_multiple_of(2) && mode.n.is_multiple_of(2)
            }
            super::PlateEigenmodeKind::CircularBesselDrumhead => mode.m >= 1,
        },
        PinConstraint2D::QuadSymmetry => {
            super::evaluate_eigenmode_amplitude(mode, 0.25, 0.25, drumhead_radius).abs() <= tol
        }
        PinConstraint2D::EdgeMidpoints => {
            let ex = super::evaluate_eigenmode_amplitude(mode, 0.5, 0.04, drumhead_radius).abs();
            let ey = super::evaluate_eigenmode_amplitude(mode, 0.04, 0.5, drumhead_radius).abs();
            ex.max(ey) <= tol
        }
    }
}

#[verified_engine::verified]
pub(super) fn distance_to_nearest_pin(x: f64, y: f64, pins: PinConstraint2D) -> f64 {
    match pins {
        PinConstraint2D::None => 1.0,
        PinConstraint2D::CenterPin => ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)).sqrt(),
        PinConstraint2D::QuadSymmetry => {
            let mut best = 1.0_f64;
            for &(px, py) in &[(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                best = best.min(((x - px) * (x - px) + (y - py) * (y - py)).sqrt());
            }
            best
        }
        PinConstraint2D::EdgeMidpoints => {
            let mut best = 1.0_f64;
            for &(px, py) in &[(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)] {
                best = best.min(((x - px) * (x - px) + (y - py) * (y - py)).sqrt());
            }
            best
        }
    }
}
