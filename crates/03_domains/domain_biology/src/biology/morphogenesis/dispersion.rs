//! Zero-allocation Schnakenberg and Gray-Scott Turing dispersion, Pearson saddle-node fold,
//! Hopf bifurcation, and 2D Forward Euler CFL classifiers.

/// Washout variance threshold below which activator/catalyst concentration is spatially uniform.
pub const WASHOUT_VARIANCE_THRESHOLD: f64 = 1e-8;
const GRAY_SCOTT_CATALYST_FLOOR: f64 = 0.015;

/// UI-free dispersion and CFL classification cell `(regime_code, primary_metric, secondary_metric)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispersionCell {
    /// Discrete regime index (`0..4` or `0..5`).
    pub regime: u8,
    /// Dominant growth rate $\operatorname{Re}(\lambda)$, fold margin, or Euler amplification factor.
    pub primary: f64,
    /// Critical wavenumber squared $k_*^2$ or stencil Nyquist bound.
    pub secondary: f64,
}

/// Discrete Schnakenberg reaction-diffusion regimes on the 2D 5-point stencil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MorphogenesisRegime {
    /// Homogeneous steady state $(u^*, v^*)$ is linearly and numerically stable.
    StableWashout = 0,
    /// Diffusion-driven Turing instability ($\tau_0 \le 0$ and $\det(J_{k_*}) < 0$).
    TuringPattern = 1,
    /// Homogeneous $k = 0$ Hopf instability ($\tau_0 > 0$).
    BulkHopfOscillation = 2,
    /// Explicit Forward Euler amplifies damped grid modes ($|1 + \Delta t \lambda| > 1$).
    EulerCflUnstable = 3,
}

impl MorphogenesisRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::StableWashout,
            1 => Self::TuringPattern,
            2 => Self::BulkHopfOscillation,
            _ => Self::EulerCflUnstable,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::StableWashout => "Stable Washout",
            Self::TuringPattern => "Turing Pattern",
            Self::BulkHopfOscillation => "Bulk Hopf Oscillation",
            Self::EulerCflUnstable => "Euler CFL Unstable",
        }
    }
}

/// Discrete Gray-Scott reaction-diffusion regimes on the 2D 5-point Laplacian stencil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GrayScottRegime {
    /// Beyond Pearson's saddle-node fold ($f + k > \sqrt{f}/2$): only $(u^*, v^*) = (1, 0)$ exists.
    SaddleNodeWashout = 0,
    /// Both $(1, 0)$ and nontrivial $(u_-, v_+)$ are linearly stable spatially uniform states.
    BistableHomogeneous = 1,
    /// Homogeneous $k = 0$ Hopf instability ($\operatorname{tr} J_0 > 0$) driving spiral/phase waves.
    HopfWave = 2,
    /// Diffusion-driven Turing / Pearson spot-splitting and coral morphogenesis instability.
    TuringMitosis = 3,
    /// Explicit Forward Euler amplifies high-wavenumber modes ($|1 + \Delta t \lambda| > 1$).
    EulerCflUnstable = 4,
}

impl GrayScottRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::SaddleNodeWashout,
            1 => Self::BistableHomogeneous,
            2 => Self::HopfWave,
            3 => Self::TuringMitosis,
            _ => Self::EulerCflUnstable,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SaddleNodeWashout => "Saddle-Node Washout",
            Self::BistableHomogeneous => "Bistable Homogeneous",
            Self::HopfWave => "Hopf Wave",
            Self::TuringMitosis => "Turing Mitosis",
            Self::EulerCflUnstable => "Euler CFL Unstable",
        }
    }
}

/// Linear stability classification for Schnakenberg `(a, b)` parameters at `(D_u, D_v)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchnakenbergRegime {
    /// Diffusion-driven Turing instability (`tr < 0`, `det > 0`, dispersion threshold exceeded).
    TuringUnstable,
    /// Homogeneous Hopf oscillatory instability (`tr >= 0`, `det > 0`).
    HopfOscillatory,
    /// Uniform stable steady state with no spatial pattern growth.
    UniformStable,
}

/// Bifurcation and pattern-forming classification for Gray-Scott `(F, k)` parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrayScottMinimapRegime {
    /// Turing or Hopf pattern-forming instability inside the saddle-node fold `F >= 4(F + k)^2`.
    BifurcationUnstable,
    /// Subcritical excitable pulse/mitosis/soliton band adjacent to the fold curve.
    SubcriticalPattern,
    /// Uniform high-catalyst or trivial substrate-only equilibrium.
    UniformStable,
}

/// Computes the population variance of `u` in two zero-allocation scalar passes.
#[must_use]
#[verified_engine::verified]
pub fn grid_u_variance(u: &[f64]) -> f64 {
    if u.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for &val in u {
        if !val.is_finite() {
            return f64::INFINITY;
        }
        sum += val;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / (count as f64);
    let mut sq_sum = 0.0;
    for &val in u {
        let diff = val - mean;
        sq_sum += diff * diff;
    }
    sq_sum / (count as f64)
}

/// Evaluates `(max_real_eigenvalue, euler_amplification, is_cfl_unstable)` at Laplacian eigenvalue `mu`.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn mode_growth_and_euler_amp(
    jac: [f64; 4],
    diff: [f64; 2],
    dt: f64,
    mu: f64,
) -> (f64, f64, bool) {
    let [fu, fv, gu, gv] = jac;
    let [d_u, d_v] = diff;
    let tr = (fu - mu * d_u) + (gv - mu * d_v);
    let det = (fu - mu * d_u) * (gv - mu * d_v) - fv * gu;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let root = disc.sqrt();
        let lam_plus = 0.5 * (tr + root);
        let lam_minus = 0.5 * (tr - root);
        let amp_plus = (1.0 + dt * lam_plus).abs();
        let amp_minus = (1.0 + dt * lam_minus).abs();
        let cfl_bad = (lam_minus <= 0.0 && 1.0 + dt * lam_minus < -1.0)
            || (lam_plus <= 0.0 && 1.0 + dt * lam_plus < -1.0);
        (lam_plus, amp_plus.max(amp_minus), cfl_bad)
    } else {
        let re = 0.5 * tr;
        let amp_sq = (1.0 + dt * tr + dt * dt * det).max(0.0);
        let cfl_bad = tr <= 0.0 && (tr + dt * det) > 0.0;
        (re, amp_sq.sqrt(), cfl_bad)
    }
}

/// Zero-allocation Schnakenberg dispersion and 2D Forward Euler CFL classifier.
///
/// # Examples
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::morphogenesis::{
///     MorphogenesisRegime, classify_schnakenberg_dispersion,
/// };
///
/// let cell = classify_schnakenberg_dispersion(0.1, 0.9, 1.0, 40.0, 0.005);
/// assert_eq!(MorphogenesisRegime::from_u8(cell.regime), MorphogenesisRegime::TuringPattern);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn classify_schnakenberg_dispersion(
    a: f64,
    b: f64,
    d_u: f64,
    d_v: f64,
    dt: f64,
) -> DispersionCell {
    if ![a, b, d_u, d_v, dt].into_iter().all(f64::is_finite)
        || d_u <= 0.0
        || d_v <= 0.0
        || dt <= 0.0
    {
        return DispersionCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: f64::INFINITY,
            secondary: 8.0,
        };
    }
    let u_eq = a + b;
    if u_eq <= 1e-6 {
        let washout = b <= 1e-9;
        return DispersionCell {
            regime: (if washout {
                MorphogenesisRegime::StableWashout
            } else {
                MorphogenesisRegime::EulerCflUnstable
            }) as u8,
            primary: if washout { -1.0 } else { 2.0 },
            secondary: 0.0,
        };
    }
    let u_sq = u_eq * u_eq;
    let (fu, fv, gu, gv) = ((b - a) / u_eq, u_sq, -2.0 * b / u_eq, -u_sq);
    let jac = [fu, fv, gu, gv];
    let diff = [d_u, d_v];
    let mu_star = ((d_v * fu + d_u * gv) / (2.0 * d_u * d_v)).clamp(0.0, 8.0);
    let (g0, a0, bad0) = mode_growth_and_euler_amp(jac, diff, dt, 0.0);
    let (gs, as_, bads) = mode_growth_and_euler_amp(jac, diff, dt, mu_star);
    let (g8, a8, bad8) = mode_growth_and_euler_amp(jac, diff, dt, 8.0);

    if bad0 || bads || bad8 {
        DispersionCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: a0.max(as_).max(a8),
            secondary: 8.0,
        }
    } else if fu + gv > 0.0 {
        DispersionCell {
            regime: MorphogenesisRegime::BulkHopfOscillation as u8,
            primary: g0.max(gs),
            secondary: 0.0,
        }
    } else if mu_star > 0.0 && gs > 0.0 {
        DispersionCell {
            regime: MorphogenesisRegime::TuringPattern as u8,
            primary: gs,
            secondary: mu_star,
        }
    } else {
        let (dom_g, dom_mu) = if gs >= g0 && gs >= g8 {
            (gs, mu_star)
        } else if g0 >= g8 {
            (g0, 0.0)
        } else {
            (g8, 8.0)
        };
        DispersionCell {
            regime: MorphogenesisRegime::StableWashout as u8,
            primary: dom_g,
            secondary: dom_mu,
        }
    }
}

/// Returns `true` when `(a, b, d_u, d_v, dt)` lies in a CFL-safe pattern-forming or oscillatory regime.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn is_pattern_forming_regime(a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) -> bool {
    matches!(
        MorphogenesisRegime::from_u8(classify_schnakenberg_dispersion(a, b, d_u, d_v, dt).regime),
        MorphogenesisRegime::TuringPattern | MorphogenesisRegime::BulkHopfOscillation
    )
}

/// Zero-allocation Gray-Scott saddle-node, Hopf, Turing-mitosis, and 2D Euler CFL classifier.
#[must_use]
#[verified_engine::verified]
pub fn classify_gray_scott_dispersion(
    f: f64,
    k: f64,
    d_u: f64,
    d_v: f64,
    dt: f64,
) -> DispersionCell {
    if ![f, k, d_u, d_v, dt].into_iter().all(f64::is_finite)
        || f < 0.0
        || k < 0.0
        || d_u <= 0.0
        || d_v <= 0.0
        || dt <= 0.0
    {
        return DispersionCell {
            regime: GrayScottRegime::EulerCflUnstable as u8,
            primary: f64::INFINITY,
            secondary: 8.0,
        };
    }
    let amp_u = (1.0 - dt * (f + 8.0 * d_u)).abs();
    let amp_v = (1.0 - dt * (f + k + 8.0 * d_v)).abs();
    let triv_amp = amp_u.max(amp_v);
    if (1.0 - dt * (f + 8.0 * d_u) < -1.0) || (1.0 - dt * (f + k + 8.0 * d_v) < -1.0) {
        return DispersionCell {
            regime: GrayScottRegime::EulerCflUnstable as u8,
            primary: triv_amp,
            secondary: 8.0,
        };
    }
    let fk = f + k;
    let half_sqrt_f = 0.5 * f.sqrt();
    if f <= 1e-9 || fk <= 1e-9 || fk > half_sqrt_f {
        return DispersionCell {
            regime: GrayScottRegime::SaddleNodeWashout as u8,
            primary: half_sqrt_f - fk,
            secondary: 0.0,
        };
    }
    let disc = (1.0 - 4.0 * fk * fk / f).max(0.0);
    let u_minus = (0.5 * (1.0 - disc.sqrt())).max(1e-9);
    let (fu, fv, gu, gv) = (-f / u_minus, -2.0 * fk, f * (1.0 - u_minus) / u_minus, fk);
    let jac = [fu, fv, gu, gv];
    let diff = [d_u, d_v];
    let mu_star = ((d_v * fu + d_u * gv) / (2.0 * d_u * d_v)).clamp(0.0, 8.0);
    let (g0, a0, bad0) = mode_growth_and_euler_amp(jac, diff, dt, 0.0);
    let (gs, as_, bads) = mode_growth_and_euler_amp(jac, diff, dt, mu_star);
    let (g8, a8, bad8) = mode_growth_and_euler_amp(jac, diff, dt, 8.0);

    if bad0 || bads || bad8 {
        DispersionCell {
            regime: GrayScottRegime::EulerCflUnstable as u8,
            primary: triv_amp.max(a0).max(as_).max(a8),
            secondary: 8.0,
        }
    } else if fu + gv > 0.0 {
        DispersionCell {
            regime: GrayScottRegime::HopfWave as u8,
            primary: g0.max(gs),
            secondary: 0.0,
        }
    } else if mu_star > 0.0 && gs > 0.0 {
        DispersionCell {
            regime: GrayScottRegime::TuringMitosis as u8,
            primary: gs,
            secondary: mu_star,
        }
    } else {
        DispersionCell {
            regime: GrayScottRegime::BistableHomogeneous as u8,
            primary: g0.max(gs).max(g8),
            secondary: mu_star,
        }
    }
}

/// Returns `true` when `(f, k, d_u, d_v, dt)` is CFL-stable and supports Gray-Scott patterns.
#[must_use]
#[verified_engine::verified]
pub fn is_gray_scott_pattern_viable(f: f64, k: f64, d_u: f64, d_v: f64, dt: f64) -> bool {
    let reg = GrayScottRegime::from_u8(classify_gray_scott_dispersion(f, k, d_u, d_v, dt).regime);
    if reg == GrayScottRegime::EulerCflUnstable {
        return false;
    }
    if matches!(
        reg,
        GrayScottRegime::TuringMitosis | GrayScottRegime::HopfWave
    ) {
        return true;
    }
    let fk = f + k;
    let sqrt_f = f.max(0.0).sqrt();
    d_u > 1.15 * d_v && f >= 0.008 && k >= 0.025 && fk >= 0.35 * sqrt_f && fk <= 0.56 * sqrt_f
}

/// Returns `true` when Gray-Scott catalyst field `v` has washed out to a spatially uniform state.
#[must_use]
#[verified_engine::verified]
pub fn is_gray_scott_v_washed_out(v: &[f64]) -> bool {
    if v.is_empty() {
        return false;
    }
    let var_v = grid_u_variance(v);
    if !var_v.is_finite() || var_v >= WASHOUT_VARIANCE_THRESHOLD {
        return false;
    }
    let mut max_v = 0.0_f64;
    for &val in v {
        max_v = max_v.max(val);
    }
    max_v < GRAY_SCOTT_CATALYST_FLOOR || var_v < WASHOUT_VARIANCE_THRESHOLD
}

/// Classifies Schnakenberg `(a, b)` continuous stability analytically for diffusion rates `(d_u, d_v)`.
#[must_use]
#[verified_engine::verified]
pub fn classify_schnakenberg(a: f64, b: f64, d_u: f64, d_v: f64) -> SchnakenbergRegime {
    let sum = a + b;
    if ![a, b, d_u, d_v].into_iter().all(f64::is_finite)
        || a < 0.0
        || b <= 0.0
        || sum <= 1e-9
        || d_u <= 0.0
        || d_v <= 0.0
    {
        return SchnakenbergRegime::UniformStable;
    }
    let f_u = (b - a) / sum;
    let f_v = sum * sum;
    let g_u = -2.0 * b / sum;
    let g_v = -(sum * sum);
    let trace = f_u + g_v;
    let det = f_u * g_v - f_v * g_u;
    if trace < 0.0 && det > 0.0 && (d_v * f_u + d_u * g_v) > 2.0 * (d_u * d_v * det).sqrt() {
        SchnakenbergRegime::TuringUnstable
    } else if trace >= 0.0 && det > 0.0 {
        SchnakenbergRegime::HopfOscillatory
    } else {
        SchnakenbergRegime::UniformStable
    }
}

/// Saddle-node fold curve `k_fold(F) = sqrt(F) / 2 - F` solving `F = 4 * (F + k)^2`.
#[must_use]
#[verified_engine::verified]
pub fn gray_scott_fold_k(f: f64) -> Option<f64> {
    (f.is_finite() && (0.0..=0.25).contains(&f)).then(|| 0.5 * f.sqrt() - f)
}

/// Hopf curve `F_hopf(k)` solving `(F + k)^2 = F * sqrt(k)` on the upper catalyst branch (`k <= 1/16`).
#[must_use]
#[verified_engine::verified]
pub fn gray_scott_hopf_f(k: f64) -> Option<f64> {
    if !k.is_finite() || !(0.0..=0.0625).contains(&k) {
        return None;
    }
    let sqrt_k = k.sqrt();
    let disc = (k * (1.0 - 4.0 * sqrt_k)).max(0.0);
    Some(0.5 * ((sqrt_k - 2.0 * k) - disc.sqrt()))
}

/// Classifies Gray-Scott `(f, k)` parameters relative to the fold `F = 4(F + k)^2` and Hopf curve.
#[must_use]
#[verified_engine::verified]
pub fn classify_gray_scott(f: f64, k: f64, d_u: f64, d_v: f64) -> GrayScottMinimapRegime {
    if ![f, k, d_u, d_v].into_iter().all(f64::is_finite)
        || f <= 0.0
        || k <= 0.0
        || d_u <= 0.0
        || d_v <= 0.0
    {
        return GrayScottMinimapRegime::UniformStable;
    }
    let sum = f + k;
    let fold_rhs = 4.0 * sum * sum;
    if f >= fold_rhs {
        let disc = (f * (f - fold_rhs)).max(0.0).sqrt();
        let v_plus = (f + disc) / (2.0 * sum);
        let f_u = -(v_plus * v_plus) - f;
        let g_v = sum;
        let det = sum * (v_plus * v_plus - f);
        let turing = det > 0.0 && (d_v * f_u + d_u * g_v) > 2.0 * (d_u * d_v * det).sqrt();
        if f_u + g_v >= 0.0 || turing {
            return GrayScottMinimapRegime::BifurcationUnstable;
        }
        return GrayScottMinimapRegime::UniformStable;
    }
    if let Some(k_fold) = gray_scott_fold_k(f) {
        if (0.018..=0.0625).contains(&f) && k > k_fold && k <= k_fold + 0.0065 {
            return GrayScottMinimapRegime::SubcriticalPattern;
        }
    }
    GrayScottMinimapRegime::UniformStable
}

#[cfg(test)]
#[path = "dispersion_tests.rs"]
mod tests;

// [cite:dwarf_galaxy_empirical_dependencies]
