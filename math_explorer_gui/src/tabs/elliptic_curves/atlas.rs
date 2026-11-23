#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]
//! 2D algebraic moduli and finite-field point-count cartographer for short-Weierstrass curves.

use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer, atlas_regime_color,
    atlas_scalar_color,
};
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    FpSingularCubic, RealSingularCubic, SingularityKind, WeierstrassCurveFp,
};
use math_explorer::pure_math::number_theory::primes::is_prime;
use scientific_metadata::theory::ParameterConstraint;

/// Canonical `(x_key, y_key)` axis pairs shared across elliptic curve tools.
pub const ELLIPTIC_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("real_a", "real_b"), ("mod_a", "mod_b")];

/// Default serialized atlas axis pair for elliptic curve tools.
#[must_use]
pub fn default_elliptic_atlas_axes() -> [String; 2] {
    ["real_a".to_string(), "real_b".to_string()]
}

/// Returns `true` when `axes` equals the default `["real_a", "real_b"]` pair.
#[must_use]
pub fn is_default_elliptic_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "real_a" && axes[1] == "real_b"
}

/// Discrete algebraic and arithmetic regimes mapped by [`WeierstrassRegimeCartographer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WeierstrassAtlasRegime {
    /// $\Delta > 0$: two real components (compact oval + unbounded branch).
    RealTwoComponent = 0,
    /// $\Delta < 0$: single smooth real branch.
    RealSingleBranch = 1,
    /// Equianharmonic CM locus $a = 0 \implies J = 0$.
    RealCmJ0 = 2,
    /// Lemniscatic CM locus $b = 0 \implies J = 1728$.
    RealCmJ1728 = 3,
    /// Origin cusp $a = b = 0$ ($\Delta = 0$).
    RealSingularCusp = 4,
    /// Self-intersecting crunode $27b^2 = -4a^3$ with $b > 0$ ($\Delta = 0$).
    RealSingularCrunode = 5,
    /// Isolated acnode $27b^2 = -4a^3$ with $b < 0$ ($\Delta = 0$).
    RealSingularAcnode = 6,
    /// Smooth ordinary reduction over $\mathbb{F}_p$.
    FpOrdinary = 7,
    /// Supersingular reduction $a_p = 0$ ($\#E(\mathbb{F}_p) = p + 1$).
    FpSupersingular = 8,
    /// Extremal Hasse-Weil boundary $|a_p| = \lfloor 2\sqrt{p}\rfloor$ (`hasse_gap == 0`).
    FpHasseBoundary = 9,
    /// Prime group order $\#E(\mathbb{F}_p)$.
    FpPrimeOrder = 10,
    /// Singular reduction $\Delta \equiv 0 \pmod p$ (`Cusp`, `Crunode`, or `Acnode`).
    FpSingular = 11,
}

impl WeierstrassAtlasRegime {
    /// Decodes a `u8` regime index into [`Self`].
    #[must_use]
    #[verified_engine::verified]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::RealTwoComponent,
            1 => Self::RealSingleBranch,
            2 => Self::RealCmJ0,
            3 => Self::RealCmJ1728,
            4 => Self::RealSingularCusp,
            5 => Self::RealSingularCrunode,
            6 => Self::RealSingularAcnode,
            7 => Self::FpOrdinary,
            8 => Self::FpSupersingular,
            9 => Self::FpHasseBoundary,
            10 => Self::FpPrimeOrder,
            _ => Self::FpSingular,
        }
    }

    /// Human-readable regime description for legends and ARIA announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::RealTwoComponent => "Δ > 0 Oval + Branch (2 real components)",
            Self::RealSingleBranch => "Δ < 0 Single Smooth Branch",
            Self::RealCmJ0 => "CM Ray J = 0 (a = 0)",
            Self::RealCmJ1728 => "CM Ray J = 1728 (b = 0)",
            Self::RealSingularCusp => "Δ = 0 Origin Cusp (0, 0)",
            Self::RealSingularCrunode => "Δ = 0 Crunode Self-Intersection",
            Self::RealSingularAcnode => "Δ = 0 Acnode Isolated Point + Branch",
            Self::FpOrdinary => "Smooth Ordinary E(F_p)",
            Self::FpSupersingular => "Supersingular (a_p = 0)",
            Self::FpHasseBoundary => "Hasse-Weil Boundary (|a_p| = ⌊2√p⌋)",
            Self::FpPrimeOrder => "Prime-Order Group #E(F_p)",
            Self::FpSingular => "Singular Reduction Δ ≡ 0 (mod p)",
        }
    }
}

#[verified_engine::verified]
fn eval_real_moduli_cell(a: f64, b: f64, band_tol: f64) -> AtlasCell {
    let core = 4.0 * a * a * a + 27.0 * b * b;
    let delta = -16.0 * core;
    if a.abs() <= band_tol && b.abs() <= band_tol {
        return AtlasCell {
            regime: WeierstrassAtlasRegime::RealSingularCusp as u8,
            primary: 0.0,
            secondary: 0.0,
        };
    }
    if a < 0.0 {
        let cusp_b = ((-4.0 * a * a * a) / 27.0).max(0.0).sqrt();
        if (b.abs() - cusp_b).abs() <= band_tol || RealSingularCubic::classify(a, b).is_some() {
            let regime = if b >= 0.0 {
                WeierstrassAtlasRegime::RealSingularCrunode
            } else {
                WeierstrassAtlasRegime::RealSingularAcnode
            };
            return AtlasCell {
                regime: regime as u8,
                primary: 0.0,
                secondary: (-3.0 * b) / (2.0 * a),
            };
        }
    }
    let j_val = if core.abs() > 1e-12 {
        (1728.0 * (4.0 * a * a * a) / core).clamp(-1.0e6, 1.0e6)
    } else {
        0.0
    };
    if a.abs() <= band_tol {
        return AtlasCell {
            regime: WeierstrassAtlasRegime::RealCmJ0 as u8,
            primary: delta,
            secondary: 0.0,
        };
    }
    if b.abs() <= band_tol {
        return AtlasCell {
            regime: WeierstrassAtlasRegime::RealCmJ1728 as u8,
            primary: delta,
            secondary: 1728.0,
        };
    }
    let regime = if delta > 0.0 {
        WeierstrassAtlasRegime::RealTwoComponent
    } else {
        WeierstrassAtlasRegime::RealSingleBranch
    };
    AtlasCell {
        regime: regime as u8,
        primary: delta,
        secondary: j_val,
    }
}

#[verified_engine::verified]
fn eval_fp_moduli_cell(p1: f64, p2: f64, p: i64) -> AtlasCell {
    let ai = p1.round() as i64;
    let bi = p2.round() as i64;
    if let Ok(Some(sing)) = FpSingularCubic::classify(ai, bi, p) {
        let kind_code = match sing.kind {
            SingularityKind::Cusp => 0.0,
            SingularityKind::Crunode => 1.0,
            SingularityKind::Acnode => 2.0,
        };
        return AtlasCell {
            regime: WeierstrassAtlasRegime::FpSingular as u8,
            primary: sing.projective_point_count() as f64,
            secondary: kind_code,
        };
    }
    let Ok(curve) = WeierstrassCurveFp::new(p, ai, bi) else {
        return AtlasCell {
            regime: WeierstrassAtlasRegime::FpSingular as u8,
            primary: p as f64,
            secondary: 0.0,
        };
    };
    let order = curve.cardinality();
    let trace = p + 1 - (order as i64);
    let hasse_floor = (2.0 * (p as f64).sqrt()).floor() as i64;
    let regime = if trace.abs() >= hasse_floor && hasse_floor > 0 {
        WeierstrassAtlasRegime::FpHasseBoundary
    } else if trace == 0 {
        WeierstrassAtlasRegime::FpSupersingular
    } else if order >= 2 && is_prime(order as u64) {
        WeierstrassAtlasRegime::FpPrimeOrder
    } else {
        WeierstrassAtlasRegime::FpOrdinary
    };
    AtlasCell {
        regime: regime as u8,
        primary: order as f64,
        secondary: trace as f64,
    }
}

/// Zero-allocation 2D discriminant, CM-ray, and `#E(F_p)` regime cartographer.
#[derive(Clone, Debug)]
pub struct WeierstrassRegimeCartographer {
    axes: AtlasAxisPair,
    is_fp_mode: bool,
    prime_p: i64,
    band_tol: f64,
}

impl WeierstrassRegimeCartographer {
    /// Creates a real `(a, b)` or modular `(mod_a, mod_b)` cartographer for `axis_keys` and `prime_p`.
    #[must_use]
    pub fn new(axis_keys: (&'static str, &'static str), prime_p: i64, real_step: f64) -> Self {
        let safe_p = prime_p.clamp(5, 997);
        let is_fp_mode = axis_keys.0 == "mod_a";
        let axes = if is_fp_mode {
            AtlasAxisPair {
                x: AtlasAxisSpec {
                    key: "mod_a",
                    label: "a (mod p)",
                    constraint: ParameterConstraint {
                        min: -15.0,
                        max: 15.0,
                        step: 1.0,
                    },
                },
                y: AtlasAxisSpec {
                    key: "mod_b",
                    label: "b (mod p)",
                    constraint: ParameterConstraint {
                        min: -15.0,
                        max: 15.0,
                        step: 1.0,
                    },
                },
            }
        } else {
            AtlasAxisPair {
                x: AtlasAxisSpec {
                    key: "real_a",
                    label: "Coefficient a",
                    constraint: ParameterConstraint {
                        min: -20.0,
                        max: 20.0,
                        step: real_step.max(0.05),
                    },
                },
                y: AtlasAxisSpec {
                    key: "real_b",
                    label: "Coefficient b",
                    constraint: ParameterConstraint {
                        min: -20.0,
                        max: 20.0,
                        step: real_step.max(0.05),
                    },
                },
            }
        };
        Self {
            axes,
            is_fp_mode,
            prime_p: safe_p,
            band_tol: 0.22,
        }
    }
}

impl RegimeCartographer for WeierstrassRegimeCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    #[verified_engine::verified]
    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        if self.is_fp_mode {
            eval_fp_moduli_cell(p1, p2, self.prime_p)
        } else {
            eval_real_moduli_cell(p1, p2, self.band_tol)
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        match WeierstrassAtlasRegime::from_u8(cell.regime) {
            WeierstrassAtlasRegime::RealTwoComponent => {
                let mag = (cell.primary.abs().ln_1p() / 12.0).clamp(0.0, 1.0);
                atlas_regime_color(1, 4, mag, high_contrast)
            }
            WeierstrassAtlasRegime::RealSingleBranch => {
                let mag = (cell.primary.abs().ln_1p() / 12.0).clamp(0.0, 1.0);
                atlas_regime_color(0, 4, mag, high_contrast)
            }
            WeierstrassAtlasRegime::RealCmJ0 | WeierstrassAtlasRegime::RealCmJ1728 => {
                atlas_regime_color(2, 4, 0.85, high_contrast)
            }
            WeierstrassAtlasRegime::RealSingularCusp
            | WeierstrassAtlasRegime::RealSingularCrunode
            | WeierstrassAtlasRegime::RealSingularAcnode => {
                atlas_regime_color(3, 4, 0.95, high_contrast)
            }
            WeierstrassAtlasRegime::FpSingular => atlas_regime_color(3, 4, 0.92, high_contrast),
            WeierstrassAtlasRegime::FpHasseBoundary => {
                atlas_regime_color(2, 4, 0.95, high_contrast)
            }
            WeierstrassAtlasRegime::FpSupersingular => {
                atlas_regime_color(1, 4, 0.90, high_contrast)
            }
            WeierstrassAtlasRegime::FpPrimeOrder | WeierstrassAtlasRegime::FpOrdinary => {
                let arch = 2.0 * (self.prime_p as f64).sqrt();
                let lo = (self.prime_p as f64) + 1.0 - arch;
                let norm = ((cell.primary - lo) / (2.0 * arch).max(1.0)).clamp(0.08, 0.86);
                atlas_scalar_color(norm, high_contrast)
            }
        }
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        WeierstrassAtlasRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.is_fp_mode {
            "#E(F_p)"
        } else {
            "Discriminant Δ"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        if self.is_fp_mode {
            let (ai, bi) = (p1.round() as i64, p2.round() as i64);
            if cell.regime == WeierstrassAtlasRegime::FpSingular as u8 {
                let kind = FpSingularCubic::classify(ai, bi, self.prime_p)
                    .ok()
                    .flatten()
                    .map_or("Singular", |s| s.kind.label());
                format!(
                    "F_{}: a = {}, b = {}\nRegime: {} ({})\nProjective Points: {:.0}",
                    self.prime_p,
                    ai,
                    bi,
                    self.regime_label(cell.regime),
                    kind,
                    cell.primary
                )
            } else {
                let hasse_floor = (2.0 * (self.prime_p as f64).sqrt()).floor();
                let gap = (hasse_floor - cell.secondary.abs()).max(0.0);
                format!(
                    "F_{}: a = {}, b = {}\nRegime: {}\n#E(F_p) = {:.0}  |  a_p = {:.0}  |  hasse_gap = {:.0}",
                    self.prime_p,
                    ai,
                    bi,
                    self.regime_label(cell.regime),
                    cell.primary,
                    cell.secondary,
                    gap
                )
            }
        } else {
            format!(
                "a = {:.2}, b = {:.2}\nRegime: {}\nΔ = {:.2}  |  J ≈ {:.2}",
                p1,
                p2,
                self.regime_label(cell.regime),
                cell.primary,
                cell.secondary
            )
        }
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.is_fp_mode {
            return None;
        }
        let a_min = self.axes.x.constraint.min.min(0.0);
        let b_lim = self.axes.y.constraint.max.abs();
        let mut pts = Vec::with_capacity(129);
        for i in (0..=64).rev() {
            let t = i as f64 / 64.0;
            let a = a_min * t * t;
            let b = ((-4.0 * a * a * a) / 27.0).max(0.0).sqrt();
            if b <= b_lim {
                pts.push([a, b]);
            }
        }
        for i in 1..=64 {
            let t = i as f64 / 64.0;
            let a = a_min * t * t;
            let b = -((-4.0 * a * a * a) / 27.0).max(0.0).sqrt();
            if b.abs() <= b_lim {
                pts.push([a, b]);
            }
        }
        Some(("Δ = 0 Cusp (27b² = -4a³)", pts))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        if self.is_fp_mode {
            vec![
                ("Ordinary #E(F_p)", atlas_scalar_color(0.45, high_contrast)),
                (
                    "Supersingular a_p=0",
                    atlas_regime_color(1, 4, 0.90, high_contrast),
                ),
                (
                    "Hasse Boundary",
                    atlas_regime_color(2, 4, 0.95, high_contrast),
                ),
                (
                    "Singular Δ≡0",
                    atlas_regime_color(3, 4, 0.92, high_contrast),
                ),
            ]
        } else {
            vec![
                (
                    "Δ > 0 (2 comp.)",
                    atlas_regime_color(1, 4, 0.75, high_contrast),
                ),
                (
                    "Δ < 0 (1 branch)",
                    atlas_regime_color(0, 4, 0.75, high_contrast),
                ),
                (
                    "CM J=0 / J=1728",
                    atlas_regime_color(2, 4, 0.85, high_contrast),
                ),
                (
                    "Δ = 0 Cusp/Node",
                    atlas_regime_color(3, 4, 0.95, high_contrast),
                ),
            ]
        }
    }
}

pub use super::observables::{
    CHALLENGE_HASSE_WEIL_BOUNDARY_P23, build_real_and_fp_observables, evaluate_hasse_weil_p23,
};
