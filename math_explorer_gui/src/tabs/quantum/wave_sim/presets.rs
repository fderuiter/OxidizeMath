#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

pub use super::observables::build_wave_datasets;
use math_commons::ast::{parse_equation, Expr};
use nalgebra::DVector;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Number of spatial grid points on the fixed 1D Dirichlet interval `[-5.0, 5.0]`.
pub const GRID_POINTS: usize = crate::url_state::CUSTOM_POTENTIAL_POINTS;
/// Left Dirichlet wall coordinate.
pub const GRID_X_MIN: f64 = -5.0;
/// Right Dirichlet wall coordinate.
pub const GRID_X_MAX: f64 = 5.0;

/// Minimum initial wavepacket center $x_0$.
pub const X0_MIN: f64 = -4.5;
/// Maximum initial wavepacket center $x_0$.
pub const X0_MAX: f64 = 4.5;
/// Default initial wavepacket center $x_0$.
pub const DEFAULT_X0: f64 = -2.0;

/// Minimum initial momentum wave number $k_0$.
pub const K0_MIN: f64 = -15.0;
/// Maximum initial momentum wave number $k_0$.
pub const K0_MAX: f64 = 15.0;
/// Default initial momentum wave number $k_0$.
pub const DEFAULT_K0: f64 = 5.0;

/// Minimum Gaussian packet width $\sigma$.
pub const SIGMA_MIN: f64 = 0.2;
/// Maximum Gaussian packet width $\sigma$.
pub const SIGMA_MAX: f64 = 2.0;
/// Default Gaussian packet width $\sigma$.
pub const DEFAULT_SIGMA: f64 = 0.5;

/// Minimum barrier height $V_0$.
pub const V0_MIN: f64 = 0.0;
/// Maximum barrier height $V_0$.
pub const V0_MAX: f64 = 40.0;
/// Default barrier height $V_0$ matching the 50% tunneling split benchmark.
pub const DEFAULT_V0: f64 = 11.35;

/// Minimum characteristic potential width.
pub const WIDTH_MIN: f64 = 0.1;
/// Maximum characteristic potential width.
pub const WIDTH_MAX: f64 = 2.0;
/// Default characteristic potential width.
pub const DEFAULT_WIDTH: f64 = 0.5;

/// Minimum slider time step $\Delta t$.
pub const DT_MIN: f64 = 0.001;
/// Minimum recovery time step permitted after auto-scaling.
pub const DT_RECOVERY_MIN: f64 = 1e-6;
/// Maximum time step $\Delta t$.
pub const DT_MAX: f64 = 0.1;
/// Default integration time step $\Delta t$.
pub const DEFAULT_DT: f64 = 0.02;

/// Default worker steps executed per rendered frame.
pub const DEFAULT_STEPS_PER_FRAME: usize = 5;
/// Maximum worker steps per frame.
pub const MAX_STEPS_PER_FRAME: usize = 100;

/// Lower clamp bound applied to every evaluated $V(x_i)$ sample before Hamiltonian construction.
pub const POTENTIAL_CLAMP_MIN: f64 = -500.0;
/// Upper clamp bound applied to every evaluated $V(x_i)$ sample before Hamiltonian construction.
pub const POTENTIAL_CLAMP_MAX: f64 = 500.0;
/// Visual vertical scale factor applied to $V(x)$ and $E_0 = k_0^2 / 2$ on the probability plot.
pub const POTENTIAL_DISPLAY_SCALE: f64 = 0.005;

/// Maximum UTF-8 byte length permitted for persisted `custom_expr` strings.
pub const MAX_CUSTOM_EXPR_BYTES: usize = 512;
/// Default custom AST potential formula.
pub const DEFAULT_CUSTOM_EXPR: &str = "v0 * exp(-(x / width)^4)";

/// Built-in 1D quantum potential profiles and custom AST formula mode.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum WavePotentialPreset {
    /// Particle in a 1D Dirichlet box ($V(x) = 0$ inside walls).
    #[default]
    InfiniteWell,
    /// Quadratic confining trap $V(x) = \frac{1}{2} x^2$.
    HarmonicOscillator,
    /// Single square barrier $V(x) = V_0$ on $[-w/2, w/2]$.
    RectangularBarrier,
    /// Symmetric double barrier enclosing a central resonant quantum well.
    DoubleBarrier,
    /// User-authored `math_commons::ast::Expr` evaluated over `(x, v0, width)`.
    Custom,
}

impl WavePotentialPreset {
    /// Ordered list of all selectable potential presets.
    pub const ALL: [Self; 5] = [
        Self::InfiniteWell,
        Self::HarmonicOscillator,
        Self::RectangularBarrier,
        Self::DoubleBarrier,
        Self::Custom,
    ];

    /// User-facing display label without raw struct or snake_case tokens.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::InfiniteWell => "Infinite Potential Well",
            Self::HarmonicOscillator => "Harmonic Oscillator",
            Self::RectangularBarrier => "Rectangular Barrier",
            Self::DoubleBarrier => "Double Barrier Resonance",
            Self::Custom => "Custom Formula",
        }
    }

    /// Maps legacy `WaveUnified` numeric `potential_type` values into [`Self`].
    #[must_use]
    pub fn from_legacy_potential_type(value: f64) -> Self {
        if value < 0.5 {
            Self::InfiniteWell
        } else if value < 1.5 {
            Self::HarmonicOscillator
        } else if value < 2.5 {
            Self::RectangularBarrier
        } else if value < 3.5 {
            Self::DoubleBarrier
        } else {
            Self::Custom
        }
    }

    /// Returns `(latex_formula, plain_formula)` for built-in presets, or `None` for custom AST mode.
    #[must_use]
    pub const fn formula_display(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::InfiniteWell => Some(("V(x) = 0 \\quad (|x| < 5)", "V(x) = 0 (|x| < 5)")),
            Self::HarmonicOscillator => Some((r"V(x) = \frac{1}{2} x^2", "V(x) = 0.5 * x^2")),
            Self::RectangularBarrier => Some((
                r"V(x) = V_0 \cdot \mathbf{1}_{|x| \le w/2}",
                "V(x) = V0 for |x| <= width/2 else 0",
            )),
            Self::DoubleBarrier => Some((
                r"V(x) = V_0 \cdot \mathbf{1}_{w/2 \le |x| \le w}",
                "V(x) = V0 for width/2 <= |x| <= width else 0",
            )),
            Self::Custom => None,
        }
    }

    /// Whether the preset depends on the `v0` and `width` barrier controls.
    #[must_use]
    pub const fn uses_barrier_controls(self) -> bool {
        matches!(
            self,
            Self::RectangularBarrier | Self::DoubleBarrier | Self::Custom
        )
    }
}

/// Clamps `(x0, k0, sigma)` wavepacket emitter parameters to safe finite domain bounds.
#[must_use]
pub fn clamp_wavepacket_params(x0: f64, k0: f64, sigma: f64) -> (f64, f64, f64) {
    let sx0 = if x0.is_finite() {
        x0.clamp(X0_MIN, X0_MAX)
    } else {
        DEFAULT_X0
    };
    let sk0 = if k0.is_finite() {
        k0.clamp(K0_MIN, K0_MAX)
    } else {
        DEFAULT_K0
    };
    let ssigma = if sigma.is_finite() {
        sigma.clamp(SIGMA_MIN, SIGMA_MAX)
    } else {
        DEFAULT_SIGMA
    };
    (sx0, sk0, ssigma)
}

/// Locates the dominant interior potential barrier index for two-region probability partitioning.
#[must_use]
pub fn detect_barrier_index(potential: &DVector<f64>) -> usize {
    let n = potential.len();
    if n < 3 {
        return 0;
    }
    let edge_ref = potential[0].max(potential[n - 1]);
    let mut max_interior = f64::NEG_INFINITY;
    for i in 1..(n - 1) {
        let v = potential[i];
        if v.is_finite() && v > max_interior {
            max_interior = v;
        }
    }
    if max_interior > edge_ref + 0.5 {
        for i in (1..(n - 1)).rev() {
            if potential[i].is_finite() && potential[i] >= max_interior - 0.25 {
                return i;
            }
        }
    }
    n / 2
}

/// Curated custom expression templates surfaced in `MathExpressionInputBar`.
#[must_use]
pub fn custom_formula_presets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Super-Gaussian Barrier", DEFAULT_CUSTOM_EXPR),
        ("Gaussian Barrier", "v0 * exp(-(x / width)^2)"),
        ("Pöschl-Teller Barrier", "v0 / cosh(x / width)^2"),
        (
            "Double Gaussian Barrier",
            "v0 * (exp(-((x + width) / 0.35)^2) + exp(-((x - width) / 0.35)^2))",
        ),
        ("Clamped Singular Pole (1/x)", "1 / x"),
    ]
}

/// Compiles the fallback default AST expression.
#[must_use]
pub fn default_compiled_expr() -> Expr {
    parse_equation(DEFAULT_CUSTOM_EXPR)
        .map(|(_, expr)| expr)
        .unwrap_or(Expr::Number(0.0))
}

/// Clamps any evaluated potential value to `[-500.0, 500.0]`, replacing `NaN` with `0.0`.
#[must_use]
pub fn clamp_potential_sample(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(POTENTIAL_CLAMP_MIN, POTENTIAL_CLAMP_MAX)
    }
}

/// Computes the scattering barrier interval `[x_left, x_right]` for Born R/B/T integration.
#[must_use]
pub fn barrier_interval_for_preset(preset: WavePotentialPreset, width: f64, dx: f64) -> (f64, f64) {
    let safe_w = if width.is_finite() {
        width.clamp(WIDTH_MIN, WIDTH_MAX)
    } else {
        DEFAULT_WIDTH
    };
    let half_w = 0.5 * safe_w;
    match preset {
        WavePotentialPreset::DoubleBarrier => {
            let wall_thickness = half_w.max(dx.abs());
            let outer = half_w + wall_thickness;
            (-outer, outer)
        }
        WavePotentialPreset::InfiniteWell
        | WavePotentialPreset::HarmonicOscillator
        | WavePotentialPreset::RectangularBarrier
        | WavePotentialPreset::Custom => (-half_w, half_w),
    }
}

/// Evaluates $V(x)$ across `x_axis` and clamps every sample to `[-500.0, 500.0]`.
#[must_use]
pub fn evaluate_potential_grid(
    preset: WavePotentialPreset,
    x_axis: &[f64],
    v0: f64,
    width: f64,
    dx: f64,
    custom_expr: &Expr,
) -> DVector<f64> {
    let safe_v0 = if v0.is_finite() { v0 } else { 0.0 };
    let safe_w = if width.is_finite() {
        width.clamp(WIDTH_MIN, WIDTH_MAX)
    } else {
        DEFAULT_WIDTH
    };
    let half_w = 0.5 * safe_w;
    let double_outer = half_w + half_w.max(dx.abs());

    let mut vars = HashMap::with_capacity(3);
    if preset == WavePotentialPreset::Custom {
        vars.insert("v0".to_string(), safe_v0);
        vars.insert("width".to_string(), safe_w);
        vars.insert("x".to_string(), 0.0);
    }

    DVector::from_fn(x_axis.len(), |i, _| {
        let x = x_axis[i];
        let raw = match preset {
            WavePotentialPreset::InfiniteWell => 0.0,
            WavePotentialPreset::HarmonicOscillator => 0.5 * x * x,
            WavePotentialPreset::RectangularBarrier => {
                if x.abs() <= half_w {
                    safe_v0
                } else {
                    0.0
                }
            }
            WavePotentialPreset::DoubleBarrier => {
                let ax = x.abs();
                if ax >= half_w && ax <= double_outer {
                    safe_v0
                } else {
                    0.0
                }
            }
            WavePotentialPreset::Custom => {
                if let Some(slot) = vars.get_mut("x") {
                    *slot = x;
                }
                custom_expr.eval(&vars).unwrap_or(0.0)
            }
        };
        clamp_potential_sample(raw)
    })
}

/// Parameter constraints advertised to `TheoryDescribable` and `state_guard` clamping.
#[must_use]
pub fn wave_parameter_constraints() -> HashMap<String, ParameterConstraint> {
    HashMap::from([
        (
            "x0".to_string(),
            ParameterConstraint {
                min: X0_MIN,
                max: X0_MAX,
                step: 0.05,
            },
        ),
        (
            "k0".to_string(),
            ParameterConstraint {
                min: K0_MIN,
                max: K0_MAX,
                step: 0.1,
            },
        ),
        (
            "sigma".to_string(),
            ParameterConstraint {
                min: SIGMA_MIN,
                max: SIGMA_MAX,
                step: 0.05,
            },
        ),
        (
            "v0".to_string(),
            ParameterConstraint {
                min: V0_MIN,
                max: V0_MAX,
                step: 0.05,
            },
        ),
        (
            "width".to_string(),
            ParameterConstraint {
                min: WIDTH_MIN,
                max: WIDTH_MAX,
                step: 0.05,
            },
        ),
        (
            "dt".to_string(),
            ParameterConstraint {
                min: DT_MIN,
                max: DT_MAX,
                step: 0.001,
            },
        ),
    ])
}

/// Human-readable display labels for all wave simulator parameters (no raw underscores).
#[must_use]
pub fn wave_parameter_label(name: &str) -> &'static str {
    match name {
        "preset" | "potential_type" => "Potential Profile",
        "x0" => "Initial Center (x0)",
        "k0" => "Wave Number (k0)",
        "sigma" => "Packet Width (σ)",
        "v0" => "Barrier Height (V0)",
        "width" => "Barrier Width (w)",
        "dt" => "Time Step (Δt)",
        "steps_per_frame" => "Speed (Steps/Frame)",
        _ => "Wave Parameter",
    }
}
