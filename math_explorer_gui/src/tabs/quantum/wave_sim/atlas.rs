// [cite:quantum_mechanics]
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation 2D quantum tunneling & Fabry-Pérot resonance parameter atlas.

use super::presets::{
    clamp_potential_sample, WavePotentialPreset, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_V0,
    DEFAULT_WIDTH, GRID_POINTS, GRID_X_MAX, GRID_X_MIN, K0_MAX, K0_MIN, SIGMA_MAX, SIGMA_MIN,
    V0_MAX, V0_MIN, WIDTH_MAX, WIDTH_MIN,
};
use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use math_commons::ast::{CompiledExpr, Expr};
use math_explorer::physics::quantum::{
    compiled_barrier_transmission, compiled_barrier_transmission_with_overrides,
    gauss_hermite_wavepacket_transmission, piecewise_potential_transmission,
    rectangular_barrier_transmission, symmetric_double_barrier_transmission,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

const SLICES_1D: usize = 96;
const WINDOW_HALF_SPAN: f64 = 4.5;

/// Supported `(x, y)` parameter axis pairs for [`QuantumTunnelingCartographer`].
pub const WAVE_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("k0", "v0"), ("k0", "width")];

/// Default serialized atlas axis pair for [`super::WaveSimulatorTool`].
#[must_use]
pub fn default_wave_atlas_axes() -> [String; 2] {
    ["k0".to_string(), "v0".to_string()]
}

/// Predicate omitting default `atlas_axes` during JSON serialization.
#[must_use]
pub fn is_default_wave_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "k0" && axes[1] == "v0"
}

/// Predicate omitting `false` boolean flags during JSON serialization.
#[must_use]
pub const fn is_false(flag: &bool) -> bool {
    !*flag
}

/// Discrete 1D quantum scattering and tunneling regimes across $(k_0, V_0, w)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantumTunnelingRegime {
    /// Sub-barrier evanescent tunneling ($E_0 < V_0$, $T < 0.70$).
    SubBarrierTunneling = 0,
    /// Sub-barrier resonant Fabry-Pérot transparency ($E_0 < V_0$, $T \ge 0.70$).
    FabryPerotResonance = 1,
    /// Over-barrier partial reflection ($E_0 \ge V_0$, $T < 0.96$).
    OverBarrierScattering = 2,
    /// Ramsauer-Townsend / full transparency ($E_0 \ge V_0$, $T \ge 0.96$).
    RamsauerTransparency = 3,
}

impl QuantumTunnelingRegime {
    /// Classifies `(energy, v0, stationary_t)` into a discrete quantum scattering regime.
    #[inline]
    #[must_use]
    pub fn classify(energy: f64, v0: f64, stationary_t: f64) -> Self {
        if energy < v0 {
            if stationary_t >= 0.70 {
                Self::FabryPerotResonance
            } else {
                Self::SubBarrierTunneling
            }
        } else if stationary_t >= 0.96 {
            Self::RamsauerTransparency
        } else {
            Self::OverBarrierScattering
        }
    }

    /// Converts a raw `u8` regime index into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::SubBarrierTunneling,
            1 => Self::FabryPerotResonance,
            2 => Self::OverBarrierScattering,
            _ => Self::RamsauerTransparency,
        }
    }

    /// Human-readable quantum scattering regime label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SubBarrierTunneling => "Sub-Barrier Tunneling (E₀ < V₀)",
            Self::FabryPerotResonance => "Fabry-Pérot Resonance (E₀ < V₀)",
            Self::OverBarrierScattering => "Over-Barrier Scattering (E₀ ≥ V₀)",
            Self::RamsauerTransparency => "Ramsauer-Townsend Transparency",
        }
    }
}

fn wave_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "v0" => AtlasAxisSpec::from_theory(
            theory_params,
            "v0",
            "Barrier Height (V₀)",
            ParameterConstraint {
                min: V0_MIN,
                max: V0_MAX,
                step: 0.05,
            },
        ),
        "width" => AtlasAxisSpec::from_theory(
            theory_params,
            "width",
            "Barrier Width (w)",
            ParameterConstraint {
                min: WIDTH_MIN,
                max: WIDTH_MAX,
                step: 0.05,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory_params,
            "k0",
            "Wave Number (k₀)",
            ParameterConstraint {
                min: K0_MIN,
                max: K0_MAX,
                step: 0.1,
            },
        ),
    }
}

fn build_harmonic_slices() -> [f64; SLICES_1D] {
    let mut slices = [0.0; SLICES_1D];
    let dx_h = (2.0 * WINDOW_HALF_SPAN) / (SLICES_1D as f64);
    for (i, slot) in slices.iter_mut().enumerate() {
        let x = -WINDOW_HALF_SPAN + (i as f64 + 0.5) * dx_h;
        *slot = 0.5 * x * x;
    }
    slices
}

fn pack_sparse_overrides(sparse: &[(u16, f64)]) -> ([(u16, f64); GRID_POINTS], usize) {
    let mut overrides = [(0_u16, 0.0_f64); GRID_POINTS];
    let mut len = 0;
    for &(idx, val) in sparse.iter().take(GRID_POINTS) {
        if usize::from(idx) < GRID_POINTS && val.is_finite() {
            overrides[len] = (idx, clamp_potential_sample(val));
            len += 1;
        }
    }
    (overrides, len)
}

/// Zero-allocation 2D transfer-matrix & Gauss-Hermite cartographer for [`super::WaveSimulatorTool`].
#[derive(Clone, Debug)]
pub struct QuantumTunnelingCartographer {
    axes: AtlasAxisPair,
    preset: WavePotentialPreset,
    base_k0: f64,
    base_v0: f64,
    base_width: f64,
    sigma: f64,
    compiled_custom: Option<CompiledExpr>,
    harmonic_slices: [f64; SLICES_1D],
    sparse_overrides: [(u16, f64); GRID_POINTS],
    sparse_len: usize,
}

impl QuantumTunnelingCartographer {
    /// Constructs a pre-compiled [`QuantumTunnelingCartographer`] safe for `VerifiedEngine::run_verified`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        preset: WavePotentialPreset,
        base_k0: f64,
        base_v0: f64,
        base_width: f64,
        sigma: f64,
        custom_expr: &Expr,
        sparse_potential: &[(u16, f64)],
    ) -> Self {
        let compiled_custom = (preset == WavePotentialPreset::Custom)
            .then(|| {
                CompiledExpr::compile_with_vars_and_params(custom_expr, &["x"], &["v0", "width"])
                    .ok()
            })
            .flatten();
        let (sparse_overrides, sparse_len) = pack_sparse_overrides(sparse_potential);
        Self {
            axes: AtlasAxisPair {
                x: wave_axis_spec(theory_params, axis_keys.0),
                y: wave_axis_spec(theory_params, axis_keys.1),
            },
            preset,
            base_k0: if base_k0.is_finite() {
                base_k0
            } else {
                DEFAULT_K0
            },
            base_v0: if base_v0.is_finite() {
                base_v0
            } else {
                DEFAULT_V0
            },
            base_width: if base_width.is_finite() {
                base_width.clamp(WIDTH_MIN, WIDTH_MAX)
            } else {
                DEFAULT_WIDTH
            },
            sigma: if sigma.is_finite() && sigma > 0.0 {
                sigma.clamp(SIGMA_MIN, SIGMA_MAX)
            } else {
                DEFAULT_SIGMA
            },
            compiled_custom,
            harmonic_slices: build_harmonic_slices(),
            sparse_overrides,
            sparse_len,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64) {
        let (mut k0, mut v0, mut width) = (self.base_k0, self.base_v0, self.base_width);
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "k0" => k0 = val,
                "v0" => v0 = val,
                "width" => width = val,
                _ => {}
            }
        }
        (k0, v0, width)
    }

    /// Evaluates stationary transmission $T(k)$ at `(v0, width)` with zero heap allocations.
    #[must_use]
    pub fn stationary_transmission(&self, k: f64, v0: f64, width: f64) -> f64 {
        if !k.is_finite() || k <= 0.0 {
            return 0.0;
        }
        if self.sparse_len > 0 {
            return self.sculpted_transmission(k, v0, width);
        }
        match self.preset {
            WavePotentialPreset::InfiniteWell => 1.0,
            WavePotentialPreset::HarmonicOscillator => piecewise_potential_transmission(
                k,
                &self.harmonic_slices,
                (2.0 * WINDOW_HALF_SPAN) / (SLICES_1D as f64),
                1.0,
                1.0,
            ),
            WavePotentialPreset::RectangularBarrier => {
                rectangular_barrier_transmission(k, v0, width, 1.0, 1.0)
            }
            WavePotentialPreset::DoubleBarrier => {
                symmetric_double_barrier_transmission(k, v0, width, 1.0, 1.0)
            }
            WavePotentialPreset::Custom => self.compiled_custom.as_ref().map_or(1.0, |compiled| {
                let params = [v0, width];
                compiled_barrier_transmission(
                    k,
                    compiled,
                    &params,
                    (-WINDOW_HALF_SPAN, WINDOW_HALF_SPAN, SLICES_1D),
                    1.0,
                    1.0,
                )
            }),
        }
    }

    fn sculpted_transmission(&self, k: f64, v0: f64, width: f64) -> f64 {
        let overrides = &self.sparse_overrides[..self.sparse_len];
        if self.preset == WavePotentialPreset::Custom {
            if let Some(compiled) = &self.compiled_custom {
                let params = [v0, width];
                return compiled_barrier_transmission_with_overrides(
                    k,
                    compiled,
                    &params,
                    (GRID_X_MIN, GRID_X_MAX, GRID_POINTS),
                    overrides,
                    1.0,
                    1.0,
                );
            }
        }
        let dx = (GRID_X_MAX - GRID_X_MIN) / (GRID_POINTS as f64 - 1.0);
        let half_w = 0.5 * width.clamp(WIDTH_MIN, WIDTH_MAX);
        let outer = half_w + half_w.max(dx);
        let clamped_v = clamp_potential_sample(v0);
        let mut grid = [0.0_f64; GRID_POINTS];
        for (i, slot) in grid.iter_mut().enumerate() {
            let x = GRID_X_MIN + i as f64 * dx;
            *slot = match self.preset {
                WavePotentialPreset::HarmonicOscillator => 0.5 * x * x,
                WavePotentialPreset::RectangularBarrier if x.abs() <= half_w => clamped_v,
                WavePotentialPreset::DoubleBarrier if (half_w..=outer).contains(&x.abs()) => {
                    clamped_v
                }
                _ => 0.0,
            };
        }
        for &(idx, val) in overrides {
            grid[usize::from(idx)] = val;
        }
        piecewise_potential_transmission(k, &grid, dx, 1.0, 1.0)
    }
}

impl RegimeCartographer for QuantumTunnelingCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (k0, v0, width) = self.resolve_params(p1, p2);
        let t_stat = self.stationary_transmission(k0, v0, width);
        let t_wp = gauss_hermite_wavepacket_transmission(k0, self.sigma, |k| {
            self.stationary_transmission(k, v0, width)
        });
        let eff_v0 = if self.preset.uses_barrier_controls() {
            v0
        } else {
            0.0
        };
        AtlasCell {
            regime: QuantumTunnelingRegime::classify(0.5 * k0 * k0, eff_v0, t_stat) as u8,
            primary: t_stat,
            secondary: t_wp,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_scalar_color(cell.primary.clamp(0.0, 1.0), high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        QuantumTunnelingRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Stationary T(k₀)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let (k0, _, _) = self.resolve_params(p1, p2);
        format!(
            "{} = {:.3}, {} = {:.3} (E₀ = {:.2})\nRegime: {}\nStationary T(k₀) = {:.2}%  |  Wavepacket ⟨T⟩_σ = {:.2}%",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            0.5 * k0 * k0,
            self.regime_label(cell.regime),
            cell.primary * 100.0,
            cell.secondary * 100.0
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if !self.preset.uses_barrier_controls() || self.axes.x.key != "k0" {
            return None;
        }
        if self.axes.y.key == "v0" {
            let k_min = self.axes.x.constraint.min;
            let k_max = self
                .axes
                .x
                .constraint
                .max
                .min((2.0 * self.axes.y.constraint.max).sqrt());
            (k_max > k_min).then(|| {
                let pts = (0..=48)
                    .map(|i| {
                        let k = k_min + (i as f64 / 48.0) * (k_max - k_min);
                        [
                            k,
                            (0.5 * k * k)
                                .clamp(self.axes.y.constraint.min, self.axes.y.constraint.max),
                        ]
                    })
                    .collect();
                ("E₀ = V₀ Threshold", pts)
            })
        } else if self.axes.y.key == "width" && self.base_v0 > 0.0 {
            let k_c = (2.0 * self.base_v0).sqrt();
            (self.axes.x.constraint.min..=self.axes.x.constraint.max)
                .contains(&k_c)
                .then(|| {
                    (
                        "E₀ = V₀ Threshold",
                        vec![
                            [k_c, self.axes.y.constraint.min],
                            [k_c, self.axes.y.constraint.max],
                        ],
                    )
                })
        } else {
            None
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        vec![
            ("Opaque (T ≈ 0%)", atlas_scalar_color(0.06, high_contrast)),
            ("50% Split", atlas_scalar_color(0.50, high_contrast)),
            (
                "Resonant (T > 95%)",
                atlas_scalar_color(0.96, high_contrast),
            ),
        ]
    }
}
