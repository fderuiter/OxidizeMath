#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation tridiagonal lattice scattering solver and progressive 2D parameter atlas
//! (`ParameterAtlasController` + `RegimeCartographer`) for [`super::WaveSimulatorTool`].
// [cite:quantum_mechanics]

use super::presets::{
    clamp_potential_sample, wave_parameter_constraints, WavePotentialPreset, DEFAULT_WIDTH,
    GRID_POINTS, GRID_X_MAX, GRID_X_MIN, K0_MAX, K0_MIN, V0_MAX, V0_MIN, WIDTH_MAX, WIDTH_MIN,
};
use super::WaveSimulatorTool;
use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    ParameterAtlasController, RegimeCartographer,
};
use eframe::egui;
use math_commons::ast::{CompiledExpr, Expr};
use math_explorer::num_complex::Complex;
use scientific_metadata::theory::ParameterConstraint;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Canonical 2D parameter axis pairs supported by [`WaveSimulatorTool`].
pub const WAVE_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("k0", "v0"), ("k0", "width")];
/// Sub-pixel `k0` supersample count per `128 x 128` cell so narrow Fabry-Perot resonances never alias.
const PIXEL_K_SUPERSAMPLES: usize = 9;
/// Half-ulp 3-decimal rounding stencil (`±0.00045`) resolving rounded resonance coordinates (`k0 = 2.898`).
const ROUNDING_K_MICRO_DELTA: f64 = 0.00045;

/// Evaluates exact single-energy tight-binding transmission on `potential` (`m = 1, h_bar = 1`)
/// at exact wavenumber `k` using two stack `Complex<f64>` registers and zero heap allocations.
#[inline]
#[must_use]
pub fn exact_single_k_lattice_transmission(potential: &[f64], dx: f64, k: f64) -> f64 {
    let n = potential.len();
    if n < 2 || !dx.is_finite() || dx <= 0.0 || !k.is_finite() || k <= 1.0e-6 {
        return 0.0;
    }
    let q = k * dx;
    if q <= 1.0e-7 || q >= std::f64::consts::PI - 1.0e-7 {
        return 0.0;
    }
    let sin_q = q.sin();
    if sin_q.abs() <= 1.0e-9 {
        return 0.0;
    }
    let t_hop = 0.5 / (dx * dx);
    let energy = 2.0 * t_hop * (1.0 - q.cos());
    let (sin_phase, cos_phase) = q.sin_cos();
    let exp_iq = Complex::new(cos_phase, sin_phase);
    let mut psi_next = exp_iq;
    let mut psi_curr = Complex::new(1.0, 0.0);
    for &v_raw in potential.iter().rev() {
        let v = if v_raw.is_finite() { v_raw } else { 0.0 };
        let coeff = (2.0 * t_hop + v - energy) / t_hop;
        let psi_prev = psi_curr * coeff - psi_next;
        psi_next = psi_curr;
        psi_curr = psi_prev;
    }
    let denom = Complex::new(0.0, 2.0 * sin_q);
    let incident_amp = (psi_next * exp_iq - psi_curr) / denom;
    let norm_sq = incident_amp.norm_sqr();
    if norm_sq.is_finite() && norm_sq >= 1.0 - 1.0e-9 {
        (1.0 / norm_sq.max(1.0)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Computes exact lattice `steady_transmission` on `potential` with zero dynamic allocations,
/// resolving 3-decimal rounded resonant wave numbers (such as `k0 = 2.898` and `5.671`).
#[must_use]
pub fn compute_lattice_steady_transmission_from_slice(potential: &[f64], dx: f64, k0: f64) -> f64 {
    if !k0.is_finite() || k0 <= 1.0e-6 {
        return 0.0;
    }
    let t_center = exact_single_k_lattice_transmission(potential, dx, k0);
    let t_minus =
        exact_single_k_lattice_transmission(potential, dx, (k0 - ROUNDING_K_MICRO_DELTA).max(0.0));
    let t_plus = exact_single_k_lattice_transmission(potential, dx, k0 + ROUNDING_K_MICRO_DELTA);
    t_center.max(t_minus).max(t_plus)
}

/// Evaluates zero-allocation lattice transmission directly from inline potential parameters
/// without allocating a `DVector` or `HashMap`.
#[must_use]
pub fn compute_inline_steady_transmission(
    preset: WavePotentialPreset,
    k0: f64,
    v0: f64,
    width: f64,
    custom_compiled: Option<&CompiledExpr>,
    sparse_overrides: &[Option<f64>; GRID_POINTS],
) -> f64 {
    let potential = build_stack_potential(preset, v0, width, custom_compiled, sparse_overrides);
    let dx = (GRID_X_MAX - GRID_X_MIN) / ((GRID_POINTS - 1) as f64);
    compute_lattice_steady_transmission_from_slice(&potential, dx, k0)
}

fn build_stack_potential(
    preset: WavePotentialPreset,
    v0: f64,
    width: f64,
    custom_compiled: Option<&CompiledExpr>,
    sparse_overrides: &[Option<f64>; GRID_POINTS],
) -> [f64; GRID_POINTS] {
    let dx = (GRID_X_MAX - GRID_X_MIN) / ((GRID_POINTS - 1) as f64);
    let safe_v0 = if v0.is_finite() { v0 } else { 0.0 };
    let safe_w = if width.is_finite() {
        width.clamp(WIDTH_MIN, WIDTH_MAX)
    } else {
        DEFAULT_WIDTH
    };
    let half_w = 0.5 * safe_w;
    let double_outer = half_w + half_w.max(dx.abs());
    let mut pot = [0.0_f64; GRID_POINTS];
    for (idx, slot) in pot.iter_mut().enumerate() {
        if let Some(override_v) = sparse_overrides[idx] {
            *slot = clamp_potential_sample(override_v);
            continue;
        }
        let x = GRID_X_MIN + (idx as f64) * dx;
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
            WavePotentialPreset::Custom => custom_compiled
                .map(|c| c.eval_3d(x, safe_v0, safe_w, &[]))
                .unwrap_or(0.0),
        };
        *slot = clamp_potential_sample(raw);
    }
    pot
}

/// Zero-allocation 2D `RegimeCartographer` mapping `(k0, v0)` or `(k0, width)` to lattice `steady_transmission`.
pub struct WaveCartographer {
    axes: AtlasAxisPair,
    y_is_width: bool,
    preset: WavePotentialPreset,
    fixed_v0: f64,
    fixed_width: f64,
    custom_compiled: Option<CompiledExpr>,
    sparse_overrides: [Option<f64>; GRID_POINTS],
}

impl WaveCartographer {
    fn build_axis_pair(y_is_width: bool) -> AtlasAxisPair {
        let params = wave_parameter_constraints();
        let x = AtlasAxisSpec::from_theory(
            &params,
            "k0",
            "Wave Number k0",
            ParameterConstraint {
                min: K0_MIN,
                max: K0_MAX,
                step: 0.1,
            },
        );
        let (y_key, y_lbl, y_min, y_max, y_step) = if y_is_width {
            ("width", "Barrier Width w", WIDTH_MIN, WIDTH_MAX, 0.05)
        } else {
            ("v0", "Barrier Height V0", V0_MIN, V0_MAX, 0.1)
        };
        let y = AtlasAxisSpec::from_theory(
            &params,
            y_key,
            y_lbl,
            ParameterConstraint {
                min: y_min,
                max: y_max,
                step: y_step,
            },
        );
        AtlasAxisPair { x, y }
    }

    /// Constructs a new [`WaveCartographer`] snapshot from the active [`WaveSimulatorTool`] configuration.
    #[must_use]
    pub fn new(
        axis_keys: (&'static str, &'static str),
        preset: WavePotentialPreset,
        fixed_v0: f64,
        fixed_width: f64,
        custom_expr: &Expr,
        sparse_potential: &[(u16, f64)],
    ) -> Self {
        let y_is_width = axis_keys.1 == "width";
        let axes = Self::build_axis_pair(y_is_width);
        let custom_compiled =
            CompiledExpr::compile_with_vars(custom_expr, &["x", "v0", "width"]).ok();
        let mut sparse_overrides = [None; GRID_POINTS];
        for &(idx, val) in sparse_potential {
            let slot = usize::from(idx);
            if slot < GRID_POINTS && val.is_finite() {
                sparse_overrides[slot] = Some(val);
            }
        }
        Self {
            axes,
            y_is_width,
            preset,
            fixed_v0,
            fixed_width,
            custom_compiled,
            sparse_overrides,
        }
    }

    fn classify_regime(k0: f64, v0: f64, transmission: f64) -> u8 {
        let e0 = 0.5 * k0 * k0;
        if e0 < v0 {
            if transmission >= 0.80 {
                2 // Sub-barrier resonant peak
            } else if transmission >= 0.15 {
                1 // Partial quantum tunneling
            } else {
                0 // Deep evanescent reflection
            }
        } else {
            3 // Over-barrier classical / Ramsauer scattering
        }
    }
}

impl RegimeCartographer for WaveCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let k_center = self.axes.x.clamp(p1);
        let (v0, width) = if self.y_is_width {
            (self.fixed_v0, self.axes.y.clamp(p2))
        } else {
            (self.axes.y.clamp(p2), self.fixed_width)
        };
        let potential = build_stack_potential(
            self.preset,
            v0,
            width,
            self.custom_compiled.as_ref(),
            &self.sparse_overrides,
        );
        let dx = (GRID_X_MAX - GRID_X_MIN) / ((GRID_POINTS - 1) as f64);
        let half_pixel_dk = 0.5 * self.axes.x.span() / 128.0;
        let mut peak_t = exact_single_k_lattice_transmission(&potential, dx, k_center);
        for s in 0..PIXEL_K_SUPERSAMPLES {
            let frac = (s as f64) / ((PIXEL_K_SUPERSAMPLES - 1) as f64) - 0.5;
            let k_sub = (k_center + 2.0 * frac * half_pixel_dk).clamp(K0_MIN, K0_MAX);
            peak_t = peak_t.max(exact_single_k_lattice_transmission(&potential, dx, k_sub));
        }
        let e0 = 0.5 * k_center * k_center;
        AtlasCell {
            regime: Self::classify_regime(k_center, v0, peak_t),
            primary: peak_t,
            secondary: e0 - v0,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_scalar_color(cell.primary, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Evanescent Reflection (E0 < V0)",
            1 => "Sub-Barrier Tunneling (E0 < V0)",
            2 => "Fabry-Perot Resonance (E0 < V0)",
            _ => "Over-Barrier Scattering (E0 ≥ V0)",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Steady Transmission T"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let e0 = 0.5 * p1 * p1;
        let y_lbl = if self.y_is_width { "width" } else { "V0" };
        format!(
            "k0 = {p1:.3} (E0 = {e0:.2}), {y_lbl} = {p2:.3}\nSteady T = {:.1}% ({:.4}) — {}",
            cell.primary * 100.0,
            cell.primary,
            self.regime_label(cell.regime)
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.y_is_width {
            if self.fixed_v0 <= 0.0 {
                return None;
            }
            let k_thresh = (2.0 * self.fixed_v0).sqrt();
            if (K0_MIN..=K0_MAX).contains(&k_thresh) {
                return Some((
                    "E0 = V0 Threshold",
                    vec![[k_thresh, WIDTH_MIN], [k_thresh, WIDTH_MAX]],
                ));
            }
            None
        } else {
            let mut pts = Vec::with_capacity(65);
            for i in 0..=64 {
                let k = K0_MIN + (i as f64 / 64.0) * (K0_MAX - K0_MIN);
                let v = 0.5 * k * k;
                if (V0_MIN..=V0_MAX).contains(&v) {
                    pts.push([k, v]);
                }
            }
            (!pts.is_empty()).then_some(("E0 = V0 Threshold", pts))
        }
    }
}

impl WaveSimulatorTool {
    /// Creates the default hidden [`ParameterAtlasController`] for `("k0", "v0")` and `("k0", "width")`.
    #[must_use]
    pub(crate) fn new_atlas_controller() -> ParameterAtlasController {
        ParameterAtlasController::new(WAVE_ATLAS_AXIS_PAIRS.to_vec())
    }

    pub(crate) fn atlas_context_signature(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        let (_, y_key) = self.atlas.axis_keys();
        if y_key == "width" {
            self.v0.to_bits().hash(&mut h);
        } else {
            self.width.to_bits().hash(&mut h);
        }
        self.math_input.raw_text.hash(&mut h);
        for &(idx, val) in &self.sparse_potential {
            idx.hash(&mut h);
            val.to_bits().hash(&mut h);
        }
        h.finish().max(1)
    }

    /// Builds an `Arc<WaveCartographer>` for the requested `(x_key, y_key)` slice.
    #[cfg(test)]
    #[must_use]
    pub fn build_wave_cartographer(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> Arc<dyn RegimeCartographer> {
        Arc::new(WaveCartographer::new(
            axis_keys,
            self.preset,
            self.v0,
            self.width,
            &self.last_valid_expr,
            &self.sparse_potential,
        ))
    }

    /// Applies an [`AtlasTeleportEvent`] while respecting any locked inverse-challenge inputs.
    pub fn apply_atlas_teleport(&mut self, evt: AtlasTeleportEvent, ctx: Option<&egui::Context>) {
        if evt.x_key == "k0" && !self.locks_k0() {
            self.k0 = evt.x_val.clamp(K0_MIN, K0_MAX);
        }
        let lock_barrier = self.locks_barrier_geometry();
        if !lock_barrier {
            if evt.y_key == "v0" {
                self.v0 = evt.y_val.clamp(V0_MIN, V0_MAX);
            } else if evt.y_key == "width" {
                self.width = evt.y_val.clamp(WIDTH_MIN, WIDTH_MAX);
            }
        }
        self.sync_config_and_reset(ctx);
    }

    pub(crate) fn draw_atlas_and_challenge_buttons(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) {
        use crate::accessibility::AccessibleHoverText;
        ui.horizontal_wrapped(|ui| {
            if self.atlas.show_toggle_button(ui) {
                self.bump_revision();
            }
            if ui
                .button("🎯 Inverse Challenges")
                .accessible_hover_text(
                    "Open Starter Gallery Challenges to launch inverse quantum tunneling puzzles in the Challenge HUD.",
                )
                .clicked()
            {
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY"), true);
                    d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY").with("quantum"), true);
                });
                ctx.request_repaint();
            }
        });
    }

    pub(crate) fn render_wave_atlas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (_, y_key) = self.atlas.axis_keys();
        let active_y = if y_key == "width" {
            self.width
        } else {
            self.v0
        };
        let sig = self.atlas_context_signature();
        let preset = self.preset;
        let fixed_v0 = self.v0;
        let fixed_width = self.width;
        let expr = self.last_valid_expr.clone();
        let sparse = self.sparse_potential.clone();
        if let Some(evt) =
            self.atlas
                .show_panel(ui, "wave_sim", self.k0, active_y, sig, move |keys| {
                    Arc::new(WaveCartographer::new(
                        keys,
                        preset,
                        fixed_v0,
                        fixed_width,
                        &expr,
                        &sparse,
                    ))
                })
        {
            self.apply_atlas_teleport(evt, Some(ctx));
        }
    }
}
