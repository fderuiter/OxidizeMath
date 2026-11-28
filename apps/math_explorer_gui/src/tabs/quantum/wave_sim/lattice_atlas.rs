#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

//! [`WaveCartographer`] for [`super::WaveSimulatorTool`] backed by
//! [`math_explorer::physics::quantum::lattice_scattering`].

use super::presets::{
    wave_parameter_constraints, WavePotentialPreset, DEFAULT_WIDTH, GRID_POINTS, GRID_X_MAX,
    GRID_X_MIN, K0_MAX, K0_MIN, POTENTIAL_CLAMP_MAX, POTENTIAL_CLAMP_MIN, V0_MAX, V0_MIN,
    WIDTH_MAX, WIDTH_MIN,
};
use super::runner::WaveRunner;
use super::WaveSimulatorTool;
use crate::async_sim::atlas::{
    atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use math_commons::ast::{CompiledExpr, Expr};
pub use math_explorer::physics::quantum::{
    classify_lattice_scattering_regime, compute_lattice_steady_transmission_from_slice,
    populate_lattice_potential_into, supersampled_lattice_transmission, LatticePotentialKind,
    DEFAULT_PIXEL_K_SUPERSAMPLES,
};
use scientific_metadata::theory::ParameterConstraint;
#[cfg(test)]
use std::sync::Arc;

fn map_preset_kind(preset: WavePotentialPreset) -> LatticePotentialKind {
    match preset {
        WavePotentialPreset::InfiniteWell => LatticePotentialKind::InfiniteWell,
        WavePotentialPreset::HarmonicOscillator => LatticePotentialKind::HarmonicOscillator,
        WavePotentialPreset::RectangularBarrier => LatticePotentialKind::RectangularBarrier,
        WavePotentialPreset::DoubleBarrier => LatticePotentialKind::DoubleBarrier,
        WavePotentialPreset::Custom => LatticePotentialKind::Custom,
    }
}

fn build_stack_potential(
    preset: WavePotentialPreset,
    v0: f64,
    width: f64,
    custom_compiled: Option<&CompiledExpr>,
    sparse_overrides: &[Option<f64>; GRID_POINTS],
) -> [f64; GRID_POINTS] {
    let safe_w = if width.is_finite() {
        width.clamp(WIDTH_MIN, WIDTH_MAX)
    } else {
        DEFAULT_WIDTH
    };
    let mut pot = [0.0_f64; GRID_POINTS];
    populate_lattice_potential_into(
        map_preset_kind(preset),
        GRID_X_MIN,
        GRID_X_MAX,
        v0,
        safe_w,
        POTENTIAL_CLAMP_MIN,
        POTENTIAL_CLAMP_MAX,
        custom_compiled,
        sparse_overrides,
        &mut pot,
    );
    pot
}

/// Evaluates zero-allocation lattice transmission directly from inline potential parameters.
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
        let peak_t = supersampled_lattice_transmission(
            &potential,
            dx,
            k_center,
            half_pixel_dk,
            K0_MIN,
            K0_MAX,
            DEFAULT_PIXEL_K_SUPERSAMPLES,
        );
        let e0 = 0.5 * k_center * k_center;
        AtlasCell {
            regime: classify_lattice_scattering_regime(k_center, v0, peak_t),
            primary: peak_t,
            secondary: e0 - v0,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_scalar_color(cell.primary.clamp(0.0, 1.0), high_contrast)
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
            (K0_MIN..=K0_MAX).contains(&k_thresh).then(|| {
                (
                    "E0 = V0 Threshold",
                    vec![[k_thresh, WIDTH_MIN], [k_thresh, WIDTH_MAX]],
                )
            })
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

impl WaveRunner {
    /// Computes exact steady-state lattice transmission on the active 1D Hamiltonian grid.
    #[must_use]
    pub fn steady_transmission(&self) -> f64 {
        let dx = (self.x_max - self.x_min) / ((self.n_points.max(2) - 1) as f64);
        compute_lattice_steady_transmission_from_slice(self.potential.as_slice(), dx, self.k0)
    }
}

impl WaveSimulatorTool {
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
}
