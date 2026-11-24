//! Interactive **Phyllotaxis & L-Systems** studio (`PhyllotaxisLSystemTool`) inside `MorphogenesisTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

pub use presets::{
    BotanicalTheory, GuiDivergencePreset, GuiLSystemPreset, PhyllotaxisLSystemState,
};

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::biology::lsystem::{LSystemEngine, MAX_LSYSTEM_SYMBOLS};
use math_explorer::biology::phyllotaxis::{
    GOLDEN_ANGLE_DEG, MAX_PRIMORDIA, VogelPhyllotaxisEngine,
};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

/// Interactive studio pairing Vogel golden-angle phyllotaxis with zero-allocation D0L and
/// parametric L-system vascular branching grammars.
pub struct PhyllotaxisLSystemTool {
    /// Serializable URL and preset configuration.
    pub state: PhyllotaxisLSystemState,
    pub(crate) vogel: VogelPhyllotaxisEngine,
    pub(crate) lsystem: LSystemEngine,
    pub(crate) selected_primordium: usize,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    revision_nonce: u64,
    theory: BotanicalTheory,
}

impl Default for PhyllotaxisLSystemTool {
    fn default() -> Self {
        let state = PhyllotaxisLSystemState::default();
        let vogel = VogelPhyllotaxisEngine::new(
            state.primordia_count,
            state.divergence_deg.to_radians(),
            state.radial_scale,
        )
        .expect("default Vogel phyllotaxis parameters are valid");
        let (grammar, cfg) = state
            .lsystem_preset
            .to_domain()
            .build_grammar(
                state.branch_angle_deg,
                state.contraction_r1,
                state.contraction_r2,
            )
            .expect("default L-system grammar parameters are valid");
        let mut lsystem = LSystemEngine::new(grammar, cfg, MAX_LSYSTEM_SYMBOLS)
            .expect("default L-system engine capacity is valid");
        let _ = lsystem.derive_bounded(state.generations);

        Self {
            state,
            vogel,
            lsystem,
            selected_primordium: 0,
            diagnostic: None,
            revision_nonce: 1,
            theory: BotanicalTheory,
        }
    }
}

impl PhyllotaxisLSystemTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision_nonce = self.revision_nonce.wrapping_add(1);
    }

    /// Applies a canonical [`GuiDivergencePreset`] and refreshes the Vogel engine in place.
    pub fn apply_divergence_preset(&mut self, preset: GuiDivergencePreset) {
        self.state.divergence_preset = preset;
        self.state.divergence_deg = preset.to_domain().angle_deg();
        self.sync_engines();
    }

    /// Applies a canonical [`GuiLSystemPreset`] and derives its default shoot geometry.
    pub fn apply_lsystem_preset(&mut self, preset: GuiLSystemPreset) {
        self.state.lsystem_preset = preset;
        let dom = preset.to_domain();
        self.state.generations = dom.default_generations();
        self.state.branch_angle_deg = dom.default_angle_deg();
        self.state.contraction_r1 = dom.default_contraction_primary();
        self.state.contraction_r2 = dom.default_contraction_lateral();
        self.sync_engines();
    }

    /// Synchronizes both pre-allocated `domain_biology` kernels without heap allocations.
    pub fn sync_engines(&mut self) {
        if let Err(msg) = self.state.validate() {
            self.diagnostic = Some(DiagnosticReport::new(0.0, 0, "parameters", msg));
            return;
        }
        self.diagnostic = None;
        if let Err(err) = self.vogel.set_parameters(
            self.state.primordia_count,
            self.state.divergence_deg.to_radians(),
            self.state.radial_scale,
        ) {
            self.diagnostic = Some(DiagnosticReport::new(
                0.0,
                0,
                "phyllotaxis",
                err.to_string(),
            ));
            return;
        }
        if self.selected_primordium >= self.vogel.count() {
            self.selected_primordium = 0;
        }
        match self.state.lsystem_preset.to_domain().build_grammar(
            self.state.branch_angle_deg,
            self.state.contraction_r1,
            self.state.contraction_r2,
        ) {
            Ok((grammar, cfg)) => {
                if let Err(err) = self.lsystem.reconfigure(grammar, cfg) {
                    self.diagnostic =
                        Some(DiagnosticReport::new(0.0, 0, "lsystem", err.to_string()));
                    return;
                }
                let _ = self.lsystem.derive_bounded(self.state.generations);
            }
            Err(err) => {
                self.diagnostic = Some(DiagnosticReport::new(0.0, 0, "lsystem", err.to_string()));
            }
        }
        self.bump_revision();
    }

    fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::with_capacity(3);
        let primordia: Vec<[f64; 2]> = self.vogel.points().iter().map(|p| [p.x, p.y]).collect();
        out.push(("Vogel Primordia (x, y)".to_string(), primordia));

        let step = self.vogel.analysis().primary_count.max(1);
        let spiral: Vec<[f64; 2]> = (0..self.vogel.count())
            .step_by(step)
            .map(|idx| {
                let p = self.vogel.points()[idx];
                [p.x, p.y]
            })
            .collect();
        if spiral.len() >= 2 {
            out.push((format!("Primary Parastichy Arm (step {step})"), spiral));
        }

        let segs = self.lsystem.segments();
        let stride = (segs.len() / 2_000).max(1);
        let mut shoot_pts = Vec::with_capacity((segs.len() / stride + 1) * 2);
        for seg in segs.iter().step_by(stride) {
            shoot_pts.push([seg.x0, seg.y0]);
            shoot_pts.push([seg.x1, seg.y1]);
        }
        if !shoot_pts.is_empty() {
            out.push(("L-System Vascular Internodes".to_string(), shoot_pts));
        }
        out
    }

    fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        let primordia: Vec<[f64; 2]> = self.vogel.points().iter().map(|p| [p.x, p.y]).collect();
        if !primordia.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Vogel Primordia".to_string(),
                points: primordia.into(),
                color: palette.success,
                radius: 2.2_f32,
            });
        }
        let step = self.vogel.analysis().primary_count.max(1);
        for arm in 0..step.min(21) {
            let pts: Vec<[f64; 2]> = (arm..self.vogel.count())
                .step_by(step)
                .map(|i| {
                    let p = self.vogel.points()[i];
                    [p.x, p.y]
                })
                .collect();
            if pts.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: if arm == 0 {
                        format!("Parastichy Family ({step})")
                    } else {
                        String::new()
                    },
                    points: pts.into(),
                    color: palette.info,
                    stroke_width: 1.2_f32,
                    closed: false,
                    fill: None,
                });
            }
        }
        let b = self.lsystem.bounds();
        let max_r = self.vogel.points().last().map_or(10.0, |p| p.radius);
        let x_offset = max_r * 2.4 + b.width() * 0.6 + 2.0;
        let segs = self.lsystem.segments();
        let stride = (segs.len() / 1_500).max(1);
        for (idx, seg) in segs.iter().step_by(stride).enumerate() {
            prims.push(ExportPrimitive::Polyline {
                name: if idx == 0 {
                    "L-System Shoot".to_string()
                } else {
                    String::new()
                },
                points: vec![[seg.x0 + x_offset, seg.y0], [seg.x1 + x_offset, seg.y1]].into(),
                color: palette.warning,
                stroke_width: 1.3_f32,
                closed: false,
                fill: None,
            });
        }
        prims
    }
}

impl FiniteStateGuard for PhyllotaxisLSystemTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none() && self.state.validate().is_ok()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        self.state.divergence_deg = if self.state.divergence_deg.is_finite() {
            self.state.divergence_deg.clamp(10.0, 350.0)
        } else {
            GOLDEN_ANGLE_DEG
        };
        self.state.primordia_count = self.state.primordia_count.clamp(8, MAX_PRIMORDIA);
        self.state.radial_scale = if self.state.radial_scale.is_finite() {
            self.state.radial_scale.clamp(0.2, 5.0)
        } else {
            1.0
        };
        self.state.generations = self.state.generations.clamp(1, 11);
        self.state.branch_angle_deg = if self.state.branch_angle_deg.is_finite() {
            self.state.branch_angle_deg.clamp(5.0, 120.0)
        } else {
            25.0
        };
        self.state.contraction_r1 = if self.state.contraction_r1.is_finite() {
            self.state.contraction_r1.clamp(0.35, 0.95)
        } else {
            0.78
        };
        self.state.contraction_r2 = if self.state.contraction_r2.is_finite() {
            self.state.contraction_r2.clamp(0.35, 0.95)
        } else {
            0.68
        };
        self.diagnostic = None;
        self.sync_engines();
    }

    fn reset_parameters(&mut self) {
        self.state = PhyllotaxisLSystemState::default();
        self.diagnostic = None;
        self.sync_engines();
    }
}

impl InteractiveTool for PhyllotaxisLSystemTool {
    fn name(&self) -> &'static str {
        "Phyllotaxis & L-Systems"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.state.primordia_count = (self.state.primordia_count + 21).min(MAX_PRIMORDIA);
        self.sync_engines();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_studio(ui);
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.mode.hash(&mut h);
        self.state.divergence_preset.hash(&mut h);
        self.state.divergence_deg.to_bits().hash(&mut h);
        self.state.primordia_count.hash(&mut h);
        self.state.radial_scale.to_bits().hash(&mut h);
        self.state.show_parastichies.hash(&mut h);
        self.state.lsystem_preset.hash(&mut h);
        self.state.generations.hash(&mut h);
        self.state.branch_angle_deg.to_bits().hash(&mut h);
        self.state.contraction_r1.to_bits().hash(&mut h);
        self.state.contraction_r2.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: PhyllotaxisLSystemState =
            serde_json::from_str(state).map_err(|e| format!("Invalid botanical state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.diagnostic = None;
        self.sync_engines();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PhyllotaxisLSystemTool",
        display_name: "Phyllotaxis & L-Systems",
        domain: "biology",
        tags: &[
            "phyllotaxis",
            "vogel",
            "fibonacci",
            "parastichy",
            "lsystem",
            "lindenmayer",
            "botany",
            "fractal",
        ],
        build: || Box::new(PhyllotaxisLSystemTool::default()),
    }
}
