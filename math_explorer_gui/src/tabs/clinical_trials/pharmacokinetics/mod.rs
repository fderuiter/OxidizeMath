//! Interactive multi-dose pharmacokinetic regimen designer (`PharmacokineticsTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod model;

use self::model::{
    PkRegimenState, PkWorkspace, CHALLENGE_PK_THERAPEUTIC_WINDOW, PK_HORIZON_HOURS, PK_SAMPLE_COUNT,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{
    CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractiveTool, PaneSidePanel,
};
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoint, PlotPoints, VLine};
use std::collections::HashMap;

/// Interactive clinical dosing tool backed by `PKParameters`, `BatemanModel`, `solve_ka`, and `SuperpositionModel`.
pub struct PharmacokineticsTool {
    /// Current oral regimen and patient pharmacokinetic parameters.
    pub state: PkRegimenState,
    /// Active inverse challenge ID when running in challenge mode.
    pub active_challenge: Option<&'static str>,
    /// Preallocated trajectory and scoring workspace.
    pub workspace: PkWorkspace,
    /// Cached plot trajectory points (`481` samples).
    plot_points: Vec<PlotPoint>,
    /// Automated and manual plot callouts.
    pub callout_manager: CalloutManager,
}

impl Default for PharmacokineticsTool {
    fn default() -> Self {
        let mut tool = Self {
            state: PkRegimenState::default(),
            active_challenge: None,
            workspace: PkWorkspace::default(),
            plot_points: vec![PlotPoint::new(0.0, 0.0); PK_SAMPLE_COUNT],
            callout_manager: CalloutManager::new(),
        };
        tool.recalculate();
        tool
    }
}

impl PharmacokineticsTool {
    /// Recomputes `solve_ka`, multi-dose superposition samples, and plot callout positions.
    pub fn recalculate(&mut self) {
        self.state.enforce_challenge_locks(self.active_challenge);
        self.workspace.evaluate(&self.state);
        let dt = PK_HORIZON_HOURS / (PK_SAMPLE_COUNT - 1) as f64;
        if self.plot_points.len() != PK_SAMPLE_COUNT {
            self.plot_points
                .resize(PK_SAMPLE_COUNT, PlotPoint::new(0.0, 0.0));
        }
        for (k, slot) in self.plot_points.iter_mut().enumerate() {
            *slot = PlotPoint::new(k as f64 * dt, self.workspace.concentrations[k]);
        }
        self.workspace
            .sync_callouts(&self.state, &mut self.callout_manager);
    }

    fn draw_regimen_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Tunable Dosing Regimen");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.dose_mg, 0.0..=600.0)
                        .step_by(5.0)
                        .text("Maintenance Dose D (mg)"),
                )
                .accessible_hover_text(
                    "Recurring oral maintenance dose administered every Δt hours",
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.interval_hours, 1.0..=24.0)
                        .step_by(0.5)
                        .text("Interval Δt (h)"),
                )
                .accessible_hover_text("Hours between successive maintenance doses")
                .changed();
            changed |= ui
                .checkbox(
                    &mut self.state.use_loading_dose,
                    "Enable Loading Dose D_load at t = 0 h",
                )
                .accessible_hover_text(
                    "Replace the initial t = 0 maintenance dose with a custom bolus",
                )
                .changed();
            ui.add_enabled_ui(self.state.use_loading_dose, |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.loading_dose_mg, 0.0..=800.0)
                            .step_by(5.0)
                            .text("Loading Dose D_load (mg)"),
                    )
                    .accessible_hover_text("Initial oral loading dose administered at t = 0 h")
                    .changed();
            });
        });
        changed
    }

    fn draw_patient_constants(&mut self, ui: &mut egui::Ui) -> bool {
        let locked = self.active_challenge.is_some();
        let mut changed = false;
        ui.group(|ui| {
            let header = if locked {
                "Patient PK & Window (Locked by Challenge)"
            } else {
                "Patient PK & Therapeutic Window"
            };
            ui.strong(header);
            ui.add_enabled_ui(!locked, |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.bioavailability, 0.1..=1.0)
                            .text("Bioavailability F"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.ke, 0.02..=0.5)
                            .text("Elimination k_e (1/h)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.t_max_target, 0.4..=4.0)
                            .text("Target T_max (h)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.volume_l, 10.0..=100.0)
                            .text("Volume V (L)"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.c_mec, 0.5..=15.0).text("C_MEC (mg/L)"))
                    .changed();
                let min_mtc = (self.state.c_mec + 0.5).min(49.5);
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.c_mtc, min_mtc..=50.0)
                            .text("C_MTC (mg/L)"),
                    )
                    .changed();
            });
        });
        changed
    }

    fn draw_summary_readout(&self, ui: &mut egui::Ui) {
        let s = &self.workspace.summary;
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.strong("Solved ADME & Window Readout");
            ui.label(format!(
                "solve_ka(T_max={:.2}h) → k_a = {:.4} h⁻¹ (t₁/₂ = {:.2} h)",
                s.verified_t_max, s.ka, s.elimination_half_life
            ));
            ui.label(format!(
                "Peak C_max = {:.2} mg/L @ {:.1} h | 12–48h Trough = {:.2} mg/L",
                s.peak_conc.1, s.peak_conc.0, s.steady_trough.1
            ));
            let status_color = if s.passed {
                palette.success
            } else {
                palette.warning
            };
            ui.colored_label(
                status_color,
                format!(
                    "E_tox = {:.2} mg/L | Sub-MEC φ_sub(12–48h) = {:.1}% | Score = {:.1}/100",
                    s.e_tox,
                    s.phi_sub * 100.0,
                    s.score
                ),
            );
        });
    }

    fn draw_concentration_plot(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        Plot::new("pk_superposition_plot")
            .legend(Legend::default())
            .x_axis_label("Time (hours)")
            .y_axis_label("Plasma Concentration C(t) (mg/L)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Plasma C(t)", PlotPoints::Borrowed(&self.plot_points))
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.hline(
                    HLine::new("C_MEC (Min Effective)", self.state.c_mec)
                        .color(palette.success)
                        .width(1.5_f32)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
                plot_ui.hline(
                    HLine::new("C_MTC (Min Toxic)", self.state.c_mtc)
                        .color(palette.error)
                        .width(1.5_f32)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
                plot_ui.vline(
                    VLine::new("Steady Window Start (12h)", 12.0)
                        .color(palette.muted_text)
                        .style(LineStyle::Dotted { spacing: 5.0_f32 }),
                );
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}

impl InteractiveTool for PharmacokineticsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Pharmacokinetic Dosing"
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let dt = PK_HORIZON_HOURS / (PK_SAMPLE_COUNT - 1) as f64;
        let pts = (0..PK_SAMPLE_COUNT)
            .map(|k| [k as f64 * dt, self.workspace.concentrations[k]])
            .collect();
        vec![("Plasma C(t) (mg/L)".to_string(), pts)]
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.dose_mg.to_bits().hash(&mut h);
        self.state.use_loading_dose.hash(&mut h);
        self.state.loading_dose_mg.to_bits().hash(&mut h);
        self.state.interval_hours.to_bits().hash(&mut h);
        self.state.bioavailability.to_bits().hash(&mut h);
        self.state.ke.to_bits().hash(&mut h);
        self.state.t_max_target.to_bits().hash(&mut h);
        self.state.volume_l.to_bits().hash(&mut h);
        self.state.c_mec.to_bits().hash(&mut h);
        self.state.c_mtc.to_bits().hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let mut loaded: PkRegimenState =
            serde_json::from_str(state).map_err(|e| format!("Invalid PK regimen state: {e}"))?;
        loaded.enforce_challenge_locks(self.active_challenge);
        loaded.validate()?;
        self.state = loaded;
        self.recalculate();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_PK_THERAPEUTIC_WINDOW) => {
                self.active_challenge = Some(CHALLENGE_PK_THERAPEUTIC_WINDOW);
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for PharmacokineticsTool"
            )),
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge
            .map(|_| self.workspace.evaluate_challenge())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "pk_dosing_controls");
        let mut changed = false;
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Oral Dosing Regimen");
            ui.separator();
            changed |= self.draw_regimen_controls(ui);
            ui.add_space(6.0);
            changed |= self.draw_patient_constants(ui);
            ui.add_space(6.0);
            self.draw_summary_readout(ui);
        });
        if changed {
            self.recalculate();
        }
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("48-Hour Multi-Dose Plasma Superposition");
            self.draw_concentration_plot(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for PharmacokineticsTool {
    fn theory_description(&self) -> String {
        "One-compartment oral Bateman absorption-elimination superposition: C(t) = Σ_j (F·D_j·k_a / (V·(k_a - k_e))) · (e^(-k_e(t - t_j)) - e^(-k_a(t - t_j))) with k_a inverted from T_max = ln(k_a/k_e)/(k_a - k_e) via Newton-Raphson.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "One compartment oral Bateman pharmacokinetics with multi-dose superposition and therapeutic window bounds".to_string()
    }

    fn theory_citation(&self) -> String {
        "Rowland & Tozer, Clinical Pharmacokinetics and Pharmacodynamics (2011); Bateman (1910)"
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("dose_mg".into(), "Oral maintenance dose D (mg)".into());
        map.insert("interval_hours".into(), "Dosing interval Δt (hours)".into());
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PharmacokineticsTool",
        display_name: "Pharmacokinetic Dosing",
        domain: "clinical_trials",
        tags: &["pharmacokinetics", "adme", "dosing"],
        build: || Box::new(PharmacokineticsTool::default()),
    }
}

const STARTER_PK_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Steady-State Oral Dosing Regimen",
    "domain": "clinical_trials",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "dose_mg": 165.0,
    "use_loading_dose": true,
    "loading_dose_mg": 240.0,
    "interval_hours": 6.0,
    "bioavailability": 0.85,
    "ke": 0.12,
    "t_max_target": 1.5,
    "volume_l": 25.0,
    "c_mec": 4.0,
    "c_mtc": 12.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "pharmacokinetics_oral_starter",
        title: "Steady-State Oral Dosing Regimen",
        description: "C(t) = Σ Bateman(t - k·Δt): Multi-dose oral accumulation within [C_MEC, C_MTC].",
        category: "Pharmacokinetics",
        icon: "💊",
        domain: "clinical_trials",
        tool_name: Some("PharmacokineticsTool"),
        preset_generator: || STARTER_PK_JSON,
    }
}

// [cite:clinical_trials_statistics]
