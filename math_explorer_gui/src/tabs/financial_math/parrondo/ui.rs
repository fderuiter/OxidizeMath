//! Side-panel controls, Kelly `NoEdge` diagnostics, and responsive workspace layout.

use super::plots::to_pairs;
use super::{
    KellyEdgeBadge, ParrondoParadoxTool, ParrondoViewMode, ALLOWED_MODULI, MAX_STEPS, MAX_WALKERS,
    MIN_STEPS, MIN_WALKERS,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};

const SCHEDULE_PRESETS: [&str; 3] = ["ABB", "AABB", "ABAB"];

impl ParrondoParadoxTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "parrondo_paradox_controls");
        PaneSidePanel::left(panel_id)
            .default_width(300.0)
            .show_inside(ui, |panel_ui| {
                self.render_controls_panel(panel_ui);
            });

        egui::CentralPanel::default().show_inside(ui, |central_ui| {
            self.render_central_views(central_ui);
        });
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Markov Ratchet Controls");
        self.render_view_mode_selector(ui);
        ui.separator();

        let mut markov_changed = false;
        markov_changed |= self.render_modulus_and_bias_controls(ui);
        ui.separator();
        markov_changed |= self.render_schedule_controls(ui, &palette);
        ui.separator();
        let ensemble_changed = self.render_ensemble_controls(ui);

        if markov_changed {
            self.recompute_all();
        } else if ensemble_changed {
            self.recompute_ensembles_only();
        }

        ui.separator();
        self.render_kelly_and_ratchet_summary(ui, &palette);
    }

    fn render_view_mode_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Workspace & Export View");
        ui.horizontal_wrapped(|ui| {
            for mode in ParrondoViewMode::ALL {
                let selected = self.view_mode == mode;
                if ui
                    .selectable_label(selected, mode.label())
                    .accessible_hover_text(
                        "Switch active chart focus and single-coordinate figure export target.",
                    )
                    .clicked()
                {
                    self.view_mode = mode;
                }
            }
        });
    }

    fn render_modulus_and_bias_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Capital Modulus (Odd M)");
        ui.horizontal(|ui| {
            for &m in &ALLOWED_MODULI {
                if ui
                    .selectable_value(&mut self.modulus, m, format!("M = {m}"))
                    .accessible_hover_text(
                        "Odd modulo-M state space avoiding bipartite parity oscillations.",
                    )
                    .changed()
                {
                    changed = true;
                }
            }
        });

        changed |= ui
            .add(
                egui::Slider::new(&mut self.epsilon, -0.02..=0.045)
                    .step_by(0.0005)
                    .fixed_decimals(4)
                    .text("Bias Perturbation (ε)"),
            )
            .accessible_hover_text(
                "Uniform coin penalty subtracted from Game A and Game B win probabilities.",
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.p_a_base, 0.35..=0.65)
                    .fixed_decimals(3)
                    .text("Game A Base Win Prob"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.p_b0_base, 0.03..=0.50)
                    .fixed_decimals(3)
                    .text("Game B Trap Prob (W ≡ 0)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.p_b1_base, 0.50..=0.95)
                    .fixed_decimals(3)
                    .text("Game B Non-Trap Prob"),
            )
            .changed();
        changed
    }

    fn render_schedule_controls(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) -> bool {
        let mut changed = false;
        ui.label("Periodic Schedule");
        ui.horizontal_wrapped(|ui| {
            for preset in SCHEDULE_PRESETS {
                let is_active =
                    self.active_schedule.as_string() == preset && self.schedule_error.is_none();
                if ui
                    .selectable_label(is_active, preset)
                    .accessible_hover_text("Apply standard periodic Game A / Game B schedule.")
                    .clicked()
                {
                    changed |= self.apply_schedule_text(preset);
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Custom (A/B):");
            let resp = ui
                .add(
                    egui::TextEdit::singleline(&mut self.schedule_input)
                        .desired_width(120.0)
                        .char_limit(16),
                )
                .accessible_hover_text("Enter 1 to 16 steps using characters A and B.");
            if resp.changed() {
                let candidate = self.schedule_input.clone();
                changed |= self.apply_schedule_text(&candidate);
            }
        });
        if let Some(ref sched_err) = self.schedule_error {
            ui.colored_label(palette.warning, sched_err);
        }
        changed
    }

    fn render_ensemble_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("OxidizeRng Ensemble");
        ui.horizontal(|ui| {
            ui.label("Deterministic Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).range(0..=1_000_000))
                .accessible_hover_text("Deterministic OxidizeRng 64-bit seed.")
                .changed();
        });
        changed |= ui
            .add(
                egui::Slider::new(&mut self.num_walkers, MIN_WALKERS..=MAX_WALKERS).text("Walkers"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_steps, MIN_STEPS..=MAX_STEPS).text("Steps"))
            .changed();

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Rerun Ensemble")
                .accessible_hover_text(
                    "Re-simulate zero-allocation OxidizeRng walker trajectories.",
                )
                .clicked()
            {
                changed = true;
            }
            if ui
                .button("↻ Reset Canonical")
                .accessible_hover_text(
                    "Restore canonical Harmer-Abbott M=3, epsilon=0.005, ABB parameters.",
                )
                .clicked()
            {
                self.reset_defaults();
            }
        });
        changed
    }

    fn render_kelly_and_ratchet_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.heading("Kelly & Detailed Balance");
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        let Some(ref report) = self.report else {
            return;
        };
        let status_color = if report.is_paradox() {
            palette.success
        } else {
            palette.warning
        };
        let status_label = if report.is_paradox() {
            "PARADOX ACTIVE: Losing A + Losing B → Winning Ratchet"
        } else {
            "Outside Paradox Wedge (adjust ε or p_A)"
        };
        ui.colored_label(status_color, egui::RichText::new(status_label).strong());

        let sched_label = format!("Schedule ({})", self.active_schedule.as_string());
        for (name, gain, badge) in [
            ("Pure Game A", report.gain_a, self.kelly_a),
            ("Pure Game B", report.gain_b, self.kelly_b),
            ("Random 50/50 Mix", report.gain_mix, self.kelly_mix),
            (
                sched_label.as_str(),
                report.gain_schedule,
                self.kelly_schedule,
            ),
        ] {
            render_strategy_badge_row(ui, palette, name, gain, badge);
        }

        ui.add_space(4.0);
        ui.small(format!(
            "Detailed-balance max flux residual |π_i P_ij - π_j P_ji|: B = {:.5}, Mix = {:.5}, Sched = {:.5}",
            report.detailed_balance_b,
            report.detailed_balance_mix,
            report.detailed_balance_schedule
        ));
        ui.small(format!(
            "Trap occupancy π(0): A = {:.3}, B = {:.3}, Sched = {:.3} | Solved in {:.2} ms",
            report.pi_a[0], report.pi_b[0], report.pi_schedule_mean[0], self.last_compute_ms
        ));
    }
}

pub(super) fn render_strategy_badge_row(
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    label: &str,
    gain: f64,
    badge: KellyEdgeBadge,
) {
    let badge_color = match badge {
        KellyEdgeBadge::NoEdge { .. } => palette.error,
        KellyEdgeBadge::PositiveEdge { .. } => palette.success,
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("{label}: E[ΔW] = {gain:+.4}"));
        ui.colored_label(
            badge_color,
            egui::RichText::new(format!("[{}]", badge.badge_text())).monospace(),
        );
    });
}

impl ParrondoParadoxTool {
    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.view_mode {
            ParrondoViewMode::SplitOverview | ParrondoViewMode::PhaseWedge => {
                self.export_phase_wedge_primitives(&pal)
            }
            ParrondoViewMode::WealthTrajectories => vec![
                polyline("Game A Ensemble", &self.traj_a, pal.series[0], 1.8),
                polyline("Game B Ensemble", &self.traj_b, pal.series[1], 1.8),
                polyline("50/50 Mix Ensemble", &self.traj_mix, pal.series[2], 1.8),
                polyline("Schedule Ensemble", &self.traj_schedule, pal.series[3], 2.4),
                polyline("Schedule Drift", &self.drift_schedule, pal.success, 1.6),
            ],
            ParrondoViewMode::TrapOccupancy => self.export_occupancy_primitives(&pal),
        }
    }

    fn export_phase_wedge_primitives(&self, pal: &SemanticPalette) -> Vec<ExportPrimitive> {
        let mut prims = vec![
            scatter_prim(
                "Non-Paradox Grid",
                to_pairs(&self.non_paradox_scatter),
                pal.series[5],
                2.2,
            ),
            scatter_prim(
                "Paradox Wedge",
                to_pairs(&self.paradox_scatter),
                pal.success,
                4.2,
            ),
            scatter_prim(
                "Active Operating Point",
                vec![[self.p_a_base - self.epsilon, self.epsilon]],
                pal.accent,
                6.0,
            ),
        ];
        for (name, pts, col) in [
            ("E[dW_A] = 0", &self.boundary_game_a, pal.series[0]),
            ("E[dW_B] = 0", &self.boundary_game_b, pal.series[1]),
            ("E[dW_sched] = 0", &self.boundary_schedule, pal.series[3]),
        ] {
            if pts.len() >= 2 {
                prims.push(polyline(name, pts, col, 2.0));
            }
        }
        prims
    }

    fn export_occupancy_primitives(&self, pal: &SemanticPalette) -> Vec<ExportPrimitive> {
        let Some(ref r) = self.report else {
            return Vec::new();
        };
        let m = self.modulus.min(r.pi_b.len());
        let mut prims = Vec::with_capacity(m * 3);
        for s in 0..m {
            let x = s as f64;
            for (name, dx, val, col) in [
                ("pi_B", -0.22, r.pi_b[s], pal.series[1]),
                ("pi_mix", 0.0, r.pi_mix[s], pal.series[2]),
                ("pi_schedule", 0.22, r.pi_schedule_mean[s], pal.series[3]),
            ] {
                prims.push(ExportPrimitive::Bar {
                    name: format!("{name}[{s}]"),
                    min: [x + dx - 0.09, 0.0],
                    max: [x + dx + 0.09, val],
                    fill: col,
                    stroke_color: pal.accent,
                    stroke_width: 1.0,
                });
            }
        }
        prims
    }
}

fn scatter_prim(
    name: &str,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
    radius: f32,
) -> ExportPrimitive {
    ExportPrimitive::Points {
        name: name.into(),
        points,
        color,
        radius,
    }
}

fn polyline(name: &str, pts: &[PlotPoint], color: egui::Color32, width: f32) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: to_pairs(pts),
        color,
        stroke_width: width,
        closed: false,
        fill: None,
    }
}

// [cite:clinical_trials_statistics]
