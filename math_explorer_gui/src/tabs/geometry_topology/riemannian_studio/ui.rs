#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, telemetry readouts, and sonification panel for [`super::RiemannianPdeStudio`].

use super::super::surface_brush::{BrushPolarity, ViewportBrushMode};
use super::audio::synthesize_manifold_frame;
use super::handoff::RiemannianPresetTarget;
#[cfg(feature = "applied")]
use super::state::FreeSurferOverlayKind;
use super::state::{ManifoldKind, PdeRegime, MAX_MANIFOLD_VERTICES};
use super::RiemannianPdeStudio;
use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractiveTool, PaneSidePanel, PlaybackAction, PlaybackTransport};
use eframe::egui;

impl RiemannianPdeStudio {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_initialized();
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "riemannian_pde_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Riemannian PDE Studio");
            ui.separator();

            self.draw_transport_bar(ui);
            self.draw_preset_launcher(ui);
            ui.separator();
            self.draw_manifold_controls(ui);
            ui.separator();
            self.draw_regime_controls(ui);
            ui.separator();
            self.draw_brush_controls(ui);
            ui.separator();
            self.draw_telemetry_panel(ui);

            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.config.camera.ui(ui));

            if let (Some(geom), Some(eng)) = (&self.geometry, &self.engine) {
                let frame = synthesize_manifold_frame(
                    geom,
                    eng,
                    self.config.regime.coerce_to_enabled_features(),
                );
                crate::audio::render_sonification_panel(
                    ui,
                    "riemannian_pde_studio",
                    &frame,
                    !self.config.running,
                );
            }
        });
    }

    fn draw_transport_bar(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "riemannian_pde_transport",
            "Riemannian PDE Studio",
            self.config.running,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.config.running = true,
                PlaybackAction::Pause => {
                    self.config.running = false;
                    crate::audio::notify_playback_paused();
                }
                PlaybackAction::Step => self.step_once(),
                PlaybackAction::Reset => {
                    self.config.running = false;
                    crate::audio::notify_playback_reset();
                    self.rebuild_all();
                }
            }
        }
    }

    fn draw_preset_launcher(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("⭐ Curated 3D Scenarios", |ui| {
            for target in RiemannianPresetTarget::ALL {
                if ui.button(target.label()).clicked() {
                    if self.import_preset_json(target.preset_json()).is_ok() {
                        self.rebuild_all();
                        crate::accessibility::announce_live_status(
                            ui.ctx(),
                            format!("Loaded scenario {}", target.label()),
                        );
                    }
                    ui.close();
                }
            }
        });
    }

    fn draw_manifold_controls(&mut self, ui: &mut egui::Ui) {
        let mut rebuild_geom = false;
        let prev_manifold = self.config.manifold;
        egui::ComboBox::from_label("Manifold")
            .selected_text(self.config.manifold.label())
            .show_ui(ui, |ui| {
                for kind in ManifoldKind::ALL {
                    ui.selectable_value(&mut self.config.manifold, kind, kind.label());
                }
            });
        rebuild_geom |= self.config.manifold != prev_manifold;

        rebuild_geom |= ui
            .add(egui::Slider::new(&mut self.config.u_res, 10..=56).text("Grid U"))
            .changed();
        rebuild_geom |= ui
            .add(egui::Slider::new(&mut self.config.v_res, 10..=56).text("Grid V"))
            .changed();
        while self.config.estimated_vertices() > MAX_MANIFOLD_VERTICES && self.config.v_res > 8 {
            self.config.v_res -= 1;
        }

        if self.config.manifold == ManifoldKind::FoldedCorticalShell {
            rebuild_geom |= ui
                .add(
                    egui::Slider::new(&mut self.config.sulcal_depth, 0.0..=0.42)
                        .text("Sulcal Depth"),
                )
                .changed();
        }
        if let Some(st) = self.config.manifold.as_custom_surface_type() {
            let fp_before = self.custom_sandbox.state_fingerprint(st);
            self.custom_sandbox.ui_for_type(ui, st);
            rebuild_geom |= self.custom_sandbox.state_fingerprint(st) != fp_before;
        }
        if rebuild_geom {
            self.rebuild_all();
        }
    }

    fn draw_regime_controls(&mut self, ui: &mut egui::Ui) {
        let prev_regime = self.config.regime;
        egui::ComboBox::from_label("PDE Regime")
            .selected_text(self.config.regime.label())
            .show_ui(ui, |ui| {
                for &reg in PdeRegime::available_regimes() {
                    ui.selectable_value(&mut self.config.regime, reg, reg.label());
                }
            });
        if self.config.regime != prev_regime {
            self.sync_active_regime();
        }

        ui.add(
            egui::Slider::new(&mut self.config.dt, 5e-4..=0.03)
                .logarithmic(true)
                .text("Step dt"),
        );

        match self.config.regime.coerce_to_enabled_features() {
            PdeRegime::HeatBeltrami => {
                ui.add(
                    egui::Slider::new(&mut self.config.heat_alpha, 0.005..=0.50)
                        .text("α (Thermal)"),
                );
            }
            PdeRegime::MeanCurvatureSmoothing => {
                ui.small("Intrinsic cotangent smoothing ∂r/∂t = H·n.");
            }
            PdeRegime::SchnakenbergTuring => {
                ui.add(egui::Slider::new(&mut self.config.schnak_a, 0.02..=0.40).text("a"));
                ui.add(egui::Slider::new(&mut self.config.schnak_b, 0.30..=1.80).text("b"));
                ui.add(egui::Slider::new(&mut self.config.d_u, 5e-4..=0.05).text("D_u"));
                ui.add(egui::Slider::new(&mut self.config.d_v, 0.01..=0.25).text("D_v"));
            }
            PdeRegime::GrayScottTuring => {
                ui.add(egui::Slider::new(&mut self.config.gs_f, 0.010..=0.080).text("Feed f"));
                ui.add(egui::Slider::new(&mut self.config.gs_k, 0.035..=0.075).text("Kill k"));
                ui.add(egui::Slider::new(&mut self.config.d_u, 5e-4..=0.03).text("D_u"));
                ui.add(egui::Slider::new(&mut self.config.d_v, 5e-4..=0.03).text("D_v"));
            }
            PdeRegime::CorticalHodgkinHuxley => {
                ui.add(
                    egui::Slider::new(&mut self.config.hh_conductance, 0.02..=1.20)
                        .text("Sheet g_s"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.hh_stimulus, 0.0..=40.0)
                        .text("Stimulus µA/cm²"),
                );
            }
            PdeRegime::FreeSurferMorphometryGlm => {
                #[cfg(feature = "applied")]
                self.draw_freesurfer_controls(ui);
            }
        }
    }

    #[cfg(feature = "applied")]
    fn draw_freesurfer_controls(&mut self, ui: &mut egui::Ui) {
        let prev_ov = self.config.overlay;
        egui::ComboBox::from_label("Overlay")
            .selected_text(self.config.overlay.label())
            .show_ui(ui, |ui| {
                for ov in FreeSurferOverlayKind::ALL {
                    ui.selectable_value(&mut self.config.overlay, ov, ov.label());
                }
            });
        if self.config.overlay != prev_ov {
            self.sync_active_regime();
        }
        ui.add(
            egui::Slider::new(&mut self.config.fs_grad_strength, 0.05..=1.50)
                .text("T1 Gradient k_ext"),
        );
        if ui
            .button("🧬 Evolve Pial Surface Step")
            .accessible_hover_text("Advance FreeSurfer pial surface one deformable energy step")
            .clicked()
        {
            self.step_once();
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Interaction:");
            ui.selectable_value(
                &mut self.config.brush_mode,
                ViewportBrushMode::Orbit,
                "🧭 Orbit 3D",
            )
            .accessible_hover_text("Primary drag orbits the 3D manifold camera");
            ui.selectable_value(
                &mut self.config.brush_mode,
                ViewportBrushMode::Sculpt,
                "🖌 Sculpt Geodesic",
            )
            .accessible_hover_text("Primary drag and Arrow+Space paint geodesic perturbations");
        });
        if self.config.brush_mode == ViewportBrushMode::Sculpt {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.config.brush_polarity,
                    BrushPolarity::Excite,
                    "+ Excite",
                );
                ui.selectable_value(
                    &mut self.config.brush_polarity,
                    BrushPolarity::Inhibit,
                    "− Inhibit",
                );
            });
            ui.add(
                egui::Slider::new(&mut self.config.brush_radius, 0.08..=1.00).text("Brush Radius"),
            );
            ui.add(
                egui::Slider::new(&mut self.config.brush_strength, 0.10..=1.80)
                    .text("Brush Strength"),
            );
        }
        self.brush.mode = self.config.brush_mode;
        self.brush.polarity = self.config.brush_polarity;
        self.brush.radius = self.config.brush_radius;
        self.brush.strength = self.config.brush_strength;
    }

    fn draw_telemetry_panel(&self, ui: &mut egui::Ui) {
        let (Some(geom), Some(eng)) = (&self.geometry, &self.engine) else {
            return;
        };
        egui::CollapsingHeader::new("Manifold & Morphometry Telemetry")
            .default_open(true)
            .show(ui, |ui| {
                let c_idx = self
                    .brush
                    .cursor_vertex
                    .min(geom.positions.len().saturating_sub(1));
                let k_curv = geom.gaussian_curvature.get(c_idx).copied().unwrap_or(0.0);
                let ((u0, u1), (v0, v1)) = geom.uv_domain;
                ui.small(format!(
                    "Vertices: {} | Triangles: {} | Step: {}",
                    geom.positions.len(),
                    geom.triangles.len(),
                    eng.step_count
                ));
                ui.small(format!(
                    "Chart u∈[{u0:.2},{u1:.2}] v∈[{v0:.2},{v1:.2}] | Cursor K: {k_curv:+.3}"
                ));
                ui.small(format!(
                    "Cotangent vs Y₁⁰ L² rel err: {:.3e} | Cursor Δ_S: {:+.3}",
                    eng.chart_beltrami_l2_err, eng.cursor_beltrami_val
                ));
                ui.small(format!(
                    "Chart HeatEquationSolver peak: {:.4}",
                    eng.chart_heat_solver_peak
                ));
                #[cfg(feature = "applied")]
                {
                    let fs = &eng.freesurfer;
                    let max_t = fs
                        .glm_t_stat
                        .iter()
                        .copied()
                        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
                    ui.small(format!(
                        "Cursor cortical thickness: {:.2} mm | Peak |t|: {:.2}",
                        fs.cursor_probe_thickness_mm, max_t
                    ));
                    ui.small(format!(
                        "Pial energy E_int: {:.2} | E_ext: {:.2}",
                        fs.pial_internal_energy, fs.pial_external_energy
                    ));
                }
            });
    }
}
