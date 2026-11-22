#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Side-panel controls, `PlaybackTransport`, orbital/Turing parameter editors, cutting-plane
//! inspector, and `.obj` Wavefront mesh export dialog for [`super::VolumetricFieldStudioTool`].
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]

use super::VolumetricFieldStudioTool;
use super::state::{
    FieldSourceMode, MAX_VOLUMETRIC_GRID, MIN_VOLUMETRIC_GRID, QuantumObservableChoice, SliceAxis,
    VolumetricRenderMode,
};
use crate::framework::{PaneSidePanel, PlaybackAction, PlaybackTransport, scoped_panel_id};
use crate::volumetric_scene::OPACITY_THRESHOLD;
use crate::web_io::{PlatformInstant, SaveAsDialogConfig, spawn_save_as_file_dialog};
use eframe::egui;
use math_explorer::physics::quantum::OrbitalQuantumNumbers;
use std::sync::mpsc;
use std::time::Duration;

const FRAME_STEP_INTERVAL: Duration = Duration::from_millis(28);
const OBJ_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Wavefront OBJ Mesh",
    extensions: &["obj"],
    fallback_filename: "volumetric_nodal_surface.obj",
    write_error_prefix: "Failed to write OBJ mesh",
};

impl VolumetricFieldStudioTool {
    pub(super) fn poll_async_and_playback(&mut self, ctx: &egui::Context) {
        let completed = self
            .obj_save_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(result) = completed {
            match result {
                Ok(msg) => self.export_status = Some((msg, false)),
                Err(err) => self.export_status = Some((err, true)),
            }
            self.obj_save_rx = None;
        }
        if self.config.running && !self.engine.diverged {
            let now = PlatformInstant::now();
            if now.duration_since(self.last_step_tick) >= FRAME_STEP_INTERVAL {
                self.last_step_tick = now;
                self.engine.step_and_extract_zero_alloc(&self.config, true);
                self.splat_dirty = true;
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "volumetric_field_studio_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Volumetric Field Studio");
            ui.small("3D MarchingCubes & Gaussian3D Splatting");
            ui.separator();
            self.draw_playback_bar(ui);
            ui.separator();
            let mut dirty = self.draw_source_and_model_controls(ui);
            ui.separator();
            dirty |= self.draw_render_and_slice_controls(ui);
            if dirty {
                self.config.clamp_in_place();
                self.refresh_zero_alloc();
            }
            ui.separator();
            self.draw_probe_and_export_section(ui);
            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| {
                    let prev_cam = self.config.camera;
                    self.config.camera.ui(ui);
                    if self.config.camera != prev_cam {
                        self.engine.step_and_extract_zero_alloc(&self.config, false);
                        self.splat_dirty = true;
                    }
                });
        });
    }

    fn draw_playback_bar(&mut self, ui: &mut egui::Ui) {
        let action = PlaybackTransport::in_thread(
            "volumetric_studio_transport",
            "Volumetric Field Studio",
            self.config.running,
        )
        .show(ui);
        if let Some(act) = action {
            match act {
                PlaybackAction::Play => {
                    self.config.running = true;
                    self.last_step_tick = PlatformInstant::now();
                }
                PlaybackAction::Pause => {
                    self.config.running = false;
                }
                PlaybackAction::Step => {
                    self.config.running = false;
                    self.engine.step_and_extract_zero_alloc(&self.config, true);
                    self.splat_dirty = true;
                }
                PlaybackAction::Reset => {
                    self.config.running = false;
                    self.engine.reset_simulation(&self.config);
                    self.splat_dirty = true;
                }
            }
        }
    }

    fn draw_source_and_model_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let prev_src = self.config.source;
        egui::ComboBox::from_label("3D Volume Engine")
            .selected_text(self.config.source.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.config.source,
                    FieldSourceMode::QuantumSuperposition,
                    FieldSourceMode::QuantumSuperposition.label(),
                );
                ui.selectable_value(
                    &mut self.config.source,
                    FieldSourceMode::TuringMorphogenesis,
                    FieldSourceMode::TuringMorphogenesis.label(),
                );
            });
        changed |= self.config.source != prev_src;
        match self.config.source {
            FieldSourceMode::QuantumSuperposition => changed |= self.draw_quantum_controls(ui),
            FieldSourceMode::TuringMorphogenesis => changed |= self.draw_turing_controls(ui),
        }
        changed
    }

    fn draw_quantum_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        ui.label("Quick Orbital States:");
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("1s + 2pz").clicked() {
                self.apply_orbital_quick_preset((1, 0, 0), (2, 1, 0), 0.5, 0.032);
                c = true;
            }
            if ui.small_button("1s + 2s").clicked() {
                self.apply_orbital_quick_preset((1, 0, 0), (2, 0, 0), 0.5, 0.028);
                c = true;
            }
            if ui.small_button("2s Shell").clicked() {
                self.apply_orbital_quick_preset((2, 0, 0), (1, 0, 0), 0.0, 0.015);
                c = true;
            }
            if ui.small_button("3s Shells").clicked() {
                self.apply_orbital_quick_preset((3, 0, 0), (1, 0, 0), 0.0, 0.0025);
                c = true;
            }
            if ui.small_button("3dz² Cone").clicked() {
                self.apply_orbital_quick_preset((3, 2, 0), (2, 0, 0), 0.0, 0.011);
                c = true;
            }
        });
        c |= draw_orbital_triple(ui, "State 1", &mut self.config.n1, &mut self.config.l1, &mut self.config.m1);
        c |= draw_orbital_triple(ui, "State 2", &mut self.config.n2, &mut self.config.l2, &mut self.config.m2);
        c |= ui
            .add(egui::Slider::new(&mut self.config.superposition_mix, 0.0..=1.0).text("Mix |c₂|²"))
            .changed();
        c |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.relative_phase,
                    -std::f64::consts::PI..=std::f64::consts::PI,
                )
                .text("Phase Δφ"),
            )
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.quantum_dt, 0.005..=0.5).text("Quantum dt"))
            .changed();
        let prev_obs = self.config.observable;
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.config.observable,
                QuantumObservableChoice::SignedWavefunction,
                "Re Ψ (± Lobes)",
            );
            ui.selectable_value(
                &mut self.config.observable,
                QuantumObservableChoice::ProbabilityDensity,
                "|Ψ|² Density",
            );
        });
        c || (self.config.observable != prev_obs)
    }

    fn draw_turing_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        c |= ui
            .add(egui::Slider::new(&mut self.config.turing_a, 0.01..=0.5).text("Rate a"))
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.turing_b, 0.1..=2.0).text("Rate b"))
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.turing_du, 0.1..=5.0).text("Activator D_u"))
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.turing_dv, 5.0..=60.0).text("Inhibitor D_v"))
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.turing_dt, 0.001..=0.05).text("Turing dt"))
            .changed();
        if ui.button("⚡ Inject Center Activator Pulse").clicked() {
            self.config.probe_brush_enabled = true;
            self.engine
                .apply_probe_or_pulse(&mut self.config, [0.0, 0.0, 0.0]);
            self.splat_dirty = true;
        }
        c
    }

    fn draw_render_and_slice_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        let prev_mode = self.config.render_mode;
        egui::ComboBox::from_label("3D Renderer")
            .selected_text(self.config.render_mode.label())
            .show_ui(ui, |ui| {
                for mode in [
                    VolumetricRenderMode::MarchingCubesMesh,
                    VolumetricRenderMode::GaussianRadianceCloud,
                    VolumetricRenderMode::HybridShellAndCloud,
                ] {
                    ui.selectable_value(&mut self.config.render_mode, mode, mode.label());
                }
            });
        c |= self.config.render_mode != prev_mode;
        c |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.grid_resolution,
                    MIN_VOLUMETRIC_GRID..=MAX_VOLUMETRIC_GRID,
                )
                .text("Grid Voxels³"),
            )
            .changed();
        c |= ui
            .add(
                egui::Slider::new(&mut self.config.iso_threshold, 0.002..=0.8)
                    .logarithmic(true)
                    .text("Iso Level τ"),
            )
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.config.splat_scale, 0.2..=4.0).text("Splat Scale"))
            .changed();
        ui.separator();
        c |= ui
            .checkbox(&mut self.config.cut_enabled, "Enable 3D Cutting Plane")
            .changed();
        if self.config.cut_enabled {
            let prev_ax = self.config.cut_axis;
            ui.horizontal(|ui| {
                ui.label("Slice Axis:");
                for ax in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
                    ui.selectable_value(&mut self.config.cut_axis, ax, ax.label());
                }
            });
            c |= self.config.cut_axis != prev_ax;
            c |= ui
                .add(egui::Slider::new(&mut self.config.cut_offset, -1.0..=1.0).text("Cut Plane"))
                .changed();
            let nodes = self.engine.count_cutting_plane_radial_nodes(&self.config);
            ui.small(format!(
                "Slice nodal crossings: {nodes} | Contour segments: {}",
                self.engine.slice_contours.len()
            ));
        }
        c
    }

    fn draw_probe_and_export_section(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.config.probe_brush_enabled,
            "Click Viewport to Perturb / Pulse",
        );
        if let Some(p) = self.engine.probe {
            ui.small(format!(
                "Probe ({:.2}, {:.2}, {:.2}): val={:.4}, P_l^m={:.4}",
                p.world_pos[0], p.world_pos[1], p.world_pos[2], p.scalar_value, p.legendre_polar
            ));
        }
        if ui.button("📦 Export Nodal Mesh (.obj)").clicked() {
            self.trigger_obj_export_dialog();
        }
        if let Some((msg, is_err)) = &self.export_status {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let color = if *is_err { palette.error } else { palette.accent };
            ui.colored_label(color, msg);
        }
        let dc = self.engine.splat_ws.density_counts;
        let tris = (self.engine.pos_mesh.indices.len() + self.engine.neg_mesh.indices.len()) / 3;
        ui.small(format!(
            "Mesh: {tris} tris | Splats: {} (Split:{} Clone:{} Keep:{} Prune<{OPACITY_THRESHOLD:.2}:{})",
            self.engine.splat_ws.scene.gaussians.len(),
            dc.split,
            dc.clone,
            dc.keep,
            dc.prune,
        ));
    }

    fn apply_orbital_quick_preset(
        &mut self,
        q1: (u32, u32, i32),
        q2: (u32, u32, i32),
        mix: f64,
        iso: f64,
    ) {
        self.config.n1 = q1.0;
        self.config.l1 = q1.1;
        self.config.m1 = q1.2;
        self.config.n2 = q2.0;
        self.config.l2 = q2.1;
        self.config.m2 = q2.2;
        self.config.superposition_mix = mix;
        self.config.iso_threshold = iso;
    }

    fn trigger_obj_export_dialog(&mut self) {
        match self.export_obj_string() {
            Ok(obj_text) => {
                let default_name = match self.config.source {
                    FieldSourceMode::QuantumSuperposition => format!(
                        "orbital_n{}l{}m{}_nodal_mesh.obj",
                        self.config.n1, self.config.l1, self.config.m1
                    ),
                    FieldSourceMode::TuringMorphogenesis => "turing_3d_gyroid_mesh.obj".to_string(),
                };
                let (tx, rx) = mpsc::channel();
                self.obj_save_rx = Some(rx);
                spawn_save_as_file_dialog(tx, OBJ_SAVE_CONFIG, &default_name, obj_text, |path, _| {
                    format!("Exported Wavefront OBJ: {}", path.display())
                });
            }
            Err(err) => {
                self.export_status = Some((err, true));
            }
        }
    }
}

fn draw_orbital_triple(ui: &mut egui::Ui, label: &str, n: &mut u32, l: &mut u32, m: &mut i32) -> bool {
    let prev = (*n, *l, *m);
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(n).range(1..=6).prefix("n="));
        let max_l = n.saturating_sub(1);
        *l = (*l).min(max_l);
        ui.add(egui::DragValue::new(l).range(0..=max_l).prefix("l="));
        let lim_m = *l as i32;
        *m = (*m).clamp(-lim_m, lim_m);
        ui.add(egui::DragValue::new(m).range(-lim_m..=lim_m).prefix("m="));
    });
    let clamped = OrbitalQuantumNumbers::clamped(*n, *l, *m);
    *n = clamped.n;
    *l = clamped.l;
    *m = clamped.m;
    (*n, *l, *m) != prev
}
