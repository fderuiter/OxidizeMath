#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Side-panel controls, `PlaybackTransport`, orbital/Turing/implicit CAD/AST parameter editors,
//! cutting-plane inspector, and millimeter export sizing for [`super::VolumetricFieldStudioTool`].
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]

use super::compute_wgsl::draw_gpu_shader_preview_section;
use super::state::{
    FieldSourceMode, QuantumObservableChoice, SliceAxis, VolumetricRenderMode, MAX_EXPORT_SIZE_MM,
    MIN_EXPORT_SIZE_MM, MIN_VOLUMETRIC_GRID,
};
use super::webgpu_runner::{ComputeBackend, WebGpuAdapterStatus};
use super::VolumetricFieldStudioTool;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel, PlaybackAction, PlaybackTransport};
use crate::volumetric_scene::OPACITY_THRESHOLD;
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::quantum::OrbitalQuantumNumbers;
use std::time::Duration;

const FRAME_STEP_INTERVAL: Duration = Duration::from_millis(28);

impl VolumetricFieldStudioTool {
    pub(super) fn poll_async_and_playback(&mut self, ctx: &egui::Context) {
        if self
            .webgpu_runner
            .poll_worker_into(&self.config, &mut self.engine.voxel_grid)
        {
            self.engine.extract_current_grid_zero_alloc(&self.config);
            self.splat_dirty = true;
            ctx.request_repaint();
        }
        if self.config.running && !self.engine.diverged {
            let now = PlatformInstant::now();
            if now.duration_since(self.last_step_tick) >= FRAME_STEP_INTERVAL {
                self.last_step_tick = now;
                self.step_playback_frame();
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "volumetric_field_studio_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Volumetric Field Studio");
            ui.small("3D Implicit CAD, MarchingCubes & Gaussian3D");
            ui.separator();
            if !matches!(
                self.config.source,
                FieldSourceMode::CustomImplicit | FieldSourceMode::CustomImplicitAst
            ) {
                self.draw_playback_bar(ui);
                ui.separator();
            }
            let mut dirty = self.draw_source_and_model_controls(ui);
            ui.separator();
            dirty |= self.draw_render_and_slice_controls(ui);
            let pointer_down = ui.input(|i| i.pointer.any_down());
            if dirty {
                self.config.clamp_in_place();
                let drag_preview =
                    self.config.source == FieldSourceMode::CustomImplicitAst && pointer_down;
                self.refresh_with_drag_preview(drag_preview);
            } else if self.engine.preview_drag_active && !pointer_down {
                self.refresh_with_drag_preview(false);
            }
            ui.separator();
            self.draw_probe_and_export_section(ui);
            ui.separator();
            draw_gpu_shader_preview_section(
                ui,
                &self.config,
                &mut self.webgpu_runner.shader_preview,
            );
            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(false)
                .show(ui, |ui| {
                    let prev_cam = self.config.camera;
                    self.config.camera.ui(ui);
                    if self.config.camera != prev_cam {
                        self.engine.extract_current_grid_zero_alloc(&self.config);
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
                    self.step_zero_alloc();
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    self.config.running = false;
                    self.webgpu_runner.reset();
                    self.engine.reset_simulation(&self.config);
                    self.refresh_zero_alloc();
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
                for mode in [
                    FieldSourceMode::QuantumSuperposition,
                    FieldSourceMode::TuringMorphogenesis,
                    FieldSourceMode::CustomImplicit,
                    FieldSourceMode::CustomImplicitAst,
                ] {
                    ui.selectable_value(&mut self.config.source, mode, mode.label());
                }
            });
        changed |= self.config.source != prev_src;
        match self.config.source {
            FieldSourceMode::QuantumSuperposition => changed |= self.draw_quantum_controls(ui),
            FieldSourceMode::TuringMorphogenesis => changed |= self.draw_turing_controls(ui),
            FieldSourceMode::CustomImplicit => {
                changed |= self.implicit_ui.draw_controls(ui, &mut self.config);
            }
            FieldSourceMode::CustomImplicitAst => {
                let prev_expr = self.config.custom_expr.clone();
                changed |= self.ast_controller.ui(
                    ui,
                    &mut self.config.custom_expr,
                    &mut self.config.custom_params,
                );
                if self.config.custom_expr != prev_expr
                    && self.config.custom_expr == super::ast_field::ORBITAL_4F_XYZ_EXPR
                    && self.config.iso_threshold > 0.06
                {
                    self.config.iso_threshold = 0.045;
                }
            }
        }
        changed
    }

    fn draw_quick_orbital_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        type QuickOrbitalPreset = (&'static str, (u32, u32, i32), (u32, u32, i32), f64, f64);
        const PRESETS: &[QuickOrbitalPreset] = &[
            ("1s + 2pz", (1, 0, 0), (2, 1, 0), 0.5, 0.032),
            ("1s + 2s", (1, 0, 0), (2, 0, 0), 0.5, 0.028),
            ("2s Shell", (2, 0, 0), (1, 0, 0), 0.0, 0.015),
            ("3s Shells", (3, 0, 0), (1, 0, 0), 0.0, 0.0025),
            ("3dz² Cone", (3, 2, 0), (2, 0, 0), 0.0, 0.011),
        ];
        let mut c = false;
        ui.label("Quick Orbital States:");
        ui.horizontal_wrapped(|ui| {
            for &(label, s1, s2, mix, iso) in PRESETS {
                if ui.small_button(label).clicked() {
                    self.apply_orbital_quick_preset(s1, s2, mix, iso);
                    c = true;
                }
            }
        });
        c
    }

    fn draw_quantum_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = self.draw_quick_orbital_buttons(ui);
        c |= draw_orbital_triple(
            ui,
            "State 1",
            &mut self.config.n1,
            &mut self.config.l1,
            &mut self.config.m1,
        );
        c |= draw_orbital_triple(
            ui,
            "State 2",
            &mut self.config.n2,
            &mut self.config.l2,
            &mut self.config.m2,
        );
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
        let mut c = self.draw_compute_backend_toggle(ui);
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
        let max_grid = self.config.max_grid_resolution();
        self.config.grid_resolution = self
            .config
            .grid_resolution
            .clamp(MIN_VOLUMETRIC_GRID, max_grid);
        c |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.grid_resolution,
                    MIN_VOLUMETRIC_GRID..=max_grid,
                )
                .text("Grid Voxels³"),
            )
            .changed();
        self.draw_voxel_latency_caption(ui);
        if self.config.source != FieldSourceMode::CustomImplicit {
            c |= ui
                .add(
                    egui::Slider::new(&mut self.config.iso_threshold, 0.002..=0.8)
                        .logarithmic(true)
                        .text("Iso Level τ"),
                )
                .changed();
        }
        c |= ui
            .add(egui::Slider::new(&mut self.config.splat_scale, 0.2..=4.0).text("Splat Scale"))
            .changed();
        ui.separator();
        c |= self.draw_cutting_plane_controls(ui);
        c
    }

    fn draw_voxel_latency_caption(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let active_dim = self.active_voxel_dimension();
        let backend_tag = self
            .webgpu_runner
            .adapter_status
            .hud_backend_label(self.config.compute_backend);
        let msg = if self.config.compute_backend == ComputeBackend::WebGpuCompute {
            let vox = self.config.grid_resolution.pow(3);
            format!(
                "16³ .. 128³  ({vox} voxels) · {:.1} ms",
                self.webgpu_runner.last_step_ms
            )
        } else {
            format!(
                "Latency: {:.1} ms ({backend_tag} {active_dim}³)",
                self.webgpu_runner.last_step_ms
            )
        };
        ui.colored_label(palette.muted_text, msg);
    }

    fn draw_cutting_plane_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = ui
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
        if self.config.render_mode.uses_mesh() {
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.config.export_size_mm,
                        MIN_EXPORT_SIZE_MM..=MAX_EXPORT_SIZE_MM,
                    )
                    .suffix(" mm")
                    .text("Print Size"),
                )
                .changed()
            {
                self.config.clamp_in_place();
            }
            ui.small("Use toolbar Export STL / PLY / OBJ for watertight 3D solids.");
        } else {
            ui.small("Use toolbar Export PLY for INRIA-compatible 3DGS radiance splats.");
        }
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

    fn draw_compute_backend_toggle(&mut self, ui: &mut egui::Ui) -> bool {
        let prev = self.config.compute_backend;
        let gpu_available =
            self.webgpu_runner.adapter_status != WebGpuAdapterStatus::BrowserWorkerFallback;
        ui.label("Compute Backend");
        ui.horizontal(|ui| {
            ui.add_enabled_ui(gpu_available, |ui| {
                ui.selectable_value(
                    &mut self.config.compute_backend,
                    ComputeBackend::WebGpuCompute,
                    ComputeBackend::WebGpuCompute.label(),
                )
                .on_disabled_hover_text(
                    "navigator.gpu unavailable in this browser; clamped to <= 48³ CPU Zero-Alloc",
                );
            });
            ui.selectable_value(
                &mut self.config.compute_backend,
                ComputeBackend::CpuZeroAlloc,
                ComputeBackend::CpuZeroAlloc.label(),
            );
        });
        if !gpu_available {
            self.config.compute_backend = ComputeBackend::CpuZeroAlloc;
        }
        if self.config.compute_backend != prev {
            self.config.clamp_in_place();
            true
        } else {
            false
        }
    }
}

fn draw_orbital_triple(
    ui: &mut egui::Ui,
    label: &str,
    n: &mut u32,
    l: &mut u32,
    m: &mut i32,
) -> bool {
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
