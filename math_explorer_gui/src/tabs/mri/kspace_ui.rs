#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_tool::MriKSpaceTool;
use super::kspace_types::{KSpaceMaskPreset, MriKSpaceSnapshot, PhantomPreset, StudioBrushMode};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{
    bounded_drag_value, InputMode, InteractionContext, InteractiveTool, PlaybackAction,
    PlaybackTransport,
};
use eframe::egui;
use eframe::egui::ColorImage;

#[derive(Clone, Copy, PartialEq, Eq)]
enum StudioCanvasKind {
    Phantom,
    KSpace,
    Recon,
}

impl MriKSpaceTool {
    pub(crate) fn show_studio_inside(&mut self, ui: &mut egui::Ui) {
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.last_high_contrast != Some(high_contrast) {
            self.last_high_contrast = Some(high_contrast);
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .high_contrast = high_contrast;
            self.sync_and_notify("sync_view");
        }

        while let Some(snap) = self.worker.update() {
            if let Some(data) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriKSpaceSnapshot>())
            {
                self.latest = data.clone();
                self.textures_dirty = true;
            }
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ui.ctx().request_repaint();
        }

        self.focused_canvas = None;
        self.draw_studio_side_panel(ui);
        self.draw_studio_central_panel(ui);
    }

    fn sync_studio_textures(&mut self, ctx: &egui::Context) {
        if !self.textures_dirty
            && self.phantom_tex.is_some()
            && self.kspace_tex.is_some()
            && self.recon_tex.is_some()
        {
            return;
        }
        let n = self.latest.grid_size;
        if n == 0 || self.latest.recon_rgba.len() != n * n {
            return;
        }
        update_single_tex(
            ctx,
            &mut self.phantom_tex,
            "mri_phantom_tex",
            n,
            &self.latest.phantom_rgba,
        );
        update_single_tex(
            ctx,
            &mut self.kspace_tex,
            "mri_kspace_tex",
            n,
            &self.latest.kspace_rgba,
        );
        update_single_tex(
            ctx,
            &mut self.recon_tex,
            "mri_recon_tex",
            n,
            &self.latest.recon_rgba,
        );
        self.textures_dirty = false;
    }

    fn draw_phantom_and_grid_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Phantom & Grid Resolution", |ui| {
            ui.horizontal(|ui| {
                ui.label("Grid:");
                for size in [64_usize, 128] {
                    let label = format!("{size}×{size}");
                    if ui
                        .selectable_label(self.config.grid_size == size, label)
                        .clicked()
                        && self.config.grid_size != size
                    {
                        self.config.grid_size = size;
                        self.config.selected_voxel = [size / 2, size / 2];
                        self.config.strokes.clear();
                        self.sync_and_notify("sync_config");
                    }
                }
            });

            let mut changed_phantom = false;
            egui::ComboBox::from_label("Phantom")
                .selected_text(self.config.phantom.label())
                .show_ui(ui, |ui| {
                    for preset in [
                        PhantomPreset::SheppLogan,
                        PhantomPreset::TuringSpots,
                        PhantomPreset::TuringStripes,
                    ] {
                        if ui
                            .selectable_value(&mut self.config.phantom, preset, preset.label())
                            .changed()
                        {
                            changed_phantom = true;
                        }
                    }
                });
            if changed_phantom {
                self.sync_and_notify("sync_config");
            }
            if matches!(
                self.config.phantom,
                PhantomPreset::TuringSpots | PhantomPreset::TuringStripes
            ) && ui
                .checkbox(&mut self.config.evolve_turing, "Live Turing Morphogenesis")
                .changed()
            {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .config = self.config.clone();
            }
        });
    }

    fn draw_rf_and_sequence_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("RF Sequence & Contrast (T1 / T2)", |ui| {
            let mut seq_changed = false;
            seq_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.flip_angle_deg, 5.0..=180.0)
                        .text("Flip α [°]"),
                )
                .changed();
            seq_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.tr, 0.1..=5.0)
                        .logarithmic(true)
                        .text("TR [s] (T1)"),
                )
                .changed();
            seq_changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.te, 0.005..=0.4)
                        .logarithmic(true)
                        .text("TE [s] (T2)"),
                )
                .changed();
            if seq_changed {
                self.sync_and_notify("excite_90");
            }

            ui.horizontal(|ui| {
                if ui.button("⚡ 90° Excite").clicked() {
                    self.trigger_excite_90();
                }
                if ui.button("🔄 180° Spin-Echo Refocus").clicked() {
                    self.trigger_spin_echo_180();
                }
            });
            self.draw_gradient_controls(ui);
        });
    }

    fn draw_gradient_controls(&mut self, ui: &mut egui::Ui) {
        let mut grad_changed = false;
        ui.horizontal(|ui| {
            ui.label("∫Gx dt");
            grad_changed |= ui
                .add(bounded_drag_value(
                    &mut self.config.grad_gx_integral,
                    -5.0e-5..=5.0e-5,
                    5.0e-7,
                ))
                .changed();
            ui.label("∫Gy dt");
            grad_changed |= ui
                .add(bounded_drag_value(
                    &mut self.config.grad_gy_integral,
                    -5.0e-5..=5.0e-5,
                    5.0e-7,
                ))
                .changed();
        });
        if grad_changed {
            self.sync_and_notify("sync_view");
        }
        let [kx, ky] = self.latest.k_coord_cycles_per_m;
        ui.small(format!(
            "k = ({kx:.1}, {ky:.1}) m⁻¹ | φ_grad = {:.2} rad",
            self.latest.selected_grad_phase_rad
        ));
    }

    fn draw_mask_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("k-Space Masks & Brush Studio", |ui| {
            let mut mask_changed = false;
            egui::ComboBox::from_label("Mask")
                .selected_text(self.config.mask_preset.label())
                .show_ui(ui, |ui| {
                    for m in [
                        KSpaceMaskPreset::Full,
                        KSpaceMaskPreset::LowPass,
                        KSpaceMaskPreset::HighPass,
                        KSpaceMaskPreset::Radial,
                        KSpaceMaskPreset::Cartesian,
                    ] {
                        mask_changed |= ui
                            .selectable_value(&mut self.config.mask_preset, m, m.label())
                            .changed();
                    }
                });
            mask_changed |= ui
                .add(egui::Slider::new(&mut self.config.mask_cutoff, 0.05..=1.0).text("Cutoff"))
                .changed();
            mask_changed |= ui
                .add(egui::Slider::new(&mut self.config.radial_spokes, 4..=64).text("Spokes"))
                .changed();
            mask_changed |= ui
                .add(egui::Slider::new(&mut self.config.cartesian_stride, 1..=8).text("Stride"))
                .changed();
            if mask_changed {
                self.sync_and_notify("sync_mask");
            }

            egui::ComboBox::from_label("Brush")
                .selected_text(self.config.brush_mode.label())
                .show_ui(ui, |ui| {
                    for b in [
                        StudioBrushMode::KSpaceMaskErase,
                        StudioBrushMode::KSpaceMaskPaint,
                        StudioBrushMode::KSpaceRfSpike,
                        StudioBrushMode::SpatialFlipAnglePulse,
                        StudioBrushMode::SpatialDeltaB0Pocket,
                    ] {
                        let _ = ui.selectable_value(&mut self.config.brush_mode, b, b.label());
                    }
                });
            if ui
                .add(egui::Slider::new(&mut self.config.brush_radius, 1..=16).text("Radius"))
                .changed()
            {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .config = self.config.clone();
            }
            if ui.button("🧹 Clear Painted Strokes").clicked() {
                self.clear_painted_strokes();
            }
        });
    }

    fn draw_studio_side_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "mri_kspace_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("2D MRI Studio");
            ui.separator();

            if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
                render_worker_fault_banner(
                    ui,
                    self.worker.state(),
                    self.worker.diagnostic_report(),
                    false,
                )
            {
                self.reset();
                ctx.request_repaint();
                ui.separator();
            }

            self.draw_phantom_and_grid_controls(ui);
            self.draw_rf_and_sequence_controls(ui);
            self.draw_mask_and_brush_controls(ui);
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "mri_kspace_transport",
                "MRI k-Space Studio",
                self.worker.state(),
            )
            .can_reset(true)
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => {
                        let _ = self.worker.send_command(SimCommand::Start);
                    }
                    PlaybackAction::Pause => {
                        let _ = self.worker.send_command(SimCommand::Pause);
                    }
                    PlaybackAction::Step => {
                        let _ = self.worker.send_command(SimCommand::Step(1));
                    }
                    PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                    PlaybackAction::Reset => {
                        self.reset();
                        ctx.request_repaint();
                    }
                }
            }

            ui.separator();
            self.draw_selected_voxel_telemetry(ui);
        });
    }

    fn draw_interactive_tile(
        &mut self,
        ui: &mut egui::Ui,
        kind: StudioCanvasKind,
        title: &str,
        tile_side: f32,
    ) {
        let tex_id = match kind {
            StudioCanvasKind::Phantom => self.phantom_tex.as_ref().map(egui::TextureHandle::id),
            StudioCanvasKind::KSpace => self.kspace_tex.as_ref().map(egui::TextureHandle::id),
            StudioCanvasKind::Recon => self.recon_tex.as_ref().map(egui::TextureHandle::id),
        };
        ui.vertical(|ui| {
            ui.strong(title);
            let Some(tid) = tex_id else {
                return;
            };
            let alt = format!("{title} ({0}x{0} MRI canvas)", self.latest.grid_size);
            let response = ui.add(
                egui::Image::new(egui::load::SizedTexture::new(tid, [tile_side, tile_side]))
                    .alt_text(alt.clone())
                    .sense(egui::Sense::click_and_drag()),
            );
            let val_summary = format!(
                "Selected voxel ({}, {}), |M_xy| = {:.3}",
                self.config.selected_voxel[0],
                self.config.selected_voxel[1],
                self.latest.selected_net_m.xy().norm()
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Canvas,
                &alt,
                Some(&val_summary),
            );
            lock_canvas_arrow_keys(ui, &response);
            paint_canvas_focus_ring(ui, &response);
            if response.has_focus() {
                self.focused_canvas = Some(response.id);
            }
            self.handle_tile_interaction(ui, &response, kind);
            if kind != StudioCanvasKind::KSpace {
                self.paint_voxel_crosshair(ui, response.rect);
            }
        });
    }

    fn handle_tile_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        kind: StudioCanvasKind,
    ) {
        let keys_down = ui.input(|i| i.keys_down.clone());
        let modifiers = ui.input(|i| i.modifiers);
        let ictx = InteractionContext {
            pointer_pos: response.interact_pointer_pos(),
            delta: response.drag_delta(),
            is_dragging: response.dragged(),
            is_clicked: response.clicked(),
            response,
            input_mode: InputMode::Mouse,
            multi_touch: ui.input(|i| i.multi_touch()),
            keys_down,
            modifiers,
        };
        if response.has_focus() {
            self.on_keyboard(&ictx);
        }
        let Some(pos) = ictx.pointer_pos else {
            return;
        };
        let (cx, cy) = self.map_pointer_to_grid(response.rect, pos);
        match kind {
            StudioCanvasKind::KSpace if ictx.is_dragging || ictx.is_clicked => {
                if self.config.brush_mode.targets_kspace() {
                    self.on_brush(&ictx);
                } else {
                    self.apply_brush_at_grid(StudioBrushMode::KSpaceMaskErase, cx, cy);
                }
            }
            StudioCanvasKind::Phantom
                if ictx.is_dragging && !self.config.brush_mode.targets_kspace() =>
            {
                self.on_brush(&ictx);
            }
            StudioCanvasKind::Phantom | StudioCanvasKind::Recon if ictx.is_clicked => {
                if kind == StudioCanvasKind::Phantom
                    && !self.config.brush_mode.targets_kspace()
                    && ictx.modifiers.shift
                {
                    self.apply_brush_at_grid(self.config.brush_mode, cx, cy);
                } else {
                    self.config.selected_voxel = [cy as usize, cx as usize];
                    self.sync_and_notify("sync_view");
                }
            }
            _ => {}
        }
    }

    fn draw_studio_central_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_studio_textures(&ctx);
            let avail = ui.available_size();
            let tile_side = ((avail.x - 48.0) / 3.0)
                .min(avail.y * 0.44)
                .clamp(110.0, 260.0);

            ui.horizontal(|ui| {
                self.draw_interactive_tile(
                    ui,
                    StudioCanvasKind::Phantom,
                    "1. Spatial Tissue Phantom",
                    tile_side,
                );
                self.draw_interactive_tile(
                    ui,
                    StudioCanvasKind::KSpace,
                    "2. Centered k-Space log|S(k)|",
                    tile_side,
                );
                self.draw_interactive_tile(
                    ui,
                    StudioCanvasKind::Recon,
                    "3. Reconstructed Slice (1/RC IDFT)",
                    tile_side,
                );
            });
            ui.separator();
            self.draw_bloch_3d_readout(ui);
        });
    }
}

fn update_single_tex(
    ctx: &egui::Context,
    slot: &mut Option<egui::TextureHandle>,
    name: &str,
    size: usize,
    pixels: &[egui::Color32],
) {
    let image = ColorImage {
        size: [size, size],
        pixels: pixels.to_vec(),
        ..Default::default()
    };
    let opts = egui::TextureOptions::NEAREST;
    if let Some(tex) = slot {
        tex.set(image, opts);
    } else {
        *slot = Some(ctx.load_texture(name, image, opts));
    }
}
