#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::overlay::render_morphogenesis_overlay;
use super::presets::{PRESET_GRAY_SCOTT_SPIRAL_FLOW, PRESET_KARMAN_TURING_WAKE};
use super::types::{
    compute_peclet_damkohler, MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisKinetics,
    MorphogenesisKineticsMode, MorphogenesisOverlayMode, MorphogenesisSnapshotData,
};
use super::{MorphogenesisTool, SimpleRng};
use crate::app::{SemanticPalette, ThemeMode};
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::AdvectionDiffusion2D;
use math_explorer::biology::morphogenesis::TuringSystem;

impl MorphogenesisTool {
    fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snapshot_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snapshot_pixels.is_empty() {
            return;
        }

        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let theme_mode = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let overlay_mode = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .overlay_mode;

        if let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
            .filter(|d| d.concentration.len() == num_pixels)
        {
            self.live_peclet = data.peclet;
            self.live_damkohler = data.damkohler;
            let recolored = render_morphogenesis_overlay(
                &data.concentration,
                &data.vorticity,
                &data.obstacles,
                snapshot.width,
                snapshot.height,
                overlay_mode,
                theme_mode,
            );
            self.pixels.copy_from_slice(&recolored.pixels);
        } else if snapshot_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snapshot_pixels);
        }

        let image = ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("morphogenesis_tex", image, texture_opts));
        }
        self.dirty = false;
    }

    pub(crate) fn current_peclet_damkohler(&self) -> (f64, f64) {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let u_est = match p.flow_mode {
            MorphogenesisFlowMode::None => 0.0,
            MorphogenesisFlowMode::LbmVortexWake => p.inlet_velocity,
            MorphogenesisFlowMode::PotentialFlow => {
                let r = ((self.height as f64) / 10.0).max(2.0);
                p.inlet_velocity + p.circulation.abs() / (2.0 * std::f64::consts::PI * r)
            }
        };
        if self.worker.is_running()
            && (self.live_peclet > 0.0 || p.flow_mode == MorphogenesisFlowMode::None)
        {
            (self.live_peclet, self.live_damkohler)
        } else {
            let length = ((self.height as f64) / 6.0).max(1.0);
            compute_peclet_damkohler(u_est, length, p.d_u, p.a, p.b, p.kinetics)
        }
    }

    fn draw_dimensionless_readouts(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (pe, da) = self.current_peclet_damkohler();
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(palette.info, format!("Péclet Pe = {pe:.2}"));
            ui.separator();
            ui.colored_label(palette.accent, format!("Damköhler Da = {da:.2}"));
        });
    }

    fn draw_kinetics_and_reaction_sliders(&mut self, ui: &mut egui::Ui) {
        let mut switched_to = None;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            ui.horizontal(|ui| {
                ui.label("Kinetics:");
                for mode in [
                    MorphogenesisKineticsMode::Schnakenberg,
                    MorphogenesisKineticsMode::GrayScott,
                ] {
                    if ui
                        .selectable_value(&mut p.kinetics, mode, mode.label())
                        .changed()
                    {
                        switched_to = Some(mode);
                    }
                }
            });
            match p.kinetics {
                MorphogenesisKineticsMode::Schnakenberg => {
                    ui.add(egui::Slider::new(&mut p.a, 0.0..=1.0).text("a"));
                    ui.add(egui::Slider::new(&mut p.b, 0.0..=2.0).text("b"));
                    ui.add(egui::Slider::new(&mut p.d_u, 0.1..=5.0).text("D_u"));
                    ui.add(egui::Slider::new(&mut p.d_v, 10.0..=200.0).text("D_v"));
                }
                MorphogenesisKineticsMode::GrayScott => {
                    ui.add(egui::Slider::new(&mut p.a, 0.01..=0.10).text("Feed f"));
                    ui.add(egui::Slider::new(&mut p.b, 0.03..=0.08).text("Kill k"));
                    ui.add(egui::Slider::new(&mut p.d_u, 0.05..=0.50).text("D_u"));
                    ui.add(egui::Slider::new(&mut p.d_v, 0.02..=0.25).text("D_v"));
                }
            }
        }
        if let Some(new_mode) = switched_to {
            {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                match new_mode {
                    MorphogenesisKineticsMode::Schnakenberg => {
                        p.a = 0.14;
                        p.b = 0.86;
                        p.d_u = 1.0;
                        p.d_v = 50.0;
                        p.dt = 0.02;
                    }
                    MorphogenesisKineticsMode::GrayScott => {
                        p.a = 0.037;
                        p.b = 0.060;
                        p.d_u = 0.20;
                        p.d_v = 0.10;
                        p.dt = 0.50;
                    }
                }
            }
            self.theory.set_mode(new_mode);
            self.reset();
        }
    }

    fn draw_flow_controls(&mut self, ui: &mut egui::Ui) {
        let mut flow_changed = false;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            egui::ComboBox::from_label("Flow Field")
                .selected_text(p.flow_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [
                        MorphogenesisFlowMode::None,
                        MorphogenesisFlowMode::LbmVortexWake,
                        MorphogenesisFlowMode::PotentialFlow,
                    ] {
                        if ui
                            .selectable_value(&mut p.flow_mode, mode, mode.label())
                            .changed()
                        {
                            if mode != MorphogenesisFlowMode::None && p.inlet_velocity <= 0.0 {
                                p.inlet_velocity = 0.08;
                            }
                            flow_changed = true;
                        }
                    }
                });
            if p.flow_mode != MorphogenesisFlowMode::None {
                ui.add(egui::Slider::new(&mut p.inlet_velocity, 0.0..=0.22).text("Inlet U"));
            }
            if p.flow_mode == MorphogenesisFlowMode::LbmVortexWake {
                ui.add(egui::Slider::new(&mut p.viscosity, 0.005..=0.15).text("Viscosity ν"));
            }
            if p.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                ui.add(egui::Slider::new(&mut p.circulation, -15.0..=15.0).text("Circulation Γ"));
            }
        }
        if flow_changed {
            self.dirty = true;
        }
    }

    fn draw_overlay_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        let mut clear_obstacles = false;
        let mut overlay_changed = false;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            egui::ComboBox::from_label("Overlay")
                .selected_text(p.overlay_mode.label())
                .show_ui(ui, |ui| {
                    for ov in [
                        MorphogenesisOverlayMode::Concentration,
                        MorphogenesisOverlayMode::Vorticity,
                        MorphogenesisOverlayMode::DualOverlay,
                    ] {
                        overlay_changed |= ui
                            .selectable_value(&mut p.overlay_mode, ov, ov.label())
                            .changed();
                    }
                });
            egui::ComboBox::from_label("Sculpt Brush")
                .selected_text(p.brush_mode.label())
                .show_ui(ui, |ui| {
                    for bm in [
                        MorphogenesisBrushMode::Obstacle,
                        MorphogenesisBrushMode::EraseObstacle,
                        MorphogenesisBrushMode::Reagent,
                        MorphogenesisBrushMode::PanZoom,
                    ] {
                        ui.selectable_value(&mut p.brush_mode, bm, bm.label());
                    }
                });
            ui.add(egui::Slider::new(&mut p.brush_radius, 1..=12).text("Brush r"));
            if ui.button("Clear Obstacles").clicked() {
                clear_obstacles = true;
            }
        }
        if overlay_changed {
            self.dirty = true;
        }
        if clear_obstacles {
            let _ = self.worker.send_command(SimCommand::ClearObstacles);
            self.dirty = true;
        }
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.label("Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Spots").clicked() {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui.button("Stripes").clicked() {
                self.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
            }
            if ui.button("Kármán Wake").clicked() {
                let _ = self.import_preset_json(PRESET_KARMAN_TURING_WAKE);
            }
            if ui.button("GS Swirl").clicked() {
                let _ = self.import_preset_json(PRESET_GRAY_SCOTT_SPIRAL_FLOW);
            }
        });
    }

    fn draw_fault_and_recovery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = render_worker_fault_banner(
            ui,
            self.worker.state(),
            self.worker.diagnostic_report(),
            true,
        ) {
            match action {
                WorkerFaultAction::Reset => self.reset(),
                WorkerFaultAction::ResetParameters => FiniteStateGuard::reset_parameters(self),
                WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                    FiniteStateGuard::auto_scale_step_size(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                    FiniteStateGuard::clamp_parameters(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                    FiniteStateGuard::reset_parameters(self);
                }
            }
            ctx.request_repaint();
            ui.separator();
        }
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Morphogenesis Controls");
                self.draw_dimensionless_readouts(ui);
                ui.separator();

                self.draw_fault_and_recovery(ui, &ctx);
                self.draw_kinetics_and_reaction_sliders(ui);
                ui.separator();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "morphogenesis_transport",
                    "Morphogenesis",
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
                        PlaybackAction::Reset => {
                            self.reset();
                            ctx.request_repaint();
                        }
                    }
                }

                self.draw_presets(ui);
                ui.separator();
                self.draw_flow_controls(ui);
                self.draw_overlay_and_brush_controls(ui);
            });
        });
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);

            if let Some(texture) = &self.texture {
                let w = self.width as f32;
                let h = self.height as f32;
                let (brush_mode, brush_radius) = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    (p.brush_mode, p.brush_radius)
                };
                let allow_pan = brush_mode == MorphogenesisBrushMode::PanZoom;
                let tex_id = texture.id();
                let mut brush_cmd = None;

                Plot::new("morphogenesis_plot")
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .show_axes([false, false])
                    .allow_drag(allow_pan)
                    .allow_zoom(true)
                    .show(ui, |plot_ui| {
                        plot_ui.image(PlotImage::new(
                            "morphogenesis_image",
                            tex_id,
                            PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                            [w, h],
                        ));
                        let (primary, secondary) = ctx.input(|i| {
                            (i.pointer.primary_down(), i.pointer.secondary_down())
                        });
                        if !allow_pan && plot_ui.response().hovered() && (primary || secondary) {
                            if let Some(pos) = plot_ui.pointer_coordinate() {
                                let gx = pos.x.round() as i32;
                                let gy = ((f64::from(h) - 1.0) - pos.y).round() as i32;
                                brush_cmd = Some(SimCommand::ApplyBrush {
                                    cx: gx,
                                    cy: gy,
                                    r: brush_radius,
                                    is_obstacle: primary,
                                });
                            }
                        }
                    });

                if let Some(cmd) = brush_cmd {
                    if self.worker.send_command(cmd).is_ok() {
                        self.dirty = true;
                        ctx.request_repaint();
                    }
                }
            } else {
                render_empty_plot_guidance(
                    ui,
                    "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve Turing patterns.",
                );
            }
        });
    }

    pub(crate) fn initialize_coupled_system(
        system: &mut TuringSystem<2, MorphogenesisKinetics, AdvectionDiffusion2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);
        match system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let u_eq = (system.kinetics.a + system.kinetics.b).max(0.05);
                let v_eq = system.kinetics.b / (u_eq * u_eq);
                for i in 0..n {
                    system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
                    system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
                }
            }
            MorphogenesisKineticsMode::GrayScott => {
                system.u_mut().fill(1.0);
                system.v_mut().fill(0.0);
                let patches = [
                    (width / 2, height / 2),
                    (width / 3, height / 3),
                    ((2 * width) / 3, (2 * height) / 3),
                ];
                let patch_r = (width.min(height) / 10).clamp(2, 6) as isize;
                for (px, py) in patches {
                    for y in 0..height {
                        for x in 0..width {
                            let dx = x as isize - px as isize;
                            let dy = y as isize - py as isize;
                            if dx * dx + dy * dy <= patch_r * patch_r {
                                let idx = flatten_2d_index(x, y, width);
                                system.u_mut()[idx] = 0.50 + rng.range(-0.04, 0.04);
                                system.v_mut()[idx] = 0.25 + rng.range(-0.04, 0.04);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        Self::plot_concentration_with_contrast(data, width, height, false)
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        let theme = if high_contrast {
            ThemeMode::HighContrast
        } else {
            ThemeMode::Dark
        };
        render_morphogenesis_overlay(
            data,
            &[],
            &[],
            width,
            height,
            MorphogenesisOverlayMode::Concentration,
            theme,
        )
    }
}
