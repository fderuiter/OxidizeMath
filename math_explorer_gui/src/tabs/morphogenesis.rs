use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::{InteractivePreset, PresetDialogState};
use crate::tabs::ExplorerTab;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::heatmap_color;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
struct MorphoState {
    dt: f64,
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    paused: Option<bool>,
    #[serde(default)]
    show_starter_gallery: bool,
}

#[path = "morphogenesis/runner.rs"]
mod runner_impl;
pub use runner_impl::*;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTab {
    pub worker: BackgroundWorker<MorphogenesisRunner>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    width: usize,
    height: usize,
    texture: Option<egui::TextureHandle>,
    dialog_state: PresetDialogState,
    pixels: Vec<egui::Color32>,
    dirty: bool,
    pub show_starter_gallery: bool,
}

impl MorphogenesisTab {
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        use math_explorer::biology::morphogenesis::DiffusionCoeff;
        use math_explorer::math_kernel::types::{Dimension, StepSize};
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap();
            (p.a, p.b, p.d_u, p.d_v)
        };
        let kinetics = SchnakenbergKinetics { a, b };
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );
        Self::initialize_system(&mut system, width, height);
        MorphogenesisRunner {
            system,
            params: Arc::clone(params),
            width,
            height,
        }
    }

    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.worker = BackgroundWorker::new(runner);
        self.dirty = true;
    }

    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
        }));

        let runner = Self::create_runner(&params, width, height);
        let mut worker = BackgroundWorker::new(runner);
        let _ = worker.send_command(SimCommand::Start);

        Self {
            worker,
            params,
            width,
            height,
            texture: None,
            dialog_state: PresetDialogState::default(),
            pixels: Vec::new(),
            dirty: true,
            show_starter_gallery: false,
        }
    }
}

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MorphoState {
            dt: self.params.read().unwrap().dt,
            state: Some(self.worker.state()),
            paused: None,
            show_starter_gallery: self.show_starter_gallery,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<MorphoState>(state) {
            self.params.write().unwrap().dt = s.dt;
            self.show_starter_gallery = false;
            let target = s.state.unwrap_or_else(|| {
                if s.paused.unwrap_or(false) {
                    SimWorkerState::Ready
                } else {
                    SimWorkerState::Running
                }
            });
            if target == SimWorkerState::Running && !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            } else if target != SimWorkerState::Running && self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            self.dirty = true;
        }
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn export_preset_json(&self) -> Result<String, String> {
        self.to_json_string()
    }

    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        self.from_json_str(json_str)
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.worker.update().is_some() {
            self.dirty = true;
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_controls(ctx);
        self.draw_canvas(ctx);

        if self.show_starter_gallery {
            let cards = crate::starter_gallery::get_starter_cards_for_domain("biology", &[]);
            let mut is_open = self.show_starter_gallery;
            if let Some(card) = crate::starter_gallery::show_starter_gallery_overlay(
                ctx,
                "morphogenesis",
                &cards,
                &mut is_open,
            ) {
                if let Some(preset_json) = card.preset_json {
                    let _ = self.from_json_str(preset_json);
                }
                is_open = false;
            }
            self.show_starter_gallery = is_open;
        }
    }
}

impl FiniteStateGuard for MorphogenesisTab {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted
            && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !self.is_finite_state()
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap();
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap();
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap();
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
    }
}

impl MorphogenesisTab {
    pub(crate) fn reset_system(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.worker = BackgroundWorker::new(runner);
        let _ = self.worker.send_command(SimCommand::Start);
        self.dirty = true;
    }

    fn draw_parameter_sliders(&self, ui: &mut egui::Ui) {
        let mut params = self.params.write().unwrap();
        ui.add(egui::Slider::new(&mut params.a, 0.0..=1.0).text("a"));
        ui.add(egui::Slider::new(&mut params.b, 0.0..=2.0).text("b"));
        ui.add(egui::Slider::new(&mut params.d_u, 0.1..=5.0).text("D_u"));
        ui.add(egui::Slider::new(&mut params.d_v, 10.0..=200.0).text("D_v"));
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Presets:");
        if ui.button("Spots").clicked() {
            if let Ok(mut p) = self.params.write() {
                (p.a, p.b, p.d_u, p.d_v) = (0.1, 0.9, 1.0, 100.0);
            }
        }
        if ui.button("Stripes").clicked() {
            if let Ok(mut p) = self.params.write() {
                (p.a, p.b, p.d_u, p.d_v) = (0.14, 0.86, 1.0, 50.0);
            }
        }

        let mut dialog_state = std::mem::take(&mut self.dialog_state);
        crate::presets::render_preset_buttons(&mut dialog_state, ui, self);
        self.dialog_state = dialog_state;
    }

    fn draw_divergence_recovery(&mut self, ui: &mut egui::Ui) {
        if self.diverged() && self.worker.state() != SimWorkerState::Faulted {
            if let Some(action) = crate::framework::recovery::render_recovery_banner(
                ui,
                self.diagnostic_report(),
            ) {
                match action {
                    crate::framework::recovery::RecoveryAction::AutoScaleStepSize => {
                        self.auto_scale_step_size();
                    }
                    crate::framework::recovery::RecoveryAction::ClampParameters => {
                        self.clamp_parameters();
                    }
                    crate::framework::recovery::RecoveryAction::ResetParameters => {
                        self.reset_parameters();
                    }
                }
            }
            ui.separator();
        }
    }

    fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("morphogenesis_controls").show(ctx, |ui| {
            ui.heading("Morphogenesis Controls");
            if ui.button("🎨 Starter Gallery").clicked() {
                self.show_starter_gallery = !self.show_starter_gallery;
            }
            ui.separator();

            if self.worker.state() == SimWorkerState::Faulted {
                ui.group(|ui| {
                    ui.colored_label(egui::Color32::RED, "⚠ Worker Faulted: Simulation thread disconnected.");
                    if ui.button("🔄 Retry").accessible_hover_text("Restart background worker").clicked() {
                        self.retry();
                    }
                });
                ui.separator();
            }

            self.draw_divergence_recovery(ui);

            self.draw_parameter_sliders(ui);
            ui.separator();

            ui.horizontal(|ui| {
                let is_running = self.worker.state().is_running();
                let can_toggle = matches!(
                    self.worker.state(),
                    SimWorkerState::Ready | SimWorkerState::Running
                );
                if ui
                    .add_enabled(
                        can_toggle,
                        egui::Button::new(if is_running { "⏸ Pause" } else { "▶ Resume" }),
                    )
                    .clicked()
                {
                    if is_running {
                        let _ = self.worker.send_command(SimCommand::Pause);
                    } else {
                        let _ = self.worker.send_command(SimCommand::Start);
                    }
                }

                if ui.button("↻ Reset").clicked() {
                    if self.worker.state() == SimWorkerState::Faulted
                        || self.worker.state() == SimWorkerState::Terminated
                    {
                        self.retry();
                    } else {
                        let _ = self.worker.send_command(SimCommand::Reset);
                    }
                    self.dirty = true;
                }
            });

            self.draw_presets(ui);
        });
    }

    fn draw_canvas(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.dirty || self.texture.is_none() {
                if let Some(snapshot) = self.worker.latest_snapshot() {
                    if let Ok(snapshot_pixels) = snapshot.pixels.try_read() {
                        if !snapshot_pixels.is_empty() {
                            let num_pixels = snapshot.width * snapshot.height;
                            if self.pixels.len() != num_pixels {
                                self.pixels.resize(num_pixels, egui::Color32::BLACK);
                            }

                            if snapshot_pixels.len() == num_pixels {
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
                                self.texture = Some(ctx.load_texture(
                                    "morphogenesis_tex",
                                    image,
                                    texture_opts,
                                ));
                            }
                            self.dirty = false;
                        }
                    }
                }
            }

            if let Some(texture) = &self.texture {
                let w = self.width as f32;
                let h = self.height as f32;
                Plot::new("morphogenesis_plot")
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .show_axes([false, false])
                    .allow_drag(true)
                    .allow_zoom(true)
                    .show(ui, |plot_ui| {
                        plot_ui.image(PlotImage::new(
                            "morphogenesis_image",
                            texture.id(),
                            PlotPoint::new(w as f64 / 2.0, h as f64 / 2.0),
                            [w, h],
                        ));
                    });
            }
        });
    }

    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);

        let a = system.kinetics.a;
        let b = system.kinetics.b;
        let u_eq = a + b;
        let v_eq = b / (u_eq * u_eq);

        for i in 0..n {
            system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
            system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
        }
    }

    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        let num_pixels = width * height;
        let mut pixels = Vec::with_capacity(num_pixels);

        let (min, max) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                if val.is_finite() {
                    (min.min(val), max.max(val))
                } else {
                    (min, max)
                }
            });

        let (min, max) = if min.is_finite() && max.is_finite() && max >= min {
            (min, max)
        } else {
            (0.0, 1.0)
        };

        let range = max - min;
        let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };

        for &val in data {
            let safe_val = if val.is_finite() { val } else { min };
            let norm = (safe_val - min) * inv_range;
            let safe_norm = if norm.is_finite() {
                norm.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rgb = heatmap_color(safe_norm);
            pixels.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }

        ColorImage {
            size: [width, height],
            pixels,
            ..Default::default()
        }
    }
}

#[path = "morphogenesis/presets.rs"]
mod presets_impl;
pub use presets_impl::*;

#[cfg(test)]
#[path = "morphogenesis/tests.rs"]
mod tests;
