use crate::async_sim::{SimCommand, SimulationController, SimulationRunner, StateSnapshot};
use crate::presets::{InteractivePreset, MorphogenesisPresetConfig, PresetDialogState};
use crate::tabs::ExplorerTab;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::heatmap_color;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use std::sync::Arc;

#[allow(missing_docs)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
}

#[allow(missing_docs)]
pub struct MorphogenesisRunner {
    pub(crate) system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl SimulationRunner for MorphogenesisRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                MorphogenesisTab::initialize_system(&mut self.system, self.width, self.height);
            }
            SimCommand::Custom(ref s) if s == "reinit" => {
                MorphogenesisTab::initialize_system(&mut self.system, self.width, self.height);
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap();
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        self.system.kinetics.a = a;
        self.system.kinetics.b = b;
        self.system.diffusion_coeffs[0] = d_u;
        self.system.diffusion_coeffs[1] = d_v;
        self.system.step(dt);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let image =
            MorphogenesisTab::plot_concentration(self.system.u(), self.width, self.height);
        let pixels: Vec<egui::Color32> = image.pixels;
        StateSnapshot {
            width: self.width,
            height: self.height,
            pixels: Arc::new(std::sync::RwLock::new(pixels)),
            custom_data: Vec::new(),
            structured_data: None,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct MorphogenesisTab {
    controller: SimulationController,
    params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    width: usize,
    height: usize,
    paused: bool,
    texture: Option<egui::TextureHandle>,
    dialog_state: PresetDialogState,
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let a = 0.1;
        let b = 0.9;
        let d_u = 1.0;
        let d_v = 100.0;
        let dt = 0.05;

        let kinetics = SchnakenbergKinetics { a, b };
        let diffusion = FiniteDifference2D::new(
            math_explorer::math_kernel::types::Dimension(width),
            math_explorer::math_kernel::types::Dimension(height),
            math_explorer::math_kernel::types::StepSize(1.0),
            math_explorer::math_kernel::types::StepSize(1.0),
        );

        let mut system = TuringSystem::new_with_kinetics(
            math_explorer::math_kernel::types::Dimension(width * height),
            math_explorer::biology::morphogenesis::DiffusionCoeff(d_u),
            math_explorer::biology::morphogenesis::DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );

        Self::initialize_system(&mut system, width, height);

        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
            a,
            b,
            d_u,
            d_v,
            dt,
        }));

        let runner = MorphogenesisRunner {
            system,
            params: Arc::clone(&params),
            width,
            height,
        };

        let controller = SimulationController::new(runner);
        controller.send_command(SimCommand::Start);

        Self {
            controller,
            params,
            width,
            height,
            paused: false,
            texture: None,
            dialog_state: PresetDialogState::default(),
        }
    }
}

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct MorphoState {
            dt: f64,
            paused: bool,
        }

        let dt = self.params.read().unwrap().dt;
        serde_json::to_string(&MorphoState {
            dt,
            paused: self.paused,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct MorphoState {
            dt: f64,
            paused: bool,
        }

        if let Ok(s) = serde_json::from_str::<MorphoState>(state) {
            self.params.write().unwrap().dt = s.dt;
            self.paused = s.paused;
            if self.paused {
                self.controller.send_command(SimCommand::Pause);
            } else {
                self.controller.send_command(SimCommand::Start);
            }
        }
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.controller.update();

        if !self.paused {
            ctx.request_repaint();
        }

        self.draw_controls(ctx);
        self.draw_canvas(ctx);
    }
}

impl MorphogenesisTab {
    fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("morphogenesis_controls").show(ctx, |ui| {
            ui.heading("Morphogenesis Controls");
            ui.separator();

            let mut params = self.params.write().unwrap();
            ui.add(egui::Slider::new(&mut params.a, 0.0..=1.0).text("a"));
            ui.add(egui::Slider::new(&mut params.b, 0.0..=2.0).text("b"));
            ui.add(egui::Slider::new(&mut params.d_u, 0.1..=5.0).text("D_u"));
            ui.add(egui::Slider::new(&mut params.d_v, 10.0..=200.0).text("D_v"));

            ui.separator();

            ui.horizontal(|ui| {
                if ui
                    .button(if self.paused { "▶ Resume" } else { "⏸ Pause" })
                    .clicked()
                {
                    self.paused = !self.paused;
                    if self.paused {
                        self.controller.send_command(SimCommand::Pause);
                    } else {
                        self.controller.send_command(SimCommand::Start);
                    }
                }

                if ui.button("↻ Reset").clicked() {
                    self.controller.send_command(SimCommand::Reset);
                }
            });

            ui.separator();
            ui.label("Presets:");
            if ui.button("Spots").clicked() {
                params.a = 0.1;
                params.b = 0.9;
                params.d_u = 1.0;
                params.d_v = 100.0;
            }
            if ui.button("Stripes").clicked() {
                params.a = 0.14;
                params.b = 0.86;
                params.d_u = 1.0;
                params.d_v = 50.0;
            }
            drop(params);

            let mut dialog_state = std::mem::take(&mut self.dialog_state);
            crate::presets::render_preset_buttons(&mut dialog_state, ui, self);
            self.dialog_state = dialog_state;
        });
    }

    fn draw_canvas(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(snapshot) = self.controller.latest_snapshot() {
                if let Ok(pixels) = snapshot.pixels.try_read() {
                    if !pixels.is_empty() {
                        let image = ColorImage::new(
                            [snapshot.width, snapshot.height],
                            pixels.clone(),
                        );
                        if let Some(ref mut texture) = self.texture {
                            texture.set(image, egui::TextureOptions::NEAREST);
                        } else {
                            self.texture = Some(ctx.load_texture(
                                "morphogenesis_tex",
                                image,
                                egui::TextureOptions::NEAREST,
                            ));
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

    fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        let mut pixels = Vec::with_capacity(width * height * 4);

        let (min, max) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let range = max - min;
        let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };

        for &val in data {
            let norm = (val - min) * inv_range;
            let rgb = heatmap_color(norm);
            pixels.push(rgb[0]);
            pixels.push(rgb[1]);
            pixels.push(rgb[2]);
            pixels.push(255);
        }

        ColorImage::from_rgba_unmultiplied([width, height], &pixels)
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state;
        let x = x ^ x >> 18;
        let rot = (x >> 27) as u32;
        let val = (x as u32).rotate_right(rot);
        (val as f64) / (u32::MAX as f64)
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

impl InteractivePreset for MorphogenesisTab {
    type Config = MorphogenesisPresetConfig;

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn title(&self) -> String {
        "Turing Morphogenesis Simulation".to_string()
    }

    fn export_config(&self) -> Self::Config {
        let p = self.params.read().unwrap();
        MorphogenesisPresetConfig {
            a: p.a,
            b: p.b,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            width: self.width,
            height: self.height,
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        {
            let mut p = self.params.write().unwrap();
            p.a = config.a;
            p.b = config.b;
            p.d_u = config.d_u;
            p.d_v = config.d_v;
            p.dt = config.dt;
        }
        if config.width != self.width || config.height != self.height {
            self.width = config.width;
            self.height = config.height;
            let kinetics = SchnakenbergKinetics {
                a: config.a,
                b: config.b,
            };
            let diffusion = FiniteDifference2D::new(
                math_explorer::math_kernel::types::Dimension(self.width),
                math_explorer::math_kernel::types::Dimension(self.height),
                math_explorer::math_kernel::types::StepSize(1.0),
                math_explorer::math_kernel::types::StepSize(1.0),
            );
            let mut system = TuringSystem::new_with_kinetics(
                math_explorer::math_kernel::types::Dimension(self.width * self.height),
                math_explorer::biology::morphogenesis::DiffusionCoeff(config.d_u),
                math_explorer::biology::morphogenesis::DiffusionCoeff(config.d_v),
                kinetics,
                diffusion,
            );
            Self::initialize_system(&mut system, self.width, self.height);
            let runner = MorphogenesisRunner {
                system,
                params: Arc::clone(&self.params),
                width: self.width,
                height: self.height,
            };
            self.controller = SimulationController::new(runner);
            if !self.paused {
                self.controller.send_command(SimCommand::Start);
            }
        } else {
            self.controller.send_command(SimCommand::Reset);
        }
        Ok(())
    }
}
