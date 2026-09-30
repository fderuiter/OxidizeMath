use crate::tabs::ExplorerTab;
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::heatmap_color;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};

pub struct MorphogenesisTab {
    system: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
    dt: f64,
    paused: bool,
    texture: Option<egui::TextureHandle>,
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let a = 0.1;
        let b = 0.9;
        let d_u = 1.0;
        let d_v = 100.0;
        
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

        Self {
            system,
            width,
            height,
            dt: 0.05,
            paused: false,
            texture: None,
        }
    }
}

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.paused {
            self.system.step(self.dt);
            ctx.request_repaint();
        }

        egui::SidePanel::left("morphogenesis_controls").show(ctx, |ui| {
            ui.heading("Morphogenesis Controls");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.system.kinetics.a, 0.0..=1.0).text("a"));
            ui.add(egui::Slider::new(&mut self.system.kinetics.b, 0.0..=2.0).text("b"));
            ui.add(egui::Slider::new(&mut self.system.diffusion_coeffs[0], 0.1..=5.0).text("D_u"));
            ui.add(egui::Slider::new(&mut self.system.diffusion_coeffs[1], 10.0..=200.0).text("D_v"));

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                }
                
                if ui.button("↻ Reset").clicked() {
                    Self::initialize_system(&mut self.system, self.width, self.height);
                }
            });
            
            ui.separator();
            ui.label("Presets:");
            if ui.button("Spots").clicked() {
                self.system.kinetics.a = 0.1;
                self.system.kinetics.b = 0.9;
                self.system.diffusion_coeffs[0] = 1.0;
                self.system.diffusion_coeffs[1] = 100.0;
            }
            if ui.button("Stripes").clicked() {
                self.system.kinetics.a = 0.14;
                self.system.kinetics.b = 0.86;
                self.system.diffusion_coeffs[0] = 1.0;
                self.system.diffusion_coeffs[1] = 50.0;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let image = Self::plot_concentration(self.system.u(), self.width, self.height);
            if let Some(ref mut texture) = self.texture {
                texture.set(image, egui::TextureOptions::NEAREST);
            } else {
                self.texture = Some(ctx.load_texture(
                    "morphogenesis_tex",
                    image,
                    egui::TextureOptions::NEAREST,
                ));
            }

            if let Some(texture) = &self.texture {
                ui.add(egui::Image::new(texture).fit_to_exact_size(ui.available_size()));
            }
        });
    }
}

impl MorphogenesisTab {
    fn initialize_system(
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
// [cite:graph_parameters_rust]
