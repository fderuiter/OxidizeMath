use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::physics::solid_state::band_theory::{free_electron_1d, tight_binding_1d};
use std::f64::consts::PI;

pub struct BandStructureTool {
    e0: f64,
    t: f64,
    a: f64,
    hbar: f64,
    m: f64,
    show_free_electron: bool,
}

impl Default for BandStructureTool {
    fn default() -> Self {
        Self {
            e0: 0.0,
            t: 1.0,
            a: 1.0,
            hbar: 1.0,
            m: 1.0,
            show_free_electron: false,
        }
    }
}

impl InteractiveTool for BandStructureTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Band Structure"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BandStructureState {
            e0: f64,
            t: f64,
            a: f64,
            hbar: f64,
            m: f64,
            show_free_electron: bool,
        }
        serde_json::to_string(&BandStructureState {
            e0: self.e0,
            t: self.t,
            a: self.a,
            hbar: self.hbar,
            m: self.m,
            show_free_electron: self.show_free_electron,
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BandStructureState {
            e0: f64,
            t: f64,
            a: f64,
            hbar: f64,
            m: f64,
            show_free_electron: bool,
        }
        if let Ok(s) = serde_json::from_str::<BandStructureState>(state) {
            self.e0 = s.e0;
            self.t = s.t;
            self.a = s.a;
            self.hbar = s.hbar;
            self.m = s.m;
            self.show_free_electron = s.show_free_electron;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("band_structure_controls")
            .resizable(false)
            .min_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Parameters");
                ui.separator();

                ui.label("Tight Binding Model");
                ui.add(
                    egui::Slider::new(&mut self.e0, -5.0..=5.0)
                        .text("E₀ (On-site Energy)")
                        .step_by(0.1),
                );
                ui.add(
                    egui::Slider::new(&mut self.t, 0.0..=5.0)
                        .text("t (Hopping Parameter)")
                        .step_by(0.1),
                );
                ui.add(
                    egui::Slider::new(&mut self.a, 0.1..=5.0)
                        .text("a (Lattice Constant)")
                        .step_by(0.1),
                );

                ui.separator();
                ui.checkbox(&mut self.show_free_electron, "Show Free Electron Model");
                if self.show_free_electron {
                    ui.add(
                        egui::Slider::new(&mut self.hbar, 0.1..=5.0)
                            .text("ħ (Reduced Planck)")
                            .step_by(0.1),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.m, 0.1..=5.0)
                            .text("m (Effective Mass)")
                            .step_by(0.1),
                    );
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("E-k Diagram");

            // Generate points for the Brillouin Zone
            let k_min = -PI / self.a;
            let k_max = PI / self.a;
            let num_points = 200;
            let k_step = (k_max - k_min) / (num_points as f64 - 1.0);

            let tb_points: PlotPoints = (0..num_points)
                .map(|i| {
                    let k = k_min + (i as f64) * k_step;
                    let e = tight_binding_1d(k, self.e0, self.t, self.a);
                    [k, e]
                })
                .collect();

            let plot = Plot::new("band_structure_plot")
                .x_axis_label("Wavevector (k)")
                .y_axis_label("Energy (E)")
                .legend(egui_plot::Legend::default());

            plot.show(ui, |plot_ui| {
                plot_ui.line(Line::new("Tight Binding", tb_points).name("Tight Binding"));

                if self.show_free_electron {
                    let fe_points: PlotPoints = (0..num_points)
                        .map(|i| {
                            let k = k_min + (i as f64) * k_step;
                            let e = free_electron_1d(k, self.hbar, self.m);
                            [k, e]
                        })
                        .collect();
                    plot_ui.line(Line::new("Free Electron", fe_points).name("Free Electron"));
                }
            });
        });
    }
}

// [cite:mmwave_radiotherapy_setup]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "BandStructureTool",
        domain: "solid_state",
        tags: &[],
        build: || Box::new(BandStructureTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BandStructureTool {
    fn theory_description(&self) -> String {
        let model_str = if self.show_free_electron {
            format!("parabolic free-electron dispersion $E(k) = \\frac{{\\hbar^2 k^2}}{{2m}}$ with effective mass $m = {:.2}$", self.m)
        } else {
            format!("1D tight-binding nearest-neighbor dispersion $E(k) = E_0 - 2t \\cos(ka)$ with $E_0 = {:.2}$, hopping $t = {:.2}$, lattice constant $a = {:.2}$", self.e0, self.t, self.a)
        };
        format!(
            "Solid state electronic band structure dispersion relation in the first Brillouin zone $[-\\pi/a, \\pi/a]$: {}.",
            model_str
        )
    }
    fn phonetic_description(&self) -> String { "Electronic band structure and tight-binding dispersion theory".into() }
    fn theory_citation(&self) -> String { "Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("TightBinding".into(), "E(k) = E_0 - 2t cos(ka) atomic orbital overlap model".into());
        map.insert("FreeElectron".into(), "E(k) = hbar^2 k^2 / (2m) parabolic free Fermi gas dispersion".into());
        map
    }
}
