use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points, Polygon};
use math_explorer::pure_math::statistics::tda::{
    core::{Point2D, PointCloud},
    vietoris_rips_complex, SimplicialComplex,
};
use rand::Rng;

pub struct VietorisRipsTool {
    radius: f64,
    points: Vec<Point2D>,
    complex: SimplicialComplex,
    error_msg: Option<String>,
}

impl Default for VietorisRipsTool {
    fn default() -> Self {
        let mut tool = Self {
            radius: 1.5,
            points: Vec::new(),
            complex: SimplicialComplex::new(),
            error_msg: None,
        };
        tool.generate_points(20);
        tool.update_complex();
        tool
    }
}

impl VietorisRipsTool {
    fn generate_points(&mut self, count: usize) {
        let mut rng = oxidize_core::rng::OxidizeRng::default();

        self.points.clear();
        for _ in 0..count {
            let x = rng.gen_range(0.0..10.0);
            let y = rng.gen_range(0.0..10.0);
            self.points.push(Point2D::new(x, y));
        }
        self.update_complex();
    }

    fn update_complex(&mut self) {
        match PointCloud::new(self.points.clone()) {
            Ok(cloud) => match vietoris_rips_complex(&cloud, self.radius) {
                Ok(complex) => {
                    self.complex = complex;
                    self.error_msg = None;
                }
                Err(e) => {
                    self.error_msg = Some(format!("Error building complex: {:?}", e));
                }
            },
            Err(e) => {
                self.error_msg = Some(format!("Error building point cloud: {:?}", e));
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VietorisRipsState {
    radius: f64,
    points: Vec<[f64; 2]>,
}

impl InteractiveTool for VietorisRipsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Vietoris-Rips Complex"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn save_state(&self) -> Option<String> {
        let pts = self.points.iter().map(|p| [p.x, p.y]).collect();
        serde_json::to_string(&VietorisRipsState {
            radius: self.radius,
            points: pts,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(loaded) = serde_json::from_str::<VietorisRipsState>(state) {
            self.radius = loaded.radius;
            self.points = loaded
                .points
                .into_iter()
                .map(|[x, y]| Point2D::new(x, y))
                .collect();
            self.update_complex();
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Vietoris-Rips Complex (TDA)");

            ui.horizontal(|ui| {
                if ui.button("Generate Random Points").clicked() {
                    // allow(entropy_guard)
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let mut rng = oxidize_core::rng::OxidizeRng::new(now);

                    self.points.clear();
                    for _ in 0..20 {
                        let x = rng.gen_range(0.0..10.0);
                        let y = rng.gen_range(0.0..10.0);
                        self.points.push(Point2D::new(x, y));
                    }
                    self.update_complex();
                }

                let prev_radius = self.radius;
                ui.add(egui::Slider::new(&mut self.radius, 0.0..=15.0).text("").text("Radius (ε):"));
                if (self.radius - prev_radius).abs() > f64::EPSILON {
                    self.update_complex();
                }
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label(format!(
                    "Vertices (0-simplices): {}",
                    self.complex.count_simplices(0)
                ));
                ui.label(format!(
                    "Edges (1-simplices): {}",
                    self.complex.count_simplices(1)
                ));
                ui.label(format!(
                    "Triangles (2-simplices): {}",
                    self.complex.count_simplices(2)
                ));
            });

            if let Some(err) = &self.error_msg {
                ui.colored_label(egui::Color32::RED, err);
            }

            // Plotting
            let plot = Plot::new("vietoris_rips_plot")
                .view_aspect(1.0)
                .data_aspect(1.0)
                .allow_drag(true)
                .allow_zoom(true);

            plot.show(ui, |plot_ui| {
                // Draw 2-simplices (triangles)
                let triangles = self.complex.get_simplices(2);
                for t in triangles {
                    let p0 = &self.points[t.vertices[0]];
                    let p1 = &self.points[t.vertices[1]];
                    let p2 = &self.points[t.vertices[2]];

                    let points = vec![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y]];

                    plot_ui.polygon(
                        Polygon::new("Triangle", PlotPoints::new(points))
                            .fill_color(egui::Color32::from_rgba_unmultiplied(100, 150, 250, 50))
                            .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
                    );
                }

                // Draw 1-simplices (edges)
                let edges = self.complex.get_simplices(1);
                for e in edges {
                    let p0 = &self.points[e.vertices[0]];
                    let p1 = &self.points[e.vertices[1]];

                    let points = vec![[p0.x, p0.y], [p1.x, p1.y]];

                    plot_ui.line(
                        Line::new("Edge", PlotPoints::new(points))
                            .color(egui::Color32::from_rgb(150, 150, 150))
                            .width(1.5_f32),
                    );
                }

                // Draw 0-simplices (vertices)
                let mut vertex_coords = Vec::new();
                for p in &self.points {
                    vertex_coords.push([p.x, p.y]);
                }

                plot_ui.points(
                    Points::new("Vertices", PlotPoints::new(vertex_coords))
                        .color(egui::Color32::from_rgb(50, 50, 200))
                        .radius(4.0_f32),
                );
            });
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "VietorisRipsTool",
        domain: "geometry_topology",
        tags: &[],
        build: || Box::new(VietorisRipsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for VietorisRipsTool {
    fn theory_description(&self) -> String {
        format!(
            "Topological Data Analysis (TDA) via Vietoris-Rips simplicial filtration: a $k$-simplex $\\sigma = [v_0, \\dots, v_k]$ is included if pairwise distance $d(v_i, v_j) \\le 2\\epsilon$ with filtration parameter $\\epsilon = {:.2}$ across {} vertices.",
            self.radius, self.points.len()
        )
    }
    fn phonetic_description(&self) -> String { "Vietoris-Rips simplicial complex and topological data analysis theory".into() }
    fn theory_citation(&self) -> String { "Computational Topology: An Introduction (Edelsbrunner & Harer, 2010)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
