use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, Points, Polygon};
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
    cached_triangles: Vec<[PlotPoint; 3]>,
    cached_edges: Vec<[PlotPoint; 2]>,
    cached_vertices: Vec<PlotPoint>,
}

impl Default for VietorisRipsTool {
    fn default() -> Self {
        let mut tool = Self {
            radius: 1.5,
            points: Vec::new(),
            complex: SimplicialComplex::new(),
            error_msg: None,
            cached_triangles: Vec::new(),
            cached_edges: Vec::new(),
            cached_vertices: Vec::new(),
        };
        tool.generate_points(20);
        tool.update_complex();
        tool
    }
}

impl VietorisRipsTool {
    fn clear_cache(&mut self) {
        self.cached_triangles.clear();
        self.cached_edges.clear();
        self.cached_vertices.clear();
    }

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
                    let triangles = complex.get_simplices(2);
                    self.cached_triangles = triangles
                        .into_iter()
                        .map(|t| {
                            let p0 = &self.points[t.vertices[0]];
                            let p1 = &self.points[t.vertices[1]];
                            let p2 = &self.points[t.vertices[2]];
                            [
                                PlotPoint::new(p0.x, p0.y),
                                PlotPoint::new(p1.x, p1.y),
                                PlotPoint::new(p2.x, p2.y),
                            ]
                        })
                        .collect();

                    let edges = complex.get_simplices(1);
                    self.cached_edges = edges
                        .into_iter()
                        .map(|e| {
                            let p0 = &self.points[e.vertices[0]];
                            let p1 = &self.points[e.vertices[1]];
                            [
                                PlotPoint::new(p0.x, p0.y),
                                PlotPoint::new(p1.x, p1.y),
                            ]
                        })
                        .collect();

                    self.cached_vertices = self
                        .points
                        .iter()
                        .map(|p| PlotPoint::new(p.x, p.y))
                        .collect();

                    self.complex = complex;
                    self.error_msg = None;
                }
                Err(e) => {
                    self.clear_cache();
                    self.error_msg = Some(format!("Error building complex: {e}"));
                }
            },
            Err(e) => {
                self.clear_cache();
                self.error_msg = Some(format!("Error building point cloud: {e}"));
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
                for t in &self.cached_triangles {
                    plot_ui.polygon(
                        Polygon::new("Triangle", &t[..])
                            .fill_color(egui::Color32::from_rgba_unmultiplied(100, 150, 250, 50))
                            .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
                    );
                }

                // Draw 1-simplices (edges)
                for e in &self.cached_edges {
                    plot_ui.line(
                        Line::new("Edge", &e[..])
                            .color(egui::Color32::from_rgb(150, 150, 150))
                            .width(1.5_f32),
                    );
                }

                // Draw 0-simplices (vertices)
                plot_ui.points(
                    Points::new("Vertices", &self.cached_vertices[..])
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
        id: "VietorisRipsTool",
        display_name: "Vietoris-Rips Complex",
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

#[cfg(test)]
mod tests {
    use super::*;
    use egui_plot::PlotPoints;

    #[test]
    fn test_vietoris_rips_default_initialization() {
        let tool = VietorisRipsTool::default();
        assert_eq!(tool.cached_vertices.len(), tool.points.len());
        assert_eq!(tool.cached_vertices.len(), 20);
        assert_eq!(tool.cached_edges.len(), tool.complex.count_simplices(1));
        assert_eq!(tool.cached_triangles.len(), tool.complex.count_simplices(2));
        assert!(tool.error_msg.is_none());
    }

    #[test]
    fn test_vietoris_rips_parameter_updates() {
        let mut tool = VietorisRipsTool {
            radius: 15.0,
            ..Default::default()
        };

        // Expanding radius connects all points in [0, 10] x [0, 10]
        tool.update_complex();
        assert_eq!(tool.cached_edges.len(), tool.complex.count_simplices(1));
        assert_eq!(tool.cached_triangles.len(), tool.complex.count_simplices(2));
        assert_eq!(tool.cached_vertices.len(), tool.points.len());
        assert!(!tool.cached_edges.is_empty());
        assert!(!tool.cached_triangles.is_empty());

        // Shrinking radius to 0.0 results in 0 edges and 0 triangles
        tool.radius = 0.0;
        tool.update_complex();
        assert_eq!(tool.cached_edges.len(), 0);
        assert_eq!(tool.cached_triangles.len(), 0);
        assert_eq!(tool.cached_vertices.len(), tool.points.len());
    }

    #[test]
    fn test_vietoris_rips_error_state_cache_clearing() {
        let mut tool = VietorisRipsTool::default();
        assert!(!tool.cached_vertices.is_empty());

        // Invalid negative radius causes vietoris_rips_complex error and clears cache
        tool.radius = -1.0;
        tool.update_complex();
        assert!(tool.error_msg.is_some());
        assert!(tool.cached_triangles.is_empty());
        assert!(tool.cached_edges.is_empty());
        assert!(tool.cached_vertices.is_empty());

        // Empty points causes PointCloud error and clears cache
        let mut tool2 = VietorisRipsTool::default();
        tool2.points.clear();
        tool2.update_complex();
        assert!(tool2.error_msg.is_some());
        assert!(tool2.cached_triangles.is_empty());
        assert!(tool2.cached_edges.is_empty());
        assert!(tool2.cached_vertices.is_empty());
    }

    #[test]
    fn test_vietoris_rips_borrowed_plot_points_slice_variants() {
        let mut tool = VietorisRipsTool {
            radius: 15.0,
            ..Default::default()
        };
        tool.update_complex();

        let v_slice: &[PlotPoint] = &tool.cached_vertices[..];
        let v_plot: PlotPoints<'_> = v_slice.into();
        assert!(matches!(v_plot, PlotPoints::Borrowed(_)));
        assert_eq!(v_plot.points().len(), tool.cached_vertices.len());

        let edge = tool.cached_edges.first().expect("edges should not be empty with radius 15.0");
        let e_slice: &[PlotPoint] = &edge[..];
        let e_plot: PlotPoints<'_> = e_slice.into();
        assert!(matches!(e_plot, PlotPoints::Borrowed(_)));
        assert_eq!(e_plot.points().len(), 2);

        let triangle = tool.cached_triangles.first().expect("triangles should not be empty with radius 15.0");
        let t_slice: &[PlotPoint] = &triangle[..];
        let t_plot: PlotPoints<'_> = t_slice.into();
        assert!(matches!(t_plot, PlotPoints::Borrowed(_)));
        assert_eq!(t_plot.points().len(), 3);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Persistent Homology",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "max_distance": 2.0,
    "num_points": 50
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_starter",
        title: "Persistent Homology",
        description: "Simplicial complex filtration across distance radius ε capturing topological persistence barcodes.",
        category: "Algebraic Topology",
        icon: "🕸️",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
