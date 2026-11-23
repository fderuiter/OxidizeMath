use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, Points, Polygon};
use math_explorer::pure_math::statistics::tda::{
    core::{Point2D, PointCloud},
    vietoris_rips_complex, SimplicialComplex,
};
use rand::Rng;

const MAX_VIETORIS_RIPS_POINTS: usize = 20;

pub struct VietorisRipsTool {
    radius: f64,
    points: Vec<Point2D>,
    complex: SimplicialComplex,
    error_msg: Option<String>,
    cached_triangles: Vec<[PlotPoint; 3]>,
    cached_edges: Vec<[PlotPoint; 2]>,
    cached_vertices: Vec<PlotPoint>,
    revision: u64,
    #[cfg(test)]
    complex_builds: usize,
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
            revision: 1,
            #[cfg(test)]
            complex_builds: 0,
        };
        tool.generate_points(MAX_VIETORIS_RIPS_POINTS);
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

    fn compute_complex(
        &mut self,
        cloud: &PointCloud,
        radius: f64,
    ) -> Result<SimplicialComplex, String> {
        #[cfg(test)]
        {
            self.complex_builds += 1;
        }
        vietoris_rips_complex(cloud, radius).map_err(|e| e.to_string())
    }

    fn apply_complex(&mut self, complex: SimplicialComplex) {
        self.cached_triangles = complex
            .get_simplices(2)
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
        self.cached_edges = complex
            .get_simplices(1)
            .into_iter()
            .map(|e| {
                let p0 = &self.points[e.vertices[0]];
                let p1 = &self.points[e.vertices[1]];
                [PlotPoint::new(p0.x, p0.y), PlotPoint::new(p1.x, p1.y)]
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

    fn update_complex(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        match PointCloud::new(self.points.clone()) {
            Ok(cloud) => match self.compute_complex(&cloud, self.radius) {
                Ok(complex) => self.apply_complex(complex),
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

struct BorrowedPoints2D<'a>(&'a [Point2D]);

impl serde::Serialize for BorrowedPoints2D<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for pt in self.0 {
            seq.serialize_element(&[pt.x, pt.y])?;
        }
        seq.end()
    }
}

#[derive(serde::Serialize)]
struct VietorisRipsStateRef<'a> {
    radius: f64,
    points: BorrowedPoints2D<'a>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VietorisRipsState {
    radius: f64,
    points: Vec<[f64; 2]>,
}

impl InteractiveTool for VietorisRipsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Vietoris-Rips Complex"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.radius.to_bits().hash(&mut h);
        self.points.len().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.radius.is_finite()
            && self
                .points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&VietorisRipsStateRef {
            radius: self.radius,
            points: BorrowedPoints2D(&self.points),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: VietorisRipsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Vietoris-Rips state: {e}"))?;
        if !loaded.radius.is_finite() || !(0.0..=15.0).contains(&loaded.radius) {
            return Err(format!("radius out of [0.0, 15.0]: {}", loaded.radius));
        }
        if !(1..=MAX_VIETORIS_RIPS_POINTS).contains(&loaded.points.len())
            || !loaded
                .points
                .iter()
                .all(|p| p[0].is_finite() && p[1].is_finite())
        {
            return Err(format!(
                "points must contain 1..={MAX_VIETORIS_RIPS_POINTS} finite coordinates (got {})",
                loaded.points.len()
            ));
        }
        let pts: Vec<Point2D> = loaded
            .points
            .into_iter()
            .map(|[x, y]| Point2D::new(x, y))
            .collect();
        let cloud = PointCloud::new(pts.clone()).map_err(|e| format!("{e:?}"))?;
        let complex = self
            .compute_complex(&cloud, loaded.radius)
            .map_err(|e| format!("{e:?}"))?;
        self.radius = loaded.radius;
        self.points = pts;
        self.revision = self.revision.wrapping_add(1).max(1);
        self.apply_complex(complex);
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Vietoris-Rips Complex (TDA)");

            ui.horizontal(|ui| {
                if ui.button("Generate Random Points").clicked() {
                    let mut rng = oxidize_core::rng::OxidizeRng::new(self.revision);

                    self.points.clear();
                    for _ in 0..20 {
                        let x = rng.gen_range(0.0..10.0);
                        let y = rng.gen_range(0.0..10.0);
                        self.points.push(Point2D::new(x, y));
                    }
                    self.update_complex();
                }

                let prev_radius = self.radius;
                ui.add(egui::Slider::new(&mut self.radius, 0.0..=15.0).text("Radius (ε):"));
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

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if let Some(err) = &self.error_msg {
                ui.colored_label(palette.error, err);
            }

            // Plotting
            let plot = Plot::new("vietoris_rips_plot")
                .view_aspect(1.0)
                .data_aspect(1.0)
                .allow_drag(true)
                .allow_zoom(true);

            plot.show(ui, |plot_ui| {
                let s0 = palette.series_color(0);
                let tri_fill = egui::Color32::from_rgba_unmultiplied(s0.r(), s0.g(), s0.b(), 55);
                // Draw 2-simplices (triangles)
                for t in &self.cached_triangles {
                    plot_ui.polygon(
                        Polygon::new("Triangle", &t[..])
                            .fill_color(tri_fill)
                            .stroke(egui::Stroke::new(0.0_f32, egui::Color32::TRANSPARENT)),
                    );
                }

                // Draw 1-simplices (edges)
                for e in &self.cached_edges {
                    plot_ui.line(
                        Line::new("Edge", &e[..])
                            .color(palette.graph_edge)
                            .width(1.5_f32),
                    );
                }

                // Draw 0-simplices (vertices)
                plot_ui.points(
                    Points::new("Vertices", &self.cached_vertices[..])
                        .color(s0)
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
    fn phonetic_description(&self) -> String {
        "Vietoris-Rips simplicial complex and topological data analysis theory".into()
    }
    fn theory_citation(&self) -> String {
        "Computational Topology: An Introduction (Edelsbrunner & Harer, 2010)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
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

        let edge = tool
            .cached_edges
            .first()
            .expect("edges should not be empty with radius 15.0");
        let e_slice: &[PlotPoint] = &edge[..];
        let e_plot: PlotPoints<'_> = e_slice.into();
        assert!(matches!(e_plot, PlotPoints::Borrowed(_)));
        assert_eq!(e_plot.points().len(), 2);

        let triangle = tool
            .cached_triangles
            .first()
            .expect("triangles should not be empty with radius 15.0");
        let t_slice: &[PlotPoint] = &triangle[..];
        let t_plot: PlotPoints<'_> = t_slice.into();
        assert!(matches!(t_plot, PlotPoints::Borrowed(_)));
        assert_eq!(t_plot.points().len(), 3);
    }

    #[test]
    fn test_vietoris_rips_try_load_state_bounds_and_single_build() {
        let mut tool = VietorisRipsTool::default();
        let before = tool.complex_builds;

        assert!(tool
            .try_load_state(r#"{"radius":1.5,"points":[]}"#)
            .is_err());
        assert!(tool
            .try_load_state(r#"{"radius":1.5,"points":[[0.0,"NaN"]]}"#)
            .is_err());
        let over_20 = serde_json::json!({
            "radius": 2.0,
            "points": (0..21).map(|i| [i as f64, 0.0]).collect::<Vec<_>>()
        });
        assert!(tool.try_load_state(&over_20.to_string()).is_err());
        assert_eq!(tool.complex_builds, before);

        let valid = r#"{"radius":2.0,"points":[[0.0,0.0],[1.0,0.0],[0.0,1.0]]}"#;
        assert!(tool.try_load_state(valid).is_ok());
        assert_eq!(tool.complex_builds, before + 1);
        assert_eq!(tool.points.len(), 3);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Vietoris-Rips Filtration",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "radius": 2.2,
    "points": [
      [5.0, 8.0], [6.5, 7.6], [7.6, 6.5], [8.0, 5.0],
      [7.6, 3.5], [6.5, 2.4], [5.0, 2.0], [3.5, 2.4],
      [2.4, 3.5], [2.0, 5.0], [2.4, 6.5], [3.5, 7.6]
    ]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_starter",
        title: "Vietoris-Rips Filtration",
        description: "Simplicial complex filtration of vertices, edges, and triangles across distance radius ε.",
        category: "Algebraic Topology",
        icon: "🕸️",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
