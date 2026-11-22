#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    GeodesicArc, HyperbolicModel, HyperbolicTilingBuffer, MAX_TILING_DEPTH,
    POINCARE_DISK_RADIUS_MAX, Su11Matrix, clamp_to_disk, gauss_bonnet_triangle,
    is_hyperbolic_schlafli, project_disk_point,
};
use std::f64::consts::PI;

pub(crate) const DISK_TILING_CAPACITY: usize = 220;
pub(crate) const ARC_SAMPLE_POINTS: usize = 13;

/// Active pointer interaction mode on the hyperbolic disk canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RulerTarget {
    /// Dragging pans the entire hyperbolic plane via an $\mathrm{SU}(1,1)$ Möbius boost.
    #[default]
    PanOnly,
    /// Clicking or dragging repositions geodesic triangle vertex $A$.
    VertexA,
    /// Clicking or dragging repositions geodesic triangle vertex $B$.
    VertexB,
    /// Clicking or dragging repositions geodesic triangle vertex $C$.
    VertexC,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HyperbolicDiskState {
    model: HyperbolicModel,
    p: usize,
    q: usize,
    depth: usize,
    ruler_re: [f64; 3],
    ruler_im: [f64; 3],
    view_coeffs: [f64; 4],
    show_ruler: bool,
}

/// Interactive Poincaré and Beltrami-Klein disk explorer with Fuchsian $\{p, q\}$ tessellations,
/// Möbius drag-panning, and Gauss-Bonnet geodesic triangle rulers.
pub struct HyperbolicDiskTool {
    pub model: HyperbolicModel,
    pub p: usize,
    pub q: usize,
    pub depth: usize,
    pub tiling: HyperbolicTilingBuffer,
    pub ruler_vertices: [Complex64; 3],
    pub active_ruler_target: RulerTarget,
    pub show_ruler: bool,
    pub status_note: Option<String>,
    pub revision: u64,
    pub last_canvas_id: Option<egui::Id>,
    pub(crate) arc_scratch: [Complex64; ARC_SAMPLE_POINTS],
}

impl Default for HyperbolicDiskTool {
    fn default() -> Self {
        let mut tiling = HyperbolicTilingBuffer::with_capacity(DISK_TILING_CAPACITY);
        let _ = tiling.configure(7, 3, 3);
        let (center, v0, v1) = {
            let central = &tiling.polygons()[0];
            (central.center, central.vertices[0], central.vertices[1])
        };
        Self {
            model: HyperbolicModel::Poincare,
            p: 7,
            q: 3,
            depth: 3,
            tiling,
            ruler_vertices: [center, v0, v1],
            active_ruler_target: RulerTarget::PanOnly,
            show_ruler: true,
            status_note: None,
            revision: 1,
            last_canvas_id: None,
            arc_scratch: [Complex64::new(0.0, 0.0); ARC_SAMPLE_POINTS],
        }
    }
}

impl HyperbolicDiskTool {
    /// Applies an $\mathrm{SU}(1,1)$ Möbius pan `delta` to both the Fuchsian tiling and ruler vertices.
    pub fn apply_mobius_pan(&mut self, delta: Complex64) {
        if !delta.re.is_finite() || !delta.im.is_finite() || delta.norm_sqr() <= 1e-18 {
            return;
        }
        let step = self.tiling.apply_pan(delta);
        for vertex in &mut self.ruler_vertices {
            *vertex = step.apply(*vertex);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    /// Resets camera panning and snaps the Gauss-Bonnet triangle onto a central $\{p, q\}$ sector.
    pub fn reset_pan_and_ruler(&mut self) {
        self.tiling.reset_view();
        if let Some(central) = self.tiling.polygons().first() {
            self.ruler_vertices = [central.center, central.vertices[0], central.vertices[1]];
        }
        self.revision = self.revision.wrapping_add(1);
    }

    /// Rebuilds the $\{p, q\}$ tiling when `(p, q)` satisfies $(p - 2)(q - 2) > 4$.
    pub fn apply_schlafli(&mut self, new_p: usize, new_q: usize, new_depth: usize) -> bool {
        let bounded_depth = new_depth.clamp(1, MAX_TILING_DEPTH);
        if !is_hyperbolic_schlafli(new_p, new_q) {
            self.status_note = Some(format!(
                "{{{new_p}, {new_q}}} is spherical/Euclidean; hyperbolic tilings require (p - 2)(q - 2) > 4."
            ));
            return false;
        }
        if self.tiling.configure(new_p, new_q, bounded_depth).is_ok() {
            self.p = new_p;
            self.q = new_q;
            self.depth = bounded_depth;
            self.status_note = None;
            if let Some(central) = self.tiling.polygons().first() {
                let view_center = central.center;
                if view_center.norm() < 0.25 {
                    self.ruler_vertices =
                        [central.center, central.vertices[0], central.vertices[1]];
                }
            }
            self.revision = self.revision.wrapping_add(1);
            return true;
        }
        false
    }

    fn export_unit_circle_series() -> (String, Vec<[f64; 2]>) {
        let count = 96_usize;
        let mut pts = Vec::with_capacity(count + 1);
        for idx in 0..=count {
            let ang = 2.0 * PI * (idx as f64) / (count as f64);
            pts.push([ang.cos(), ang.sin()]);
        }
        ("Ideal Boundary |z| = 1".to_string(), pts)
    }

    fn export_tiling_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::with_capacity(2 + self.tiling.polygons().len().min(64));
        out.push(Self::export_unit_circle_series());
        let mut buf = [Complex64::new(0.0, 0.0); 9];
        for (poly_idx, poly) in self.tiling.polygons().iter().take(48).enumerate() {
            let mut ring = Vec::with_capacity(poly.sides * 8 + 1);
            for side in 0..poly.sides {
                let a = poly.vertices[side];
                let b = poly.vertices[(side + 1) % poly.sides];
                let arc = GeodesicArc::through_points(a, b);
                arc.sample_into(self.model, &mut buf);
                let limit = if side + 1 == poly.sides { 9 } else { 8 };
                for &pt in &buf[..limit] {
                    ring.push([pt.re, pt.im]);
                }
            }
            out.push((
                format!("Tile {poly_idx} (layer {})", poly.layer),
                ring,
            ));
        }
        if self.show_ruler {
            out.push(self.export_ruler_series());
        }
        out
    }

    fn export_ruler_series(&self) -> (String, Vec<[f64; 2]>) {
        let mut pts = Vec::with_capacity(37);
        let mut buf = [Complex64::new(0.0, 0.0); ARC_SAMPLE_POINTS];
        for side in 0..3 {
            let a = self.ruler_vertices[side];
            let b = self.ruler_vertices[(side + 1) % 3];
            let arc = GeodesicArc::through_points(a, b);
            arc.sample_into(self.model, &mut buf);
            for &pt in &buf {
                pts.push([pt.re, pt.im]);
            }
        }
        if let Some(&first) = self.ruler_vertices.first() {
            let proj = project_disk_point(first, self.model);
            pts.push([proj.re, proj.im]);
        }
        ("Gauss-Bonnet Geodesic Triangle".to_string(), pts)
    }
}

impl InteractiveTool for HyperbolicDiskTool {
    fn name(&self) -> &'static str {
        "Hyperbolic Disk Explorer"
    }

    fn domain(&self) -> &'static str {
        "hyperbolic_geometry"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.last_canvas_id
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn has_finite_parameters(&self) -> bool {
        let coeffs = self.tiling.view_matrix().to_coeffs();
        let view_ok = coeffs.iter().all(|c| c.is_finite())
            && self.tiling.view_matrix().determinant() > 0.5;
        let ruler_ok = self.ruler_vertices.iter().all(|z| {
            z.re.is_finite() && z.im.is_finite() && z.norm() <= POINCARE_DISK_RADIUS_MAX + 1e-7
        });
        view_ok
            && ruler_ok
            && is_hyperbolic_schlafli(self.p, self.q)
            && (1..=MAX_TILING_DEPTH).contains(&self.depth)
    }

    fn clamp_parameters(&mut self) -> bool {
        if !is_hyperbolic_schlafli(self.p, self.q) {
            self.p = 7;
            self.q = 3;
        }
        self.depth = self.depth.clamp(1, MAX_TILING_DEPTH);
        for v in &mut self.ruler_vertices {
            *v = clamp_to_disk(*v);
        }
        let mut view = self.tiling.view_matrix();
        view.renormalize();
        let _ = self.tiling.configure(self.p, self.q, self.depth);
        self.tiling.set_view_matrix(view);
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn save_state(&self) -> Option<String> {
        let state = HyperbolicDiskState {
            model: self.model,
            p: self.p,
            q: self.q,
            depth: self.depth,
            ruler_re: [
                self.ruler_vertices[0].re,
                self.ruler_vertices[1].re,
                self.ruler_vertices[2].re,
            ],
            ruler_im: [
                self.ruler_vertices[0].im,
                self.ruler_vertices[1].im,
                self.ruler_vertices[2].im,
            ],
            view_coeffs: self.tiling.view_matrix().to_coeffs(),
            show_ruler: self.show_ruler,
        };
        serde_json::to_string(&state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: HyperbolicDiskState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid HyperbolicDiskTool state: {e}"))?;
        if !is_hyperbolic_schlafli(parsed.p, parsed.q) {
            return Err(format!(
                "Schlafli symbol {{{}, {}}} does not satisfy (p - 2)*(q - 2) > 4",
                parsed.p, parsed.q
            ));
        }
        if !(1..=MAX_TILING_DEPTH).contains(&parsed.depth) {
            return Err(format!(
                "Tiling depth {} outside 1..={MAX_TILING_DEPTH}",
                parsed.depth
            ));
        }
        let view = Su11Matrix::from_coeffs(parsed.view_coeffs)
            .map_err(|e| format!("Invalid SU(1,1) view matrix: {e}"))?;
        let mut rulers = [Complex64::new(0.0, 0.0); 3];
        for (idx, slot) in rulers.iter_mut().enumerate() {
            let re = parsed.ruler_re[idx];
            let im = parsed.ruler_im[idx];
            if !re.is_finite() || !im.is_finite() {
                return Err("Ruler vertex coordinates must be finite".into());
            }
            let z = Complex64::new(re, im);
            if z.norm() >= 1.0 {
                return Err("Ruler vertex must lie strictly inside the unit disk |z| < 1".into());
            }
            *slot = clamp_to_disk(z);
        }
        self.tiling
            .configure(parsed.p, parsed.q, parsed.depth)
            .map_err(|e| e.to_string())?;
        self.tiling.set_view_matrix(view);
        self.model = parsed.model;
        self.p = parsed.p;
        self.q = parsed.q;
        self.depth = parsed.depth;
        self.ruler_vertices = rulers;
        self.show_ruler = parsed.show_ruler;
        self.status_note = None;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_disk_tool(ui);
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.export_tiling_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        egui_plot::export_primitives::datasets_to_primitives(&self.export_datasets())
    }

    fn step_simulation(&mut self) {
        self.apply_mobius_pan(Complex64::new(0.06, 0.035));
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for HyperbolicDiskTool {
    fn theory_description(&self) -> String {
        let gb = gauss_bonnet_triangle(
            self.ruler_vertices[0],
            self.ruler_vertices[1],
            self.ruler_vertices[2],
        );
        format!(
            "Poincaré & Beltrami-Klein disk models of $\\mathbb{{H}}^2$ ($K = -1$) under $\\mathrm{{SU}}(1,1)$ Möbius isometries $z \\mapsto \\frac{{az + b}}{{\\bar{{b}}z + \\bar{{a}}}}$ ($|a|^2 - |b|^2 = 1$). Active Fuchsian {{{}, {}}} tiling with Gauss-Bonnet triangle area $\\pi - (\\alpha + \\beta + \\gamma) = {:.4}$.",
            self.p, self.q, gb.area
        )
    }

    fn phonetic_description(&self) -> String {
        "Hyperbolic disk tessellation and S U one one Mobius isometries with Gauss Bonnet curvature defect".into()
    }

    fn theory_citation(&self) -> String {
        "Poincare (1882), Acta Mathematica 1: Theory of Fuchsian Groups".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "HyperbolicDiskTool",
        display_name: "Hyperbolic Disk Explorer",
        domain: "hyperbolic_geometry",
        tags: &["hyperbolic", "poincare", "klein", "fuchsian", "gauss-bonnet"],
        build: || Box::new(HyperbolicDiskTool::default()),
    }
}

const PRESET_POINCARE_7_3: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hyperbolic Disk Explorer - {7,3} Poincare Tiling",
    "domain": "hyperbolic_geometry",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "model": "Poincare",
    "p": 7,
    "q": 3,
    "depth": 3,
    "ruler_re": [0.0, 0.3007426, 0.1875098],
    "ruler_im": [0.0, 0.0, 0.2351301],
    "view_coeffs": [1.0, 0.0, 0.0, 0.0],
    "show_ruler": true
  }
}"#;

const PRESET_KLEIN_5_4: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hyperbolic Disk Explorer - {5,4} Klein Chord Tiling",
    "domain": "hyperbolic_geometry",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "model": "Klein",
    "p": 5,
    "q": 4,
    "depth": 3,
    "ruler_re": [0.0, 0.5060819, 0.1563882],
    "ruler_im": [0.0, 0.0, 0.4813125],
    "view_coeffs": [1.0, 0.0, 0.0, 0.0],
    "show_ruler": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hyperbolic_poincare_7_3",
        title: "Heptagonal {7,3} Poincare Tiling",
        description: "Conformal {7,3} Fuchsian heptagon tessellation with orthogonal geodesic arcs and Gauss-Bonnet defect pi/21.",
        category: "Differential Geometry",
        icon: "🌐",
        domain: "hyperbolic_geometry",
        tool_name: Some("HyperbolicDiskTool"),
        preset_generator: || PRESET_POINCARE_7_3,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hyperbolic_klein_5_4",
        title: "Pentagonal {5,4} Klein Chord Tiling",
        description: "Projective Beltrami-Klein {5,4} right-angled pentagon tiling rendering hyperbolic geodesics as straight chords.",
        category: "Differential Geometry",
        icon: "🔷",
        domain: "hyperbolic_geometry",
        tool_name: Some("HyperbolicDiskTool"),
        preset_generator: || PRESET_KLEIN_5_4,
    }
}

// [cite:modular_polynomials_review]
