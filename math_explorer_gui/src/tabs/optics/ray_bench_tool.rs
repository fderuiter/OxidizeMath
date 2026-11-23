#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D optical bench, GRIN metamaterial ray tracer, and caustic envelope analyzer.

// [cite:optics]

use eframe::egui::{self, Color32, ColorImage};
use egui_plot::ExportPrimitive;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::optics::{
    brewster_angle, cauchy_refractive_index, critical_angle, extract_caustic_envelope,
    fresnel_polarized, trace_ray_fan, CausticEnvelopeBuffer, GlassMaterial, PolarizationMode,
    RayFanConfig, RayFanWorkspace,
};
use serde::{Deserialize, Serialize};

use super::presets::{clamp_finite, OpticsColormap, RayBenchSceneChoice, RaySecondaryPlot};
use crate::app::palette::DARK_PALETTE;
use crate::framework::{ExportCapabilities, FiniteStateGuard, InteractiveTool, ToolStateBounds};

pub(super) const BENCH_EXTENT: f64 = 2.6;
pub(super) const RASTER_SIDE: usize = 160;
const MAX_RAYS: usize = 320;
const MAX_PTS_PER_RAY: usize = 220;

/// Serializable parameter configuration for [`CausticsRayBenchTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausticsRayBenchState {
    /// Active optical element or continuous GRIN medium.
    pub scene: RayBenchSceneChoice,
    /// Dispersive glass material for lenses and prisms.
    pub glass: GlassMaterial,
    /// Incident polarization state.
    pub polarization: PolarizationMode,
    /// WCAG-compliant ray and caustic colormap.
    pub colormap: OpticsColormap,
    /// Secondary graph mode (Fresnel $R_s/R_p$ curves or detector caustic histogram).
    pub secondary_plot: RaySecondaryPlot,
    /// Number of spatial rays in the incident bundle (`3..=61`).
    pub ray_count: usize,
    /// Half-width of the incident beam (`0.02..=2.2`).
    pub beam_half_width: f64,
    /// Source plane x-coordinate (`-2.5..=-0.3`).
    pub source_x: f64,
    /// Source center y-offset (`-1.6..=1.6`).
    pub source_y: f64,
    /// Launch angle relative to $+x$ in radians (`-1.2..=1.2`).
    pub launch_angle_rad: f64,
    /// Monochromatic wavelength in nanometers (`380.0..=720.0`).
    pub wavelength_nm: f64,
    /// Expands each spatial ray into a 5-band polychromatic fan when enabled.
    pub polychromatic: bool,
    /// Characteristic radius or size scale of the optical element (`0.4..=2.4`).
    pub element_scale: f64,
    /// Apex angle in degrees for the dispersive prism (`25.0..=75.0`).
    pub prism_apex_deg: f64,
    /// Second medium refractive index $n_2$ for the planar Fresnel interface (`1.0..=2.5`).
    pub interface_n2: f64,
    /// Transverse detector screen x-coordinate (`-2.2..=2.4`).
    pub detector_x: f64,
    /// Overlays extracted adjacent-ray caustic focal points when enabled.
    pub show_caustic_envelope: bool,
}

impl Default for CausticsRayBenchState {
    fn default() -> Self {
        Self {
            scene: RayBenchSceneChoice::SphericalBiconvexLens,
            glass: GlassMaterial::Bk7Crown,
            polarization: PolarizationMode::Unpolarized,
            colormap: OpticsColormap::PerceptualThermal,
            secondary_plot: RaySecondaryPlot::FresnelPolarization,
            ray_count: 25,
            beam_half_width: 0.78,
            source_x: -2.1,
            source_y: 0.0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            element_scale: 1.25,
            prism_apex_deg: 55.0,
            interface_n2: 1.0,
            detector_x: 1.5,
            show_caustic_envelope: true,
        }
    }
}

/// Interactive 2D geometric ray-tracing, GRIN metamaterial, and caustic envelope tool.
pub struct CausticsRayBenchTool {
    pub(crate) state: CausticsRayBenchState,
    pub(crate) workspace: RayFanWorkspace,
    pub(crate) caustics: CausticEnvelopeBuffer,
    pub(super) focused_canvas: Option<egui::Id>,
    pub(super) revision: u64,
}

impl Default for CausticsRayBenchTool {
    fn default() -> Self {
        let mut tool = Self {
            state: CausticsRayBenchState::default(),
            workspace: RayFanWorkspace::new(MAX_RAYS, MAX_PTS_PER_RAY),
            caustics: CausticEnvelopeBuffer::new(512, 64, -BENCH_EXTENT, BENCH_EXTENT),
            focused_canvas: None,
            revision: 1,
        };
        tool.recompute();
        tool
    }
}

impl CausticsRayBenchTool {
    pub(crate) fn ray_fan_config(&self) -> RayFanConfig {
        let s = &self.state;
        RayFanConfig {
            scene: s
                .scene
                .build_scene(s.element_scale, s.prism_apex_deg, s.glass, s.interface_n2),
            ray_count: s.ray_count,
            beam_half_width: s.beam_half_width,
            source_x: s.source_x,
            source_y: s.source_y,
            launch_angle_rad: s.launch_angle_rad,
            wavelength_nm: s.wavelength_nm,
            polychromatic: s.polychromatic,
            polarization: s.polarization,
            bench_half_extent: BENCH_EXTENT,
        }
    }

    pub(crate) fn recompute(&mut self) {
        let cfg = self.ray_fan_config();
        trace_ray_fan(&cfg, &mut self.workspace);
        extract_caustic_envelope(
            &self.workspace,
            self.state.detector_x,
            BENCH_EXTENT,
            &mut self.caustics,
        );
    }

    pub(crate) fn interface_indices(&self) -> (f64, f64) {
        if self.state.scene == RayBenchSceneChoice::PlanarFresnelInterface {
            (1.5168, self.state.interface_n2)
        } else {
            (
                1.0,
                cauchy_refractive_index(self.state.glass.cauchy(), self.state.wavelength_nm),
            )
        }
    }

    pub(super) fn is_valid_state(s: &CausticsRayBenchState) -> bool {
        (3..=61).contains(&s.ray_count)
            && s.beam_half_width.is_finite()
            && (0.02..=2.2).contains(&s.beam_half_width)
            && s.source_x.is_finite()
            && (-2.5..=-0.3).contains(&s.source_x)
            && s.source_y.is_finite()
            && (-1.6..=1.6).contains(&s.source_y)
            && s.launch_angle_rad.is_finite()
            && (-1.2..=1.2).contains(&s.launch_angle_rad)
            && s.wavelength_nm.is_finite()
            && (380.0..=720.0).contains(&s.wavelength_nm)
            && s.element_scale.is_finite()
            && (0.4..=2.4).contains(&s.element_scale)
            && s.prism_apex_deg.is_finite()
            && (25.0..=75.0).contains(&s.prism_apex_deg)
            && s.interface_n2.is_finite()
            && (1.0..=2.5).contains(&s.interface_n2)
            && s.detector_x.is_finite()
            && (-2.2..=2.4).contains(&s.detector_x)
    }

    pub(super) fn for_each_element_segment(&self, mut emit: impl FnMut([f64; 2], [f64; 2])) {
        let r = self.state.element_scale.clamp(0.4, 2.4);
        match self.state.scene {
            RayBenchSceneChoice::CoffeeCupNephroid => {
                for k in 0..32 {
                    let a0 =
                        -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * (k as f64) / 32.0;
                    let a1 = -std::f64::consts::FRAC_PI_2
                        + std::f64::consts::PI * ((k + 1) as f64) / 32.0;
                    emit([r * a0.cos(), r * a0.sin()], [r * a1.cos(), r * a1.sin()]);
                }
            }
            RayBenchSceneChoice::SphericalBiconvexLens => {
                let (rc, ht) = (1.35 * r, 0.18 * r);
                let y_lim = (0.72 * r).min((rc * rc - (rc - ht).powi(2)).max(0.0).sqrt());
                for k in 0..24 {
                    let y0 = -y_lim + 2.0 * y_lim * (k as f64) / 24.0;
                    let y1 = -y_lim + 2.0 * y_lim * ((k + 1) as f64) / 24.0;
                    let dx0 = (rc * rc - y0 * y0).max(0.0).sqrt() - (rc - ht);
                    let dx1 = (rc * rc - y1 * y1).max(0.0).sqrt() - (rc - ht);
                    emit([-dx0, y0], [-dx1, y1]);
                    emit([dx0, y0], [dx1, y1]);
                }
            }
            RayBenchSceneChoice::DispersivePrism => {
                let half_a = 0.5 * self.state.prism_apex_deg.clamp(25.0, 75.0).to_radians();
                let side = 1.25 * r;
                let (h, w) = (side * half_a.cos(), side * half_a.sin());
                let v = [[0.0, 0.6 * h], [-w, -0.4 * h], [w, -0.4 * h]];
                emit(v[0], v[1]);
                emit(v[1], v[2]);
                emit(v[2], v[0]);
            }
            RayBenchSceneChoice::PlanarFresnelInterface => {
                emit([0.0, -2.4], [0.0, 2.4]);
            }
            RayBenchSceneChoice::ParabolicGrinFiber => {
                let y_clad = (1.35 * r).min(2.2);
                emit([-2.5, y_clad], [2.5, y_clad]);
                emit([-2.5, -y_clad], [2.5, -y_clad]);
            }
            RayBenchSceneChoice::LuneburgMetamaterial
            | RayBenchSceneChoice::MaxwellFisheye
            | RayBenchSceneChoice::EatonRetroreflector => {
                for k in 0..40 {
                    let a0 = std::f64::consts::TAU * (k as f64) / 40.0;
                    let a1 = std::f64::consts::TAU * ((k + 1) as f64) / 40.0;
                    emit([r * a0.cos(), r * a0.sin()], [r * a1.cos(), r * a1.sin()]);
                }
            }
        }
    }

    fn build_export_image(&self) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; RASTER_SIDE * RASTER_SIDE];
        let to_px = |wx: f64, wy: f64| -> (isize, isize) {
            let nx = ((wx + BENCH_EXTENT) / (2.0 * BENCH_EXTENT)).clamp(0.0, 1.0);
            let ny = ((BENCH_EXTENT - wy) / (2.0 * BENCH_EXTENT)).clamp(0.0, 1.0);
            (
                (nx * ((RASTER_SIDE - 1) as f64)).round() as isize,
                (ny * ((RASTER_SIDE - 1) as f64)).round() as isize,
            )
        };
        self.for_each_element_segment(|a, b| {
            rasterize_line(
                &mut pixels,
                to_px(a[0], a[1]),
                to_px(b[0], b[1]),
                DARK_PALETTE.node_stroke,
            );
        });
        for (idx, meta) in self.workspace.rays().iter().enumerate() {
            let col = self.state.colormap.ray_stroke_color(meta.wavelength_nm);
            for seg in self.workspace.ray_vertices(idx).windows(2) {
                rasterize_line(
                    &mut pixels,
                    to_px(seg[0][0], seg[0][1]),
                    to_px(seg[1][0], seg[1][1]),
                    col,
                );
            }
        }
        for cp in self.caustics.points() {
            let (cx, cy) = to_px(cp.position[0], cp.position[1]);
            if (0..RASTER_SIDE as isize).contains(&cx) && (0..RASTER_SIDE as isize).contains(&cy) {
                pixels[flatten_2d_index(cx as usize, cy as usize, RASTER_SIDE)] =
                    DARK_PALETTE.warning;
            }
        }
        ColorImage {
            size: [RASTER_SIDE, RASTER_SIDE],
            pixels,
            ..Default::default()
        }
    }

    fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut out = Vec::new();
        for (idx, meta) in self.workspace.rays().iter().enumerate() {
            let pts: Vec<[f64; 2]> = self
                .workspace
                .ray_vertices(idx)
                .iter()
                .copied()
                .filter(|p| p[0].is_finite() && p[1].is_finite())
                .collect();
            if pts.len() >= 2 {
                out.push(ExportPrimitive::Polyline {
                    name: if idx == 0 {
                        "Traced Rays".into()
                    } else {
                        String::new()
                    },
                    points: pts.into(),
                    color: self.state.colormap.ray_stroke_color(meta.wavelength_nm),
                    stroke_width: 1.4,
                    closed: false,
                    fill: None,
                });
            }
        }
        if self.state.show_caustic_envelope && !self.caustics.points().is_empty() {
            out.push(ExportPrimitive::Points {
                name: "Caustic Envelope".into(),
                points: self.caustics.points().iter().map(|c| c.position).collect(),
                color: DARK_PALETTE.warning,
                radius: 2.2,
            });
        }
        out
    }

    fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (n1, n2) = self.interface_indices();
        let rs_curve: Vec<[f64; 2]> = (0..=90)
            .map(|deg| {
                let rad = (deg as f64).min(89.5).to_radians();
                [deg as f64, fresnel_polarized(n1, n2, rad).reflectance_s]
            })
            .collect();
        let rp_curve: Vec<[f64; 2]> = (0..=90)
            .map(|deg| {
                let rad = (deg as f64).min(89.5).to_radians();
                [deg as f64, fresnel_polarized(n1, n2, rad).reflectance_p]
            })
            .collect();
        let det_curve: Vec<[f64; 2]> = self
            .caustics
            .detector_bins()
            .iter()
            .enumerate()
            .map(|(i, &v)| [self.caustics.bin_center_y(i), v])
            .collect();
        let mut datasets = vec![
            ("fresnel_reflectance_s".into(), rs_curve),
            ("fresnel_reflectance_p".into(), rp_curve),
            ("detector_caustic_histogram".into(), det_curve),
        ];
        if !self.caustics.points().is_empty() {
            datasets.push((
                "caustic_envelope_xy".into(),
                self.caustics.points().iter().map(|c| c.position).collect(),
            ));
        }
        datasets
    }
}

fn rasterize_line(pixels: &mut [Color32], p0: (isize, isize), p1: (isize, isize), color: Color32) {
    let steps = (p1.0 - p0.0).abs().max((p1.1 - p0.1).abs()).clamp(1, 320);
    for s in 0..=steps {
        let x = p0.0 + (p1.0 - p0.0) * s / steps;
        let y = p0.1 + (p1.1 - p0.1) * s / steps;
        if (0..RASTER_SIDE as isize).contains(&x) && (0..RASTER_SIDE as isize).contains(&y) {
            pixels[flatten_2d_index(x as usize, y as usize, RASTER_SIDE)] = color;
        }
    }
}

impl FiniteStateGuard for CausticsRayBenchTool {
    fn is_finite_state(&self) -> bool {
        Self::is_valid_state(&self.state)
    }

    fn auto_scale_step_size(&mut self) {
        self.state.beam_half_width = (self.state.beam_half_width * 0.8).clamp(0.05, 2.0);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn clamp_parameters(&mut self) {
        let s = &mut self.state;
        s.ray_count = s.ray_count.clamp(3, 61);
        s.beam_half_width = clamp_finite(s.beam_half_width, 0.02, 2.2, 0.78);
        s.source_x = clamp_finite(s.source_x, -2.5, -0.3, -2.1);
        s.source_y = clamp_finite(s.source_y, -1.6, 1.6, 0.0);
        s.launch_angle_rad = clamp_finite(s.launch_angle_rad, -1.2, 1.2, 0.0);
        s.wavelength_nm = clamp_finite(s.wavelength_nm, 380.0, 720.0, 550.0);
        s.element_scale = clamp_finite(s.element_scale, 0.4, 2.4, 1.25);
        s.prism_apex_deg = clamp_finite(s.prism_apex_deg, 25.0, 75.0, 55.0);
        s.interface_n2 = clamp_finite(s.interface_n2, 1.0, 2.5, 1.0);
        s.detector_x = clamp_finite(s.detector_x, -2.2, 2.4, 1.5);
        self.revision = self.revision.wrapping_add(1).max(1);
        self.recompute();
    }
}

impl InteractiveTool for CausticsRayBenchTool {
    fn name(&self) -> &'static str {
        "Caustics & Ray Bench"
    }
    fn domain(&self) -> &'static str {
        "optics"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 5,
            max_array_len: 16,
        }
    }
    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }
    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: CausticsRayBenchState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid CausticsRayBenchTool state JSON: {e}"))?;
        if !Self::is_valid_state(&parsed) {
            return Err("CausticsRayBenchTool parameters out of valid physical bounds".into());
        }
        self.state = parsed;
        self.revision = self.revision.wrapping_add(1).max(1);
        self.recompute();
        Ok(())
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(self.build_export_image())
    }
    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }
    fn step_simulation(&mut self) {
        self.recompute();
    }
    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if !self.has_finite_parameters() {
            FiniteStateGuard::clamp_parameters(self);
        }
        self.draw_controls_panel(parent_ui);
        self.draw_central_workspace(parent_ui);
    }
}

impl scientific_metadata::theory::TheoryDescribable for CausticsRayBenchTool {
    fn theory_description(&self) -> String {
        let (n1, n2) = self.interface_indices();
        let tb_deg = brewster_angle(n1, n2).to_degrees();
        let tc_str = critical_angle(n1, n2)
            .map(|c| format!(", $\\theta_c = {:.1}^\\circ$", c.to_degrees()))
            .unwrap_or_default();
        format!(
            "Canonical eikonal ray tracing $\\dot{{\\mathbf{{r}}}} = \\mathbf{{p}}, \\dot{{\\mathbf{{p}}}} = \\frac{{1}}{{2}}\\nabla(n^2)$ ({}) with polarized Fresnel interfaces ($\\theta_B = {:.1}^\\circ${tc_str}) and caustic envelope extraction.",
            self.state.scene.label(),
            tb_deg
        )
    }
    fn phonetic_description(&self) -> String {
        "Geometric ray bench, graded index metamaterial, and caustic envelope simulator".into()
    }
    fn theory_citation(&self) -> String {
        "Born & Wolf (1999), Principles of Optics (born1999principles); Luneburg (1964), Mathematical Theory of Optics (luneburg1964mathematical); Berry & Upstill (1980) (berry1980catastrophe)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
