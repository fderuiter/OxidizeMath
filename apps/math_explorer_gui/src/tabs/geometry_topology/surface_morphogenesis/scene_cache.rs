#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cached 3D scene projection for [`super::SurfaceMorphogenesisTool`].

use eframe::egui;
use egui_plot::LineStyle;
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::buckled_flow::SurfaceTopology;

use super::SurfaceMorphogenesisTool;

const COLOR_BINS: usize = 24;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct MorphogenesisSceneKey {
    geometry_key: u64,
    kinetics_tag: u8,
    overlay_tag: u8,
    step_count: u64,
    mutation_epoch: u64,
    camera_pose: u64,
    series_rgba: [u8; 4],
}

impl SurfaceMorphogenesisTool {
    pub(super) fn populate_scene(&mut self, series_color: egui::Color32) {
        let Some(eng) = self.engine.as_ref() else {
            return;
        };
        let key = MorphogenesisSceneKey {
            geometry_key: eng.geometry_key,
            kinetics_tag: eng.kinetics_kind as u8,
            overlay_tag: self.overlay_mode as u8,
            step_count: eng.step_count,
            mutation_epoch: self.scene_mutation_epoch,
            camera_pose: self.camera.pose_fingerprint(),
            series_rgba: series_color.to_array(),
        };
        if self.cached_scene_key == Some(key) {
            return;
        }
        let pts = eng.flow.points();
        let scalars = eng.overlay_field(self.overlay_mode);
        let (min_s, max_s, lut) = build_overlay_lut(scalars);
        self.scene_builder.begin(self.camera);
        emit_morphogenesis_wireframe(&mut self.scene_builder, eng, scalars, min_s, max_s, &lut);
        self.scene_builder
            .record_series("Surface Morphogenesis", pts, series_color);
        self.scene_builder.ensure_sorted();
        self.cached_scene_key = Some(key);
    }
}

fn emit_morphogenesis_wireframe(
    scene_builder: &mut crate::camera::scene::Scene3DBuilder,
    eng: &super::solver::MorphogenesisSurfaceEngine,
    scalars: &[f64],
    min_s: f64,
    max_s: f64,
    lut: &[egui::Color32; COLOR_BINS],
) {
    let pts = eng.flow.points();
    let (nu, nv) = (eng.nu, eng.nv);
    let wrap_periodic = matches!(
        eng.topology,
        SurfaceTopology::PeriodicTorus | SurfaceTopology::TwistedKlein
    );
    let wrap_sphere_u = eng.topology == SurfaceTopology::PolarSphere;
    for i in 0..nu {
        for j in 0..nv {
            let idx = flatten_2d_index(j, i, nv);
            let p0 = pts[idx];
            let s0 = scalars[idx];
            if j + 1 < nv || wrap_periodic {
                let idx_v = flatten_2d_index((j + 1) % nv, i, nv);
                let col = lut[quantize_bin(0.5 * (s0 + scalars[idx_v]), min_s, max_s)];
                scene_builder.add_styled_segment(
                    "Surface Morphogenesis",
                    [p0, pts[idx_v]],
                    egui::Stroke::new(1.35_f32, col),
                    LineStyle::Solid,
                    true,
                    false,
                );
            }
            if i + 1 < nu || wrap_periodic || wrap_sphere_u {
                let i_next = (i + 1) % nu;
                let j_seam = if eng.topology == SurfaceTopology::TwistedKlein && i + 1 >= nu {
                    eng.flow.seam_forward_map()[j]
                } else {
                    j
                };
                let idx_u = flatten_2d_index(j_seam, i_next, nv);
                let col = lut[quantize_bin(0.5 * (s0 + scalars[idx_u]), min_s, max_s)];
                scene_builder.add_styled_segment(
                    "Surface Morphogenesis",
                    [p0, pts[idx_u]],
                    egui::Stroke::new(1.35_f32, col),
                    LineStyle::Solid,
                    true,
                    false,
                );
            }
        }
    }
}

fn build_overlay_lut(values: &[f64]) -> (f64, f64, [egui::Color32; COLOR_BINS]) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    let mut lut = [egui::Color32::WHITE; COLOR_BINS];
    for (idx, slot) in lut.iter_mut().enumerate() {
        let t = idx as f64 / (COLOR_BINS - 1) as f64;
        let rgb = mapped_gradient(t, 0.0, 1.0);
        *slot = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    }
    (min_v, max_v, lut)
}

fn quantize_bin(val: f64, min_v: f64, max_v: f64) -> usize {
    let span = max_v - min_v;
    if !val.is_finite() || span.abs() < 1e-10 {
        return COLOR_BINS / 2;
    }
    let norm = ((val - min_v) / span).clamp(0.0, 1.0);
    ((norm * (COLOR_BINS - 1) as f64).round() as usize).min(COLOR_BINS - 1)
}
