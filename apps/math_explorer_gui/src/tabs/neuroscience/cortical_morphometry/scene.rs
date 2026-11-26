#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D viewport scene projection (`Scene3DBuilder` + `heatmap_color_high_contrast`) and
//! watertight per-vertex colored `ColoredMesh` export for [`super::CorticalMorphometryTool`].

use super::phantom::{build_polar_faces, canonical_bits};
use super::{CorticalMorphometryTool, CorticalOverlayMode, SurfaceDisplayMode};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::LineStyle;
use math_commons::math_kernel::colormap::heatmap_color_high_contrast;
use nalgebra::{Point3, Vector3};
use oxidize_core::mesh::{ColoredMesh, Mesh, Point3D, VertexColor};

const COLOR_BINS: usize = 64;

fn lerp_point(a: [f64; 3], b: [f64; 3], t: f64) -> Point3<f64> {
    let w = t.clamp(0.0, 1.0);
    Point3::new(
        a[0] + (b[0] - a[0]) * w,
        a[1] + (b[1] - a[1]) * w,
        a[2] + (b[2] - a[2]) * w,
    )
}

fn scalar_bounds(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

fn normalize_scalar(val: f64, lo: f64, hi: f64) -> f64 {
    let span = hi - lo;
    if !val.is_finite() || !span.is_finite() || span.abs() < 1e-10 {
        0.5
    } else {
        ((val - lo) / span).clamp(0.0, 1.0)
    }
}

fn build_high_contrast_lut() -> [egui::Color32; COLOR_BINS] {
    let denom = (COLOR_BINS - 1) as f64;
    std::array::from_fn(|idx| {
        let rgb = heatmap_color_high_contrast(idx as f64 / denom);
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    })
}

fn quantize_bin(val: f64, lo: f64, hi: f64) -> usize {
    let t = normalize_scalar(val, lo, hi);
    ((t * (COLOR_BINS - 1) as f64).round() as usize).min(COLOR_BINS - 1)
}

fn emit_polar_wireframe_edges(
    builder: &mut crate::framework::Scene3DBuilder,
    series_name: &str,
    verts: &[Point3<f64>],
    scalars: Option<(&[f64], f64, f64, &[egui::Color32; COLOR_BINS])>,
    fallback_stroke: egui::Stroke,
    nu: usize,
    nv: usize,
) {
    let ring = |j: usize, i: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
    let edge_stroke = |a: usize, b: usize| -> egui::Stroke {
        if let Some((vals, lo, hi, lut)) = scalars {
            let mid = 0.5 * (vals[a] + vals[b]);
            egui::Stroke::new(fallback_stroke.width, lut[quantize_bin(mid, lo, hi)])
        } else {
            fallback_stroke
        }
    };
    let pole_stride = (nu / 12).max(1);
    for i in (0..nu).step_by(pole_stride) {
        let r_top = ring(1, i);
        builder.add_styled_segment(
            series_name,
            [verts[0], verts[r_top]],
            edge_stroke(0, r_top),
            LineStyle::Solid,
            false,
            false,
        );
        let r_bot = ring(nv - 1, i);
        builder.add_styled_segment(
            series_name,
            [verts[1], verts[r_bot]],
            edge_stroke(1, r_bot),
            LineStyle::Solid,
            false,
            false,
        );
    }
    for j in 1..nv {
        for i in 0..nu {
            let a = ring(j, i);
            let b = ring(j, i + 1);
            builder.add_styled_segment(
                series_name,
                [verts[a], verts[b]],
                edge_stroke(a, b),
                LineStyle::Solid,
                true,
                false,
            );
            if j + 1 < nv {
                let c = ring(j + 1, i);
                builder.add_styled_segment(
                    series_name,
                    [verts[a], verts[c]],
                    edge_stroke(a, c),
                    LineStyle::Solid,
                    true,
                    false,
                );
            }
        }
    }
}

fn compute_vertex_normals(centered_verts: &[Vector3<f64>], faces: &[[usize; 3]]) -> Vec<Point3D> {
    let mut accum = vec![Vector3::zeros(); centered_verts.len()];
    for &[i0, i1, i2] in faces {
        let e1 = centered_verts[i1] - centered_verts[i0];
        let e2 = centered_verts[i2] - centered_verts[i0];
        let n = e1.cross(&e2);
        if n.x.is_finite() && n.y.is_finite() && n.z.is_finite() {
            accum[i0] += n;
            accum[i1] += n;
            accum[i2] += n;
        }
    }
    accum
        .into_iter()
        .enumerate()
        .map(|(idx, n)| {
            let radial = centered_verts[idx]
                .try_normalize(1e-12)
                .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));
            let unit = n.try_normalize(1e-12).unwrap_or(radial);
            Point3D::new(unit.x as f32, unit.y as f32, unit.z as f32)
        })
        .collect()
}

impl CorticalMorphometryTool {
    fn active_scalars(&self) -> &[f64] {
        match self.overlay_mode {
            CorticalOverlayMode::Thickness => &self.cache.thickness_mm,
            CorticalOverlayMode::GlmTStatistic => &self.cache.glm_t_stats,
            CorticalOverlayMode::MrfTissueClass => &self.cache.gm_posteriors,
            CorticalOverlayMode::SulcalDepth => &self.cache.sulcal_depth_mm,
        }
    }

    pub(super) fn blended_vertices(&self, pial: bool) -> Vec<Point3<f64>> {
        let (folded_opt, inflated) = if pial {
            (&self.cache.pial_surface, &self.cache.pial_inflated)
        } else {
            (&self.cache.white_surface, &self.cache.white_inflated)
        };
        let Some(folded) = folded_opt.as_ref() else {
            return Vec::new();
        };
        folded
            .vertices
            .iter()
            .zip(inflated)
            .map(|(&f, &inf)| lerp_point(f, inf, self.inflation))
            .collect()
    }

    /// Rebuilds the depth-sorted 3D cortical wireframe and sidecar datasets when state or pose changes.
    pub fn populate_scene(&mut self, dark_mode: bool) {
        self.ensure_evaluated();
        let proj_key = (
            self.cache.compute_key.unwrap_or(1),
            self.camera.pose_fingerprint(),
            self.overlay_mode as u8,
            self.surface_mode as u8,
            canonical_bits(self.inflation) ^ u64::from(dark_mode),
        );
        if self.cache.proj_key == Some(proj_key) {
            return;
        }
        let (nu, nv) = (
            self.u_resolution.clamp(12, 48),
            self.v_resolution.clamp(12, 36),
        );
        let (pial_verts, white_verts) = (self.blended_vertices(true), self.blended_vertices(false));
        let scalars = self.active_scalars().to_vec();
        let (lo, hi) = scalar_bounds(&scalars);
        let lut = build_high_contrast_lut();

        self.scene_builder.begin(self.camera);
        self.scene_builder.add_axes(34.0, dark_mode);
        match self.surface_mode {
            SurfaceDisplayMode::Pial => self.emit_single_shell(
                "Pial Cortical Surface",
                &pial_verts,
                (&scalars, lo, hi, &lut),
                (nu, nv),
                lut[COLOR_BINS - 8],
            ),
            SurfaceDisplayMode::White => self.emit_single_shell(
                "White Cortical Surface",
                &white_verts,
                (&scalars, lo, hi, &lut),
                (nu, nv),
                lut[12],
            ),
            SurfaceDisplayMode::DualShell => {
                self.emit_dual_shell(
                    &pial_verts,
                    &white_verts,
                    (&scalars, lo, hi, &lut),
                    (nu, nv),
                    dark_mode,
                );
            }
        }
        self.scene_builder.ensure_sorted();
        self.cache.proj_key = Some(proj_key);
    }

    fn emit_single_shell(
        &mut self,
        label: &'static str,
        verts: &[Point3<f64>],
        overlay: (&[f64], f64, f64, &[egui::Color32; COLOR_BINS]),
        res: (usize, usize),
        series_color: egui::Color32,
    ) {
        let base_stroke = egui::Stroke::new(1.55_f32, overlay.3[COLOR_BINS / 2]);
        emit_polar_wireframe_edges(
            &mut self.scene_builder,
            label,
            verts,
            Some(overlay),
            base_stroke,
            res.0,
            res.1,
        );
        self.scene_builder.record_series(label, verts, series_color);
    }

    fn emit_dual_shell(
        &mut self,
        pial_verts: &[Point3<f64>],
        white_verts: &[Point3<f64>],
        overlay: (&[f64], f64, f64, &[egui::Color32; COLOR_BINS]),
        res: (usize, usize),
        dark_mode: bool,
    ) {
        let inner_col = if dark_mode {
            egui::Color32::from_rgba_unmultiplied(150, 175, 210, 105)
        } else {
            egui::Color32::from_rgba_unmultiplied(70, 90, 120, 115)
        };
        emit_polar_wireframe_edges(
            &mut self.scene_builder,
            "White Matter Boundary",
            white_verts,
            None,
            egui::Stroke::new(0.9_f32, inner_col),
            res.0,
            res.1,
        );
        emit_polar_wireframe_edges(
            &mut self.scene_builder,
            "Pial Cortical Surface",
            pial_verts,
            Some(overlay),
            egui::Stroke::new(1.5_f32, overlay.3[COLOR_BINS / 2]),
            res.0,
            res.1,
        );
        self.scene_builder.record_series(
            "Pial Cortical Surface",
            pial_verts,
            overlay.3[COLOR_BINS - 8],
        );
        self.scene_builder
            .record_series("White Matter Boundary", white_verts, inner_col);
    }

    fn clone_and_evaluate(&self) -> Self {
        let mut tmp = Self {
            overlay_mode: self.overlay_mode,
            surface_mode: self.surface_mode,
            inflation: self.inflation,
            u_resolution: self.u_resolution,
            v_resolution: self.v_resolution,
            sulcal_depth_mm: self.sulcal_depth_mm,
            base_thickness_mm: self.base_thickness_mm,
            alpha: self.alpha,
            beta: self.beta,
            sigma: self.sigma,
            learning_rate: self.learning_rate,
            evolution_steps: self.evolution_steps,
            potts_beta: self.potts_beta,
            mri_noise_level: self.mri_noise_level,
            atrophy_effect_mm: self.atrophy_effect_mm,
            collinear_glm_design: self.collinear_glm_design,
            camera: self.camera,
            scene_builder: crate::framework::Scene3DBuilder::new(self.camera),
            cache: super::phantom::CorticalPhantomCache::default(),
            diagnostic_report: None,
        };
        tmp.ensure_evaluated();
        tmp
    }

    pub(super) fn build_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        if !self.has_finite_parameters() {
            return Err(
                "Cortical morphometry parameters contain non-finite or out-of-bounds values"
                    .to_string(),
            );
        }
        let warm = self.cache.compute_key == Some(self.compute_fingerprint())
            && self.cache.white_surface.is_some()
            && self.cache.pial_surface.is_some();
        let fallback_eval = (!warm).then(|| self.clone_and_evaluate());
        let source = fallback_eval.as_ref().unwrap_or(self);
        let use_pial = !matches!(source.surface_mode, SurfaceDisplayMode::White);
        let world_verts = source.blended_vertices(use_pial);
        if world_verts.is_empty() {
            return Err("Cortical surface vertices unavailable".to_string());
        }
        let mut centered = Vec::with_capacity(world_verts.len());
        let mut vertices = Vec::with_capacity(world_verts.len());
        for p in &world_verts {
            let c = p.coords;
            if !c.x.is_finite() || !c.y.is_finite() || !c.z.is_finite() {
                return Err("Cortical mesh contains non-finite vertex coordinates".to_string());
            }
            centered.push(c);
            vertices.push(Point3D::new(c.x as f32, c.y as f32, c.z as f32));
        }
        let nu = source.u_resolution.clamp(12, 48);
        let nv = source.v_resolution.clamp(12, 36);
        let faces = build_polar_faces(nu, nv);
        let normals = compute_vertex_normals(&centered, &faces);
        let mut indices = Vec::with_capacity(faces.len() * 3);
        for &[i0, i1, i2] in &faces {
            indices.extend_from_slice(&[i0, i1, i2]);
        }
        let scalars = source.active_scalars();
        let (lo, hi) = scalar_bounds(scalars);
        let colors = scalars
            .iter()
            .map(|&v| {
                let rgb = heatmap_color_high_contrast(normalize_scalar(v, lo, hi));
                VertexColor::new(rgb[0], rgb[1], rgb[2])
            })
            .collect();
        Ok(ColoredMesh::new(
            Mesh {
                vertices,
                normals,
                indices,
            },
            colors,
        ))
    }
}
