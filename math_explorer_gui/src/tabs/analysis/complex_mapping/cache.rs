#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ui::{split_contour_segments, CONTOUR_RESOLUTION};
use super::{ComplexMappingTool, MappingFunction};
use eframe::egui::Color32;
use egui_plot::{ColoredSegment, PlotBounds, PlotPoint};
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

type ColoredBatch = (Arc<Vec<ColoredSegment>>, PlotBounds);

pub(crate) struct ContourGridCache {
    key: Option<u64>,
    palette_pair: Option<[Color32; 2]>,
    z_raw_vert: Vec<Vec<[f64; 2]>>,
    z_raw_horiz: Vec<Vec<[f64; 2]>>,
    w_raw_vert: Vec<Vec<[f64; 2]>>,
    w_raw_horiz: Vec<Vec<[f64; 2]>>,
    pub(super) z_batch: ColoredBatch,
    pub(super) w_batch: ColoredBatch,
    probe_key: Option<u64>,
    pub(super) cached_w0: Complex64,
    pub(super) cached_scale: f64,
    pub(crate) eval_count: usize,
    pub(crate) probe_eval_count: usize,
}

impl Default for ContourGridCache {
    fn default() -> Self {
        let empty = || (Arc::new(Vec::new()), PlotBounds::NOTHING);
        Self {
            key: None,
            palette_pair: None,
            z_raw_vert: Vec::new(),
            z_raw_horiz: Vec::new(),
            w_raw_vert: Vec::new(),
            w_raw_horiz: Vec::new(),
            z_batch: empty(),
            w_batch: empty(),
            probe_key: None,
            cached_w0: Complex64::new(0.0, 0.0),
            cached_scale: 0.0,
            eval_count: 0,
            probe_eval_count: 0,
        }
    }
}

fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

fn build_colored_grid_batch(
    vert: &[Vec<[f64; 2]>],
    horiz: &[Vec<[f64; 2]>],
    colors: [Color32; 2],
) -> ColoredBatch {
    let mut bounds = PlotBounds::NOTHING;
    let mut out = Vec::with_capacity(vert.len() + horiz.len());
    for (family, color) in [(vert, colors[0]), (horiz, colors[1])] {
        for seg in family {
            if seg.len() < 2 {
                continue;
            }
            let mut pts = Vec::with_capacity(seg.len());
            for &[x, y] in seg {
                let pt = PlotPoint::new(x, y);
                bounds.extend_with(&pt);
                pts.push(pt);
            }
            out.push(ColoredSegment::new(pts, color, 1.5));
        }
    }
    (Arc::new(out), bounds)
}

impl ComplexMappingTool {
    pub(crate) fn contour_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.function as u8).hash(&mut h);
        canonical_bits(self.grid_min).hash(&mut h);
        canonical_bits(self.grid_max).hash(&mut h);
        self.grid_density.hash(&mut h);
        if self.function == MappingFunction::Custom {
            self.math_input.state_fingerprint().hash(&mut h);
        }
        h.finish().max(1)
    }

    fn probe_cache_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.function as u8).hash(&mut h);
        canonical_bits(self.inspection_point[0]).hash(&mut h);
        canonical_bits(self.inspection_point[1]).hash(&mut h);
        if self.function == MappingFunction::Custom {
            self.math_input.state_fingerprint().hash(&mut h);
        }
        h.finish().max(1)
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.contour_geometry_key().hash(&mut h);
        canonical_bits(self.inspection_point[0]).hash(&mut h);
        canonical_bits(self.inspection_point[1]).hash(&mut h);
        self.math_input.raw_text.hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn ensure_cached_probe(&mut self) -> (Complex64, f64) {
        let p_key = self.probe_cache_key();
        if self.cache.probe_key != Some(p_key) {
            let z0 = Complex64::new(self.inspection_point[0], self.inspection_point[1]);
            let w0 = self.eval_z(z0);
            let scale = conformal_scale_factor(|z| self.deriv_z(z), z0);
            self.cache.cached_w0 = w0;
            self.cache.cached_scale = if scale.is_finite() { scale } else { 0.0 };
            self.cache.probe_key = Some(p_key);
            self.cache.probe_eval_count += 1;
        }
        (self.cache.cached_w0, self.cache.cached_scale)
    }

    pub(super) fn ensure_cached_contours(
        &mut self,
        palette: &crate::app::SemanticPalette,
    ) -> bool {
        self.ensure_cached_probe();
        let grid_ok = self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && self.grid_max > self.grid_min
            && (2..=crate::presets::MAX_INTERACTIVE_CONFORMAL_GRID_SIZE)
                .contains(&self.grid_density);
        if !grid_ok {
            return false;
        }
        let key = self.contour_geometry_key();
        let colors = [palette.series_color(0), palette.series_color(1)];
        let geom_dirty = self.cache.key != Some(key);
        if geom_dirty {
            self.rebuild_raw_contours();
            self.cache.key = Some(key);
            self.cache.eval_count += 1;
        }
        if geom_dirty || self.cache.palette_pair != Some(colors) {
            self.cache.z_batch =
                build_colored_grid_batch(&self.cache.z_raw_vert, &self.cache.z_raw_horiz, colors);
            self.cache.w_batch =
                build_colored_grid_batch(&self.cache.w_raw_vert, &self.cache.w_raw_horiz, colors);
            self.cache.palette_pair = Some(colors);
        }
        true
    }

    fn rebuild_raw_contours(&mut self) {
        let span = self.grid_max - self.grid_min;
        let step = span / (self.grid_density as f64 - 1.0);
        let denom = (CONTOUR_RESOLUTION - 1) as f64;

        let mut z_vert = Vec::with_capacity(self.grid_density);
        let mut z_horiz = Vec::with_capacity(self.grid_density);
        let mut w_vert = Vec::new();
        let mut w_horiz = Vec::new();

        for i in 0..self.grid_density {
            let x = self.grid_min + (i as f64) * step;
            z_vert.push(vec![[x, self.grid_min], [x, self.grid_max]]);
            w_vert.extend(split_contour_segments(CONTOUR_RESOLUTION, span, |j| {
                let y = self.grid_min + span * (j as f64) / denom;
                self.eval_z(Complex64::new(x, y))
            }));
        }
        for i in 0..self.grid_density {
            let y = self.grid_min + (i as f64) * step;
            z_horiz.push(vec![[self.grid_min, y], [self.grid_max, y]]);
            w_horiz.extend(split_contour_segments(CONTOUR_RESOLUTION, span, |j| {
                let x = self.grid_min + span * (j as f64) / denom;
                self.eval_z(Complex64::new(x, y))
            }));
        }

        self.cache.z_raw_vert = z_vert;
        self.cache.z_raw_horiz = z_horiz;
        self.cache.w_raw_vert = w_vert;
        self.cache.w_raw_horiz = w_horiz;
    }
}
