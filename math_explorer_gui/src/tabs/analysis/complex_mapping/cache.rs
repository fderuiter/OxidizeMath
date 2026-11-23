#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ui::{split_contour_segments, CONTOUR_RESOLUTION};
use super::{ComplexMappingTool, MappingFunction};
use egui_plot::{PlotBounds, PlotPoint};
use num_complex::Complex64;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

type SharedBatch = (Arc<Vec<Vec<PlotPoint>>>, PlotBounds);

pub(crate) struct ContourGridCache {
    key: Option<u64>,
    pub(super) z_vertical: SharedBatch,
    pub(super) z_horizontal: SharedBatch,
    pub(super) w_vertical: SharedBatch,
    pub(super) w_horizontal: SharedBatch,
    pub(crate) eval_count: usize,
}

impl Default for ContourGridCache {
    fn default() -> Self {
        let empty = || (Arc::new(Vec::new()), PlotBounds::NOTHING);
        Self {
            key: None,
            z_vertical: empty(),
            z_horizontal: empty(),
            w_vertical: empty(),
            w_horizontal: empty(),
            eval_count: 0,
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

fn to_shared_batch(raw_segments: Vec<Vec<[f64; 2]>>) -> SharedBatch {
    let mut bounds = PlotBounds::NOTHING;
    let mut converted = Vec::with_capacity(raw_segments.len());
    for seg in raw_segments {
        let mut pts = Vec::with_capacity(seg.len());
        for [x, y] in seg {
            let pt = PlotPoint::new(x, y);
            bounds.extend_with(&pt);
            pts.push(pt);
        }
        if pts.len() >= 2 {
            converted.push(pts);
        }
    }
    (Arc::new(converted), bounds)
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

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.contour_geometry_key().hash(&mut h);
        canonical_bits(self.inspection_point[0]).hash(&mut h);
        canonical_bits(self.inspection_point[1]).hash(&mut h);
        self.math_input.raw_text.hash(&mut h);
        self.domain_coloring.hash(&mut h);
        self.palette.hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn ensure_cached_contours(&mut self) -> bool {
        let grid_ok = self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && self.grid_max > self.grid_min
            && (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_density);
        if !grid_ok {
            return false;
        }
        let key = self.contour_geometry_key();
        if self.cache.key != Some(key) {
            self.rebuild_contour_batches();
            self.cache.key = Some(key);
            self.cache.eval_count += 1;
        }
        true
    }

    fn rebuild_contour_batches(&mut self) {
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

        self.cache.z_vertical = to_shared_batch(z_vert);
        self.cache.z_horizontal = to_shared_batch(z_horiz);
        self.cache.w_vertical = to_shared_batch(w_vert);
        self.cache.w_horizontal = to_shared_batch(w_horiz);
    }
}
