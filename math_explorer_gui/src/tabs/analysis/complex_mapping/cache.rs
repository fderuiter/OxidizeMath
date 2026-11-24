#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ui::CONTOUR_RESOLUTION;
use super::{COMPLEX_COORD_MAX, COMPLEX_COORD_MIN, ComplexMappingTool, MappingFunction};
use egui_plot::{PlotBounds, PlotPoint};
use num_complex::Complex64;
use std::hash::{Hash as _, Hasher as _};
use std::ops::Range;
use std::sync::Arc;

#[path = "contour_batch.rs"]
mod contour_batch;
pub(super) use contour_batch::ContourBatch;

/// Maximum synchronous instruction budget (`instructions * contour_samples`) for conformal grids.
pub(crate) const MAX_CONFORMAL_SYNC_INSTRUCTIONS: usize = 1_200_000;

/// Minimum per-contour sample count preserved before scaling `grid_density`.
pub(crate) const MIN_CONTOUR_RESOLUTION: usize = 25;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ConformalCacheKey {
    function: MappingFunction,
    grid_min_bits: u64,
    grid_max_bits: u64,
    grid_density: usize,
    math_fingerprint: u64,
}

pub(super) struct ContourFamily {
    pub(super) points: Arc<Vec<PlotPoint>>,
    pub(super) ranges: Vec<Range<usize>>,
    pub(super) bounds: PlotBounds,
}

impl Default for ContourFamily {
    fn default() -> Self {
        Self {
            points: Arc::new(Vec::new()),
            ranges: Vec::new(),
            bounds: PlotBounds::NOTHING,
        }
    }
}

impl ContourFamily {
    fn clear(&mut self) {
        Arc::make_mut(&mut self.points).clear();
        self.ranges.clear();
        self.bounds = PlotBounds::NOTHING;
    }

    fn reserve(&mut self, contours: usize, point_cap: usize) {
        self.ranges.reserve(contours);
        Arc::make_mut(&mut self.points).reserve(point_cap);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn push_line_segment(&mut self, a: [f64; 2], b: [f64; 2]) {
        let pts = Arc::make_mut(&mut self.points);
        let start = pts.len();
        let pa = PlotPoint::new(a[0], a[1]);
        let pb = PlotPoint::new(b[0], b[1]);
        self.bounds.extend_with(&pa);
        self.bounds.extend_with(&pb);
        pts.push(pa);
        pts.push(pb);
        self.ranges.push(start..start + 2);
    }

    fn append_sampled_contour<F: Fn(usize) -> Complex64>(
        &mut self,
        resolution: usize,
        grid_span: f64,
        sample: F,
    ) {
        let max_radius = (grid_span.abs() * 12.0).max(25.0);
        let max_jump = (max_radius * 0.6).max(12.0);
        let pts = Arc::make_mut(&mut self.points);
        let mut seg_start = pts.len();
        let mut prev: Option<PlotPoint> = None;

        for j in 0..resolution {
            let w = sample(j);
            if !w.re.is_finite() || !w.im.is_finite() || w.norm() > max_radius {
                Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
                seg_start = pts.len();
                prev = None;
                continue;
            }
            if let Some(p) = prev
                && (w.re - p.x).hypot(w.im - p.y) > max_jump
            {
                Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
                seg_start = pts.len();
            }
            let pt = PlotPoint::new(w.re, w.im);
            pts.push(pt);
            prev = Some(pt);
        }
        Self::finish_run(pts, &mut self.ranges, &mut self.bounds, seg_start);
    }

    fn finish_run(
        pts: &mut Vec<PlotPoint>,
        ranges: &mut Vec<Range<usize>>,
        bounds: &mut PlotBounds,
        seg_start: usize,
    ) {
        let end = pts.len();
        if end - seg_start >= 2 {
            for pt in &pts[seg_start..end] {
                bounds.extend_with(pt);
            }
            ranges.push(seg_start..end);
        } else {
            pts.truncate(seg_start);
        }
    }
}

#[derive(Default)]
pub(crate) struct ConformalGridCache {
    key: Option<ConformalCacheKey>,
    pub(super) generation: u64,
    pub(super) z_vertical: ContourFamily,
    pub(super) z_horizontal: ContourFamily,
    pub(super) w_vertical: ContourFamily,
    pub(super) w_horizontal: ContourFamily,
    pub(crate) eval_count: usize,
    pub(crate) budget_notice: Option<String>,
    pub(super) last_announced_budget: Option<String>,
}

impl ConformalGridCache {
    pub(super) fn clear(&mut self) {
        if self.key.is_some()
            || !self.z_vertical.is_empty()
            || !self.z_horizontal.is_empty()
            || !self.w_vertical.is_empty()
            || !self.w_horizontal.is_empty()
        {
            self.generation = self.generation.wrapping_add(1);
        }
        self.key = None;
        self.budget_notice = None;
        self.z_vertical.clear();
        self.z_horizontal.clear();
        self.w_vertical.clear();
        self.w_horizontal.clear();
    }
}

impl ComplexMappingTool {
    pub(crate) fn contour_cache_key(&self) -> ConformalCacheKey {
        let mut mh = std::collections::hash_map::DefaultHasher::new();
        self.math_input.raw_text.hash(&mut mh);
        self.math_input.state_fingerprint().hash(&mut mh);
        ConformalCacheKey {
            function: self.function,
            grid_min_bits: self.grid_min.to_bits(),
            grid_max_bits: self.grid_max.to_bits(),
            grid_density: self.grid_density,
            math_fingerprint: mh.finish(),
        }
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.function as u8).hash(&mut h);
        self.grid_min.to_bits().hash(&mut h);
        self.grid_max.to_bits().hash(&mut h);
        self.grid_density.hash(&mut h);
        self.inspection_point[0].to_bits().hash(&mut h);
        self.inspection_point[1].to_bits().hash(&mut h);
        self.math_input.raw_text.hash(&mut h);
        self.math_input.state_fingerprint().hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn has_valid_grid_bounds(&self) -> bool {
        self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_min)
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_max)
            && self.grid_min < self.grid_max
            && (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_density)
    }

    pub(super) fn ensure_cached_contours(&mut self) -> bool {
        if !self.has_valid_grid_bounds() {
            self.cache.clear();
            return false;
        }
        let key = self.contour_cache_key();
        if self.cache.key != Some(key) {
            let mut cache = std::mem::take(&mut self.cache);
            self.populate_cache(&mut cache, key);
            cache.eval_count += 1;
            self.cache = cache;
        }
        true
    }

    pub(crate) fn effective_contour_grid(&self) -> (usize, usize, Option<String>) {
        let inst = if self.function == MappingFunction::Custom {
            self.math_input.instruction_count().max(1)
        } else {
            1
        };
        let requested_evals = 2_usize
            .saturating_mul(self.grid_density)
            .saturating_mul(CONTOUR_RESOLUTION);
        if requested_evals.saturating_mul(inst) <= MAX_CONFORMAL_SYNC_INSTRUCTIONS {
            return (self.grid_density, CONTOUR_RESOLUTION, None);
        }
        let max_evals = (MAX_CONFORMAL_SYNC_INSTRUCTIONS / inst).max(4);
        let lines = (2 * self.grid_density).max(4);
        let (eff_density, eff_res) = if max_evals / lines >= MIN_CONTOUR_RESOLUTION {
            (
                self.grid_density,
                (max_evals / lines).min(CONTOUR_RESOLUTION),
            )
        } else {
            let d = (max_evals / (2 * MIN_CONTOUR_RESOLUTION)).clamp(2, self.grid_density);
            (d, MIN_CONTOUR_RESOLUTION)
        };
        let notice = format!(
            "Contour work budget applied: sampled {eff_density}x2 lines at {eff_res} pts ({MAX_CONFORMAL_SYNC_INSTRUCTIONS} instruction cap)."
        );
        (eff_density, eff_res, Some(notice))
    }

    fn populate_cache(&self, cache: &mut ConformalGridCache, key: ConformalCacheKey) {
        let (eff_density, eff_res, notice) = self.effective_contour_grid();
        cache.budget_notice = notice;
        cache.z_vertical.clear();
        cache.z_horizontal.clear();
        cache.w_vertical.clear();
        cache.w_horizontal.clear();
        cache.z_vertical.reserve(eff_density, eff_density * 2);
        cache.z_horizontal.reserve(eff_density, eff_density * 2);
        cache.w_vertical.reserve(eff_density, eff_density * eff_res);
        cache
            .w_horizontal
            .reserve(eff_density, eff_density * eff_res);

        let span = self.grid_max - self.grid_min;
        let step = span / (eff_density as f64 - 1.0);
        let denom = (eff_res - 1) as f64;

        for i in 0..eff_density {
            let x = self.grid_min + (i as f64) * step;
            cache
                .z_vertical
                .push_line_segment([x, self.grid_min], [x, self.grid_max]);
            cache.w_vertical.append_sampled_contour(eff_res, span, |j| {
                self.eval_z(Complex64::new(x, self.grid_min + span * (j as f64) / denom))
            });
        }
        for i in 0..eff_density {
            let y = self.grid_min + (i as f64) * step;
            cache
                .z_horizontal
                .push_line_segment([self.grid_min, y], [self.grid_max, y]);
            cache
                .w_horizontal
                .append_sampled_contour(eff_res, span, |j| {
                    self.eval_z(Complex64::new(self.grid_min + span * (j as f64) / denom, y))
                });
        }
        cache.key = Some(key);
        cache.generation = cache.generation.wrapping_add(1).max(1);
    }

    pub(super) fn resolve_export_cache<'a>(
        &'a self,
        cold: &'a mut ConformalGridCache,
    ) -> Option<&'a ConformalGridCache> {
        if !self.has_valid_grid_bounds() {
            return None;
        }
        let key = self.contour_cache_key();
        if self.cache.key == Some(key) {
            Some(&self.cache)
        } else {
            self.populate_cache(cold, key);
            Some(cold)
        }
    }
}
