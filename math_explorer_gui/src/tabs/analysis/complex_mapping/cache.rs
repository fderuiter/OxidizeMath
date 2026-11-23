#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub(super) use super::contour_batch::ContourBatch;
use super::ui::CONTOUR_RESOLUTION;
use super::{COMPLEX_COORD_MAX, COMPLEX_COORD_MIN, ComplexMappingTool, MappingFunction};
use egui_plot::{PlotBounds, PlotPoint};
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;
use std::hash::{Hash as _, Hasher as _};
use std::ops::Range;
use std::sync::Arc;

fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ConformalCacheKey {
    function: MappingFunction,
    grid_min_bits: u64,
    grid_max_bits: u64,
    grid_density: usize,
    math_fingerprint: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ConformalProbeKey {
    function: MappingFunction,
    z_re_bits: u64,
    z_im_bits: u64,
    math_fingerprint: u64,
}

pub(super) struct ContourFamily {
    pub(super) points: Arc<Vec<PlotPoint>>,
    pub(super) ranges: Arc<Vec<Range<usize>>>,
    pub(super) bounds: PlotBounds,
}

impl Default for ContourFamily {
    fn default() -> Self {
        Self {
            points: Arc::new(Vec::new()),
            ranges: Arc::new(Vec::new()),
            bounds: PlotBounds::NOTHING,
        }
    }
}

impl ContourFamily {
    fn clear(&mut self) {
        Arc::make_mut(&mut self.points).clear();
        Arc::make_mut(&mut self.ranges).clear();
        self.bounds = PlotBounds::NOTHING;
    }

    fn reserve(&mut self, contours: usize, point_cap: usize) {
        Arc::make_mut(&mut self.ranges).reserve(contours);
        Arc::make_mut(&mut self.points).reserve(point_cap);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn push_line_segment(&mut self, a: [f64; 2], b: [f64; 2]) {
        let pts = Arc::make_mut(&mut self.points);
        let ranges = Arc::make_mut(&mut self.ranges);
        let start = pts.len();
        let pa = PlotPoint::new(a[0], a[1]);
        let pb = PlotPoint::new(b[0], b[1]);
        self.bounds.extend_with(&pa);
        self.bounds.extend_with(&pb);
        pts.push(pa);
        pts.push(pb);
        ranges.push(start..start + 2);
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
        let ranges = Arc::make_mut(&mut self.ranges);
        let mut seg_start = pts.len();
        let mut prev: Option<PlotPoint> = None;

        for j in 0..resolution {
            let w = sample(j);
            if !w.re.is_finite() || !w.im.is_finite() || w.norm() > max_radius {
                Self::finish_run(pts, ranges, &mut self.bounds, seg_start);
                seg_start = pts.len();
                prev = None;
                continue;
            }
            if let Some(p) = prev {
                if (w.re - p.x).hypot(w.im - p.y) > max_jump {
                    Self::finish_run(pts, ranges, &mut self.bounds, seg_start);
                    seg_start = pts.len();
                }
            }
            let pt = PlotPoint::new(w.re, w.im);
            pts.push(pt);
            prev = Some(pt);
        }
        Self::finish_run(pts, ranges, &mut self.bounds, seg_start);
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

pub(crate) struct ConformalGridCache {
    key: Option<ConformalCacheKey>,
    pub(super) generation: u64,
    pub(super) z_vertical: ContourFamily,
    pub(super) z_horizontal: ContourFamily,
    pub(super) w_vertical: ContourFamily,
    pub(super) w_horizontal: ContourFamily,
    probe_key: Option<ConformalProbeKey>,
    pub(super) cached_w0: Complex64,
    pub(super) cached_scale: f64,
    pub(crate) eval_count: usize,
    pub(crate) probe_eval_count: usize,
}

impl Default for ConformalGridCache {
    fn default() -> Self {
        Self {
            key: None,
            generation: 0,
            z_vertical: ContourFamily::default(),
            z_horizontal: ContourFamily::default(),
            w_vertical: ContourFamily::default(),
            w_horizontal: ContourFamily::default(),
            probe_key: None,
            cached_w0: Complex64::new(0.0, 0.0),
            cached_scale: 0.0,
            eval_count: 0,
            probe_eval_count: 0,
        }
    }
}

impl ConformalGridCache {
    pub(super) fn clear_contours(&mut self) {
        if self.key.is_some()
            || !self.z_vertical.is_empty()
            || !self.z_horizontal.is_empty()
            || !self.w_vertical.is_empty()
            || !self.w_horizontal.is_empty()
        {
            self.generation = self.generation.wrapping_add(1);
        }
        self.key = None;
        self.z_vertical.clear();
        self.z_horizontal.clear();
        self.w_vertical.clear();
        self.w_horizontal.clear();
    }
}

impl ComplexMappingTool {
    fn math_input_fingerprint(&self) -> u64 {
        let mut mh = std::collections::hash_map::DefaultHasher::new();
        self.math_input.raw_text.hash(&mut mh);
        self.math_input.is_valid().hash(&mut mh);
        self.math_input.state_fingerprint().hash(&mut mh);
        mh.finish()
    }

    pub(crate) fn contour_cache_key(&self) -> ConformalCacheKey {
        ConformalCacheKey {
            function: self.function,
            grid_min_bits: canonical_bits(self.grid_min),
            grid_max_bits: canonical_bits(self.grid_max),
            grid_density: self.grid_density,
            math_fingerprint: self.math_input_fingerprint(),
        }
    }

    fn probe_cache_key(&self) -> ConformalProbeKey {
        ConformalProbeKey {
            function: self.function,
            z_re_bits: canonical_bits(self.inspection_point[0]),
            z_im_bits: canonical_bits(self.inspection_point[1]),
            math_fingerprint: self.math_input_fingerprint(),
        }
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.function as u8).hash(&mut h);
        canonical_bits(self.grid_min).hash(&mut h);
        canonical_bits(self.grid_max).hash(&mut h);
        self.grid_density.hash(&mut h);
        canonical_bits(self.inspection_point[0]).hash(&mut h);
        canonical_bits(self.inspection_point[1]).hash(&mut h);
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
            && (2..=crate::presets::MAX_INTERACTIVE_CONFORMAL_GRID_SIZE)
                .contains(&self.grid_density)
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

    pub(super) fn ensure_cached_contours(&mut self) -> bool {
        self.ensure_cached_probe();
        if !self.has_valid_grid_bounds() {
            self.cache.clear_contours();
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

    fn populate_cache(&self, cache: &mut ConformalGridCache, key: ConformalCacheKey) {
        cache.z_vertical.clear();
        cache.z_horizontal.clear();
        cache.w_vertical.clear();
        cache.w_horizontal.clear();
        cache
            .z_vertical
            .reserve(self.grid_density, self.grid_density * 2);
        cache
            .z_horizontal
            .reserve(self.grid_density, self.grid_density * 2);
        cache
            .w_vertical
            .reserve(self.grid_density, self.grid_density * CONTOUR_RESOLUTION);
        cache
            .w_horizontal
            .reserve(self.grid_density, self.grid_density * CONTOUR_RESOLUTION);

        let span = self.grid_max - self.grid_min;
        let step = span / (self.grid_density as f64 - 1.0);
        let denom = (CONTOUR_RESOLUTION - 1) as f64;

        for i in 0..self.grid_density {
            let x = self.grid_min + (i as f64) * step;
            cache
                .z_vertical
                .push_line_segment([x, self.grid_min], [x, self.grid_max]);
            cache
                .w_vertical
                .append_sampled_contour(CONTOUR_RESOLUTION, span, |j| {
                    self.eval_z(Complex64::new(x, self.grid_min + span * (j as f64) / denom))
                });
        }
        for i in 0..self.grid_density {
            let y = self.grid_min + (i as f64) * step;
            cache
                .z_horizontal
                .push_line_segment([self.grid_min, y], [self.grid_max, y]);
            cache
                .w_horizontal
                .append_sampled_contour(CONTOUR_RESOLUTION, span, |j| {
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
