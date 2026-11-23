//! Transient per-plot memoization caches stored inside [`crate::PlotMemory`].
//!
//! Uses copy-on-write (`Arc<PlotRuntimeCacheInner>`) with read-first hit checks so
//! `PlotMemory::load` and `PlotMemory::store` perform $O(1)$ reference-count clones
//! and zero heap allocations on idle frames.

#[path = "runtime_cache/poi_cache.rs"]
mod poi_cache;

use std::sync::Arc;

use egui::{Color32, Id};

use self::poi_cache::{CachedPoiEntry, bounds_to_bits};
use crate::{
    SharedPoints,
    fit::{FitResult, FitType, fit_model_with_axes},
    fit_overlay,
    items::PlotItem,
    poi::CriticalPoint,
    transform::{PlotBounds, PlotTransform},
};

#[derive(Clone, Debug)]
pub(crate) struct CachedItemSeries {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) points: SharedPoints,
    pub(crate) source_vec_ptr: Option<usize>,
    pub(crate) color: Color32,
    pub(crate) bounds: PlotBounds,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedFitCurve {
    pub(crate) bounds_x_bits: [u64; 2],
    pub(crate) use_log_x: bool,
    pub(crate) points: SharedPoints,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedFitEntry {
    pub(crate) dataset_points: SharedPoints,
    pub(crate) dataset_name: String,
    pub(crate) fit_type: FitType,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) fit_log_mode: bool,
    pub(crate) custom_formula: String,
    pub(crate) custom_seeds_text: String,
    pub(crate) custom_callback_ptr: usize,
    pub(crate) custom_error: Option<String>,
    pub(crate) result: Option<Arc<FitResult>>,
    pub(crate) curve: Option<CachedFitCurve>,
}

pub(crate) struct FitRequest<'a> {
    pub(crate) ds_pts: &'a SharedPoints,
    pub(crate) ds_name: &'a str,
    pub(crate) fit_type: FitType,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) fit_log_mode: bool,
    pub(crate) custom_formula: &'a str,
    pub(crate) custom_seeds_text: &'a str,
    pub(crate) custom_callback_ptr: usize,
    pub(crate) need_curve: bool,
    pub(crate) transform: &'a PlotTransform,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestCounter(std::sync::atomic::AtomicU64);

#[cfg(test)]
impl Clone for TestCounter {
    fn clone(&self) -> Self {
        Self(std::sync::atomic::AtomicU64::new(
            self.0.load(std::sync::atomic::Ordering::Relaxed),
        ))
    }
}

#[cfg(test)]
impl TestCounter {
    #[inline]
    fn inc(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Clone, Debug, Default)]
struct PlotRuntimeCacheInner {
    series: Vec<CachedItemSeries>,
    fit: Option<CachedFitEntry>,
    poi: Option<CachedPoiEntry>,
    #[cfg(test)]
    series_hits: TestCounter,
    #[cfg(test)]
    series_misses: TestCounter,
    #[cfg(test)]
    fit_hits: TestCounter,
    #[cfg(test)]
    fit_misses: TestCounter,
    #[cfg(test)]
    poi_hits: TestCounter,
    #[cfg(test)]
    poi_misses: TestCounter,
}

/// Non-serialized runtime memoization cache attached to [`crate::PlotMemory`].
#[derive(Clone, Debug, Default)]
pub(crate) struct PlotRuntimeCache(Arc<PlotRuntimeCacheInner>);

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PlotCacheCounters {
    pub(crate) series_hits: u64,
    pub(crate) series_misses: u64,
    pub(crate) fit_hits: u64,
    pub(crate) fit_misses: u64,
    pub(crate) poi_hits: u64,
    pub(crate) poi_misses: u64,
}

impl PlotRuntimeCache {
    #[cfg(test)]
    pub(crate) fn counters(&self) -> PlotCacheCounters {
        PlotCacheCounters {
            series_hits: self.0.series_hits.get(),
            series_misses: self.0.series_misses.get(),
            fit_hits: self.0.fit_hits.get(),
            fit_misses: self.0.fit_misses.get(),
            poi_hits: self.0.poi_hits.get(),
            poi_misses: self.0.poi_misses.get(),
        }
    }

    /// Returns cached [`CachedItemSeries`] slice if all `items` match without mutating counters.
    pub(crate) fn peek_matching_series<'a>(
        &self,
        items: &[Box<dyn PlotItem + 'a>],
    ) -> Option<&[CachedItemSeries]> {
        let matches = self.0.series.len() == items.len()
            && self.0.series.iter().zip(items.iter()).all(|(entry, item)| {
                entry.id == item.id()
                    && entry.name == item.name()
                    && item.matches_cached_series(&entry.points, entry.source_vec_ptr)
            });
        matches.then_some(self.0.series.as_slice())
    }

    /// Extract or reuse [`CachedItemSeries`] handles for `items`.
    pub(crate) fn resolve_base_series<'a>(
        &mut self,
        items: &[Box<dyn PlotItem + 'a>],
    ) -> &[CachedItemSeries] {
        let all_match = self.0.series.len() == items.len()
            && self.0.series.iter().zip(items.iter()).all(|(entry, item)| {
                entry.id == item.id()
                    && entry.name == item.name()
                    && entry.color == item.color()
                    && item.matches_cached_series(&entry.points, entry.source_vec_ptr)
            });
        if all_match {
            #[cfg(test)]
            self.0.series_hits.inc();
            return &self.0.series;
        }

        let prev = &self.0.series;
        let mut rebuilt = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let reused = prev
                .get(idx)
                .filter(|entry| {
                    entry.id == item.id()
                        && entry.name == item.name()
                        && item.matches_cached_series(&entry.points, entry.source_vec_ptr)
                })
                .map(|entry| (entry.points.clone(), entry.bounds))
                .or_else(|| {
                    prev.iter()
                        .find(|entry| {
                            entry.id == item.id()
                                && entry.name == item.name()
                                && item.matches_cached_series(&entry.points, entry.source_vec_ptr)
                        })
                        .map(|entry| (entry.points.clone(), entry.bounds))
                });
            let (points, bounds) = match reused {
                Some((pts, b)) if item.is_line_item() => (pts, b),
                Some((pts, _)) => (pts, item.bounds()),
                None => (item.extract_shared_points(), item.bounds()),
            };
            rebuilt.push(CachedItemSeries {
                id: item.id(),
                name: item.name().to_owned(),
                points,
                source_vec_ptr: item.shared_vec_ptr(),
                color: item.color(),
                bounds,
            });
        }
        let inner = Arc::make_mut(&mut self.0);
        inner.series = rebuilt;
        #[cfg(test)]
        inner.series_misses.inc();
        &self.0.series
    }

    pub(crate) fn clear_fit(&mut self) {
        if self.0.fit.is_some() {
            Arc::make_mut(&mut self.0).fit = None;
        }
    }

    pub(crate) fn resolve_fit_and_curve(
        &mut self,
        req: &FitRequest<'_>,
    ) -> (Option<Arc<FitResult>>, SharedPoints, Option<String>) {
        let bounds = req.transform.bounds();
        let (x_min, x_max) = (bounds.min[0], bounds.max[0]);
        let valid_x = x_min.is_finite() && x_max.is_finite() && x_max > x_min;
        let use_log_x = req.transform.log_axis().x && x_min > 0.0;
        let bounds_x_bits = [x_min.to_bits(), x_max.to_bits()];

        if let Some(hit) =
            self.try_reuse_cached_fit(req, x_min, x_max, valid_x, use_log_x, bounds_x_bits)
        {
            return hit;
        }

        let (computed, custom_error) = compute_requested_fit(req);
        let (curve_entry, curve_pts) = match (&computed, req.need_curve && valid_x) {
            (Some(res), true) => {
                let pts = sample_fit_curve_shared(res, x_min, x_max, use_log_x);
                (
                    Some(CachedFitCurve {
                        bounds_x_bits,
                        use_log_x,
                        points: pts.clone(),
                    }),
                    pts,
                )
            }
            _ => (None, SharedPoints::default()),
        };
        let inner = Arc::make_mut(&mut self.0);
        inner.fit = Some(CachedFitEntry {
            dataset_points: req.ds_pts.clone(),
            dataset_name: req.ds_name.to_owned(),
            fit_type: req.fit_type,
            log_x: req.log_x,
            log_y: req.log_y,
            fit_log_mode: req.fit_log_mode,
            custom_formula: if req.fit_type == FitType::Custom {
                req.custom_formula.to_owned()
            } else {
                String::new()
            },
            custom_seeds_text: if req.fit_type == FitType::Custom {
                req.custom_seeds_text.to_owned()
            } else {
                String::new()
            },
            custom_callback_ptr: req.custom_callback_ptr,
            custom_error: custom_error.clone(),
            result: computed.clone(),
            curve: curve_entry,
        });
        #[cfg(test)]
        inner.fit_misses.inc();
        (computed, curve_pts, custom_error)
    }

    fn try_reuse_cached_fit(
        &mut self,
        req: &FitRequest<'_>,
        x_min: f64,
        x_max: f64,
        valid_x: bool,
        use_log_x: bool,
        bounds_x_bits: [u64; 2],
    ) -> Option<(Option<Arc<FitResult>>, SharedPoints, Option<String>)> {
        let cached = self.0.fit.as_ref()?;
        if !fit_entry_matches(cached, req) {
            return None;
        }
        let err = cached.custom_error.clone();
        let Some(ref res_arc) = cached.result else {
            #[cfg(test)]
            self.0.fit_hits.inc();
            return Some((None, SharedPoints::default(), err));
        };
        if !req.need_curve || !valid_x {
            #[cfg(test)]
            self.0.fit_hits.inc();
            return Some((Some(Arc::clone(res_arc)), SharedPoints::default(), err));
        }
        if let Some(ref curve) = cached.curve
            && curve.bounds_x_bits == bounds_x_bits
            && curve.use_log_x == use_log_x
        {
            #[cfg(test)]
            self.0.fit_hits.inc();
            return Some((Some(Arc::clone(res_arc)), curve.points.clone(), err));
        }
        let res = Arc::clone(res_arc);
        let sampled = sample_fit_curve_shared(&res, x_min, x_max, use_log_x);
        let inner = Arc::make_mut(&mut self.0);
        if let Some(ref mut entry) = inner.fit {
            entry.curve = Some(CachedFitCurve {
                bounds_x_bits,
                use_log_x,
                points: sampled.clone(),
            });
        }
        #[cfg(test)]
        inner.fit_hits.inc();
        Some((Some(res), sampled, err))
    }

    /// Resolve memoized [`CriticalPoint`] slice for `items` and their extracted `item_points`.
    pub(crate) fn resolve_critical_points<'a>(
        &mut self,
        items: &[Box<dyn PlotItem + 'a>],
        item_points: &[SharedPoints],
        transform: &PlotTransform,
    ) -> Arc<[CriticalPoint]> {
        let bounds = *transform.bounds();
        if !bounds.is_finite() {
            return Arc::from([]);
        }
        let bounds_bits = bounds_to_bits(&bounds);
        let log_axis = transform.log_axis();

        if let Some(cached) = &self.0.poi
            && cached.matches(bounds_bits, log_axis, items, item_points)
        {
            #[cfg(test)]
            self.0.poi_hits.inc();
            return Arc::clone(&cached.critical_points);
        }

        let built = CachedPoiEntry::build(bounds_bits, log_axis, items, item_points, transform);
        let arc = Arc::clone(&built.critical_points);
        let inner = Arc::make_mut(&mut self.0);
        inner.poi = Some(built);
        #[cfg(test)]
        inner.poi_misses.inc();
        arc
    }
}

#[inline]
fn fit_entry_matches(cached: &CachedFitEntry, req: &FitRequest<'_>) -> bool {
    let base_ok = cached.dataset_points.ptr_eq(req.ds_pts)
        && cached.dataset_name == req.ds_name
        && cached.fit_type == req.fit_type
        && cached.log_x == req.log_x
        && cached.log_y == req.log_y
        && cached.fit_log_mode == req.fit_log_mode;
    if !base_ok {
        return false;
    }
    if req.fit_type == FitType::Custom {
        cached.custom_formula == req.custom_formula
            && cached.custom_seeds_text == req.custom_seeds_text
            && cached.custom_callback_ptr == req.custom_callback_ptr
    } else {
        true
    }
}

fn compute_requested_fit(req: &FitRequest<'_>) -> (Option<Arc<FitResult>>, Option<String>) {
    if req.fit_type == FitType::Custom {
        match fit_overlay::solve_custom_fit(
            req.ds_pts,
            req.ds_name,
            req.custom_formula,
            req.custom_seeds_text,
        ) {
            Ok(res) => (Some(Arc::new(res)), None),
            Err(err) => (None, Some(err)),
        }
    } else {
        let solved = fit_model_with_axes(
            req.ds_pts,
            req.fit_type,
            req.ds_name,
            req.log_x,
            req.log_y,
            req.fit_log_mode,
        )
        .map(Arc::new);
        (solved, None)
    }
}

fn sample_fit_curve_shared(
    fit_res: &FitResult,
    x_min: f64,
    x_max: f64,
    use_log_x: bool,
) -> SharedPoints {
    let count = 200_usize;
    let (lo_log, hi_log) = if use_log_x {
        (x_min.log10(), x_max.log10())
    } else {
        (0.0, 0.0)
    };
    let mut out = Vec::with_capacity(count + 1);
    for i in 0..=count {
        let t = i as f64 / (count as f64);
        let x = if use_log_x {
            10.0_f64.powf(lo_log + t * (hi_log - lo_log))
        } else {
            x_min + t * (x_max - x_min)
        };
        let y = fit_res.eval(x);
        if x.is_finite() && y.is_finite() {
            out.push([x, y]);
        }
    }
    SharedPoints::from(out)
}
