//! Transient per-plot memoization caches stored inside [`crate::PlotMemory`].
//!
//! Uses copy-on-write (`Arc<PlotRuntimeCacheInner>`) with read-first hit checks so
//! `PlotMemory::load` and `PlotMemory::store` perform $O(1)$ reference-count clones
//! and zero heap allocations on idle frames.

use std::sync::Arc;

use egui::{Color32, Id, Vec2b};

use crate::{
    SharedPoints,
    fit::{FitResult, FitType, fit_model_with_axes},
    items::PlotItem,
    poi::{self, CriticalPoint, RefLineEntry},
    transform::{PlotBounds, PlotTransform},
};

#[derive(Clone, Debug)]
pub(crate) struct CachedItemSeries {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) points: SharedPoints,
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
    pub(crate) result: Option<Arc<FitResult>>,
    pub(crate) curve: Option<CachedFitCurve>,
}

type PoiLineEntry = (Id, String, SharedPoints);
type CollectedPoiInputs = (Vec<PoiLineEntry>, Vec<RefLineEntry>, Vec<RefLineEntry>);

#[derive(Clone, Debug)]
struct CachedPoiEntry {
    bounds_bits: [u64; 4],
    log_axis: Vec2b,
    lines: Vec<PoiLineEntry>,
    hlines: Vec<RefLineEntry>,
    vlines: Vec<RefLineEntry>,
    critical_points: Arc<[CriticalPoint]>,
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
                    && item.matches_cached_shared_points(&entry.points)
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
                    && item.matches_cached_shared_points(&entry.points)
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
                        && item.matches_cached_shared_points(&entry.points)
                })
                .map(|entry| (entry.points.clone(), entry.bounds))
                .or_else(|| {
                    prev.iter()
                        .find(|entry| {
                            entry.id == item.id()
                                && entry.name == item.name()
                                && item.matches_cached_shared_points(&entry.points)
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_fit_and_curve(
        &mut self,
        ds_pts: &SharedPoints,
        ds_name: &str,
        fit_type: FitType,
        log_x: bool,
        log_y: bool,
        fit_log_mode: bool,
        need_curve: bool,
        transform: &PlotTransform,
    ) -> (Option<Arc<FitResult>>, SharedPoints) {
        let bounds = transform.bounds();
        let (x_min, x_max) = (bounds.min[0], bounds.max[0]);
        let valid_x = x_min.is_finite() && x_max.is_finite() && x_max > x_min;
        let use_log_x = transform.log_axis().x && x_min > 0.0;
        let bounds_x_bits = [x_min.to_bits(), x_max.to_bits()];

        if let Some(cached) = &self.0.fit {
            let model_hit = cached.dataset_points.ptr_eq(ds_pts)
                && cached.dataset_name == ds_name
                && cached.fit_type == fit_type
                && cached.log_x == log_x
                && cached.log_y == log_y
                && cached.fit_log_mode == fit_log_mode;
            if model_hit {
                let Some(ref res_arc) = cached.result else {
                    #[cfg(test)]
                    self.0.fit_hits.inc();
                    return (None, SharedPoints::default());
                };
                if !need_curve || !valid_x {
                    #[cfg(test)]
                    self.0.fit_hits.inc();
                    return (Some(Arc::clone(res_arc)), SharedPoints::default());
                }
                if let Some(ref curve) = cached.curve
                    && curve.bounds_x_bits == bounds_x_bits
                    && curve.use_log_x == use_log_x
                {
                    #[cfg(test)]
                    self.0.fit_hits.inc();
                    return (Some(Arc::clone(res_arc)), curve.points.clone());
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
                return (Some(res), sampled);
            }
        }

        let computed = fit_model_with_axes(ds_pts, fit_type, ds_name, log_x, log_y, fit_log_mode)
            .map(Arc::new);
        let (curve_entry, curve_pts) = match (&computed, need_curve && valid_x) {
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
            dataset_points: ds_pts.clone(),
            dataset_name: ds_name.to_owned(),
            fit_type,
            log_x,
            log_y,
            fit_log_mode,
            result: computed.clone(),
            curve: curve_entry,
        });
        #[cfg(test)]
        inner.fit_misses.inc();
        (computed, curve_pts)
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
            && cached.bounds_bits == bounds_bits
            && cached.log_axis == log_axis
            && poi_inputs_match_cached(items, item_points, cached)
        {
            #[cfg(test)]
            self.0.poi_hits.inc();
            return Arc::clone(&cached.critical_points);
        }

        let (lines, hlines, vlines) = collect_poi_inputs(items, item_points);
        let detected = poi::detect_critical_points_from_shared(&lines, &hlines, &vlines, transform);
        let arc: Arc<[CriticalPoint]> = Arc::from(detected);
        let inner = Arc::make_mut(&mut self.0);
        inner.poi = Some(CachedPoiEntry {
            bounds_bits,
            log_axis,
            lines,
            hlines,
            vlines,
            critical_points: Arc::clone(&arc),
        });
        #[cfg(test)]
        inner.poi_misses.inc();
        arc
    }
}

#[inline]
fn bounds_to_bits(bounds: &PlotBounds) -> [u64; 4] {
    [
        bounds.min[0].to_bits(),
        bounds.min[1].to_bits(),
        bounds.max[0].to_bits(),
        bounds.max[1].to_bits(),
    ]
}

fn poi_inputs_match_cached<'a>(
    items: &[Box<dyn PlotItem + 'a>],
    item_points: &[SharedPoints],
    cached: &CachedPoiEntry,
) -> bool {
    let mut line_idx = 0_usize;
    let mut hline_idx = 0_usize;
    let mut vline_idx = 0_usize;
    for (item, pts) in items.iter().zip(item_points.iter()) {
        if !item.allow_hover() {
            continue;
        }
        if item.is_line_item() {
            if pts.len() >= 2 {
                let Some((cid, cname, cpts)) = cached.lines.get(line_idx) else {
                    return false;
                };
                if *cid != item.id() || cname != item.name() || !cpts.ptr_eq(pts) {
                    return false;
                }
                line_idx += 1;
            }
        } else if let Some(y) = item.hline_y() {
            if y.is_finite() {
                let Some(entry) = cached.hlines.get(hline_idx) else {
                    return false;
                };
                if !entry.matches(item.id(), item.name(), y) {
                    return false;
                }
                hline_idx += 1;
            }
        } else if let Some(x) = item.vline_x()
            && x.is_finite()
        {
            let Some(entry) = cached.vlines.get(vline_idx) else {
                return false;
            };
            if !entry.matches(item.id(), item.name(), x) {
                return false;
            }
            vline_idx += 1;
        }
    }
    line_idx == cached.lines.len()
        && hline_idx == cached.hlines.len()
        && vline_idx == cached.vlines.len()
}

fn collect_poi_inputs<'a>(
    items: &[Box<dyn PlotItem + 'a>],
    item_points: &[SharedPoints],
) -> CollectedPoiInputs {
    let mut lines = Vec::new();
    let mut hlines = Vec::new();
    let mut vlines = Vec::new();
    for (item, pts) in items.iter().zip(item_points.iter()) {
        if !item.allow_hover() {
            continue;
        }
        if item.is_line_item() {
            if pts.len() >= 2 {
                lines.push((item.id(), item.name().to_owned(), pts.clone()));
            }
        } else if let Some(y) = item.hline_y() {
            if y.is_finite() {
                hlines.push(RefLineEntry {
                    id: item.id(),
                    name: item.name().to_owned(),
                    value: y,
                });
            }
        } else if let Some(x) = item.vline_x()
            && x.is_finite()
        {
            vlines.push(RefLineEntry {
                id: item.id(),
                name: item.name().to_owned(),
                value: x,
            });
        }
    }
    (lines, hlines, vlines)
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
