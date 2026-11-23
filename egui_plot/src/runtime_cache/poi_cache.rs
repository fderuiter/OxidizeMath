//! Critical-point memoization helpers for [`super::PlotRuntimeCache`].

use std::sync::Arc;

use egui::{Id, Vec2b};

use crate::{
    SharedPoints,
    items::PlotItem,
    poi::{self, CriticalPoint, RefLineEntry},
    transform::{PlotBounds, PlotTransform},
};

pub(super) type PoiLineEntry = (Id, String, SharedPoints);
type CollectedPoiInputs = (Vec<PoiLineEntry>, Vec<RefLineEntry>, Vec<RefLineEntry>);

#[derive(Clone, Debug)]
pub(super) struct CachedPoiEntry {
    bounds_bits: [u64; 4],
    log_axis: Vec2b,
    lines: Vec<PoiLineEntry>,
    hlines: Vec<RefLineEntry>,
    vlines: Vec<RefLineEntry>,
    pub(super) critical_points: Arc<[CriticalPoint]>,
}

impl CachedPoiEntry {
    pub(super) fn matches<'a>(
        &self,
        bounds_bits: [u64; 4],
        log_axis: Vec2b,
        items: &[Box<dyn PlotItem + 'a>],
        item_points: &[SharedPoints],
    ) -> bool {
        self.bounds_bits == bounds_bits
            && self.log_axis == log_axis
            && poi_inputs_match_cached(items, item_points, self)
    }

    pub(super) fn build<'a>(
        bounds_bits: [u64; 4],
        log_axis: Vec2b,
        items: &[Box<dyn PlotItem + 'a>],
        item_points: &[SharedPoints],
        transform: &PlotTransform,
    ) -> Self {
        let (lines, hlines, vlines) = collect_poi_inputs(items, item_points);
        let detected = poi::detect_critical_points_from_shared(&lines, &hlines, &vlines, transform);
        Self {
            bounds_bits,
            log_axis,
            lines,
            hlines,
            vlines,
            critical_points: Arc::from(detected),
        }
    }
}

#[inline]
pub(super) fn bounds_to_bits(bounds: &PlotBounds) -> [u64; 4] {
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
