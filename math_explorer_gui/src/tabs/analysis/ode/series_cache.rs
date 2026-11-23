#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::SharedPoints;

type SeriesStamp = (usize, usize, u64, u64);

fn stamp_slice(values: &[f64]) -> SeriesStamp {
    let first = values.first().map_or(0, |v| v.to_bits());
    let last = values.last().map_or(0, |v| v.to_bits());
    (values.as_ptr() as usize, values.len(), first, last)
}

fn compute_harmonic_root_from_pairs(
    y_points: &[[f64; 2]],
    series_finite: bool,
    total_time: f64,
    ic_y0: f64,
    ic_v0: f64,
) -> Option<f64> {
    if !total_time.is_finite() {
        return None;
    }
    let crossings = if series_finite {
        egui_plot::poi::interpolate_threshold_crossings_pairs(y_points, 0.0)
    } else {
        let filtered: Vec<[f64; 2]> = y_points
            .iter()
            .copied()
            .filter(|[t, y]| t.is_finite() && y.is_finite())
            .collect();
        egui_plot::poi::interpolate_threshold_crossings_pairs(&filtered, 0.0)
    };
    let mid = 0.5 * total_time;
    crossings
        .into_iter()
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= total_time)
        .min_by(|a, b| (a - mid).abs().total_cmp(&(b - mid).abs()))
        .or_else(|| (ic_y0.abs() <= 1e-12 && ic_v0.abs() <= 1e-12).then_some(0.0))
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OdeSeriesCache {
    y_points: SharedPoints,
    v_points: SharedPoints,
    series_finite: bool,
    harmonic_root: Option<f64>,
    t_stamp: SeriesStamp,
    y_stamp: SeriesStamp,
    v_stamp: SeriesStamp,
    meta_bits: (u64, u64, u64),
    initialized: bool,
}

impl OdeSeriesCache {
    pub(crate) fn rebuild(
        &mut self,
        time_series: &[f64],
        y_series: &[f64],
        v_series: &[f64],
        total_time: f64,
        ic_y0: f64,
        ic_v0: f64,
    ) {
        let y_len = time_series.len().min(y_series.len());
        let v_len = time_series.len().min(v_series.len());
        let mut py = Vec::with_capacity(y_len);
        for i in 0..y_len {
            py.push([time_series[i], y_series[i]]);
        }
        let mut pv = Vec::with_capacity(v_len);
        for i in 0..v_len {
            pv.push([time_series[i], v_series[i]]);
        }
        let series_finite = time_series.iter().all(|t| t.is_finite())
            && y_series.iter().all(|y| y.is_finite())
            && v_series.iter().all(|v| v.is_finite());
        let harmonic_root =
            compute_harmonic_root_from_pairs(&py, series_finite, total_time, ic_y0, ic_v0);
        self.y_points = SharedPoints::from(py);
        self.v_points = SharedPoints::from(pv);
        self.series_finite = series_finite;
        self.harmonic_root = harmonic_root;
        self.t_stamp = stamp_slice(time_series);
        self.y_stamp = stamp_slice(y_series);
        self.v_stamp = stamp_slice(v_series);
        self.meta_bits = (total_time.to_bits(), ic_y0.to_bits(), ic_v0.to_bits());
        self.initialized = true;
    }

    pub(crate) fn matches_series(
        &self,
        time_series: &[f64],
        y_series: &[f64],
        v_series: &[f64],
    ) -> bool {
        self.initialized
            && self.t_stamp == stamp_slice(time_series)
            && self.y_stamp == stamp_slice(y_series)
            && self.v_stamp == stamp_slice(v_series)
    }

    pub(crate) fn matches_all(
        &self,
        time_series: &[f64],
        y_series: &[f64],
        v_series: &[f64],
        total_time: f64,
        ic_y0: f64,
        ic_v0: f64,
    ) -> bool {
        self.matches_series(time_series, y_series, v_series)
            && self.meta_bits == (total_time.to_bits(), ic_y0.to_bits(), ic_v0.to_bits())
    }

    pub(crate) fn shared_points(&self) -> (SharedPoints, SharedPoints) {
        (self.y_points.clone(), self.v_points.clone())
    }

    pub(crate) fn is_series_finite(&self) -> bool {
        self.series_finite
    }

    pub(crate) fn harmonic_root(&self) -> Option<f64> {
        self.harmonic_root
    }
}

impl super::OdeSolverTool {
    pub(crate) fn sync_series_cache(&mut self) {
        let series_matched =
            self.series_cache
                .matches_series(&self.time_series, &self.y_series, &self.v_series);
        if !self.series_cache.matches_all(
            &self.time_series,
            &self.y_series,
            &self.v_series,
            self.total_time,
            self.ic_y0,
            self.ic_v0,
        ) {
            self.series_cache.rebuild(
                &self.time_series,
                &self.y_series,
                &self.v_series,
                self.total_time,
                self.ic_y0,
                self.ic_v0,
            );
            if !series_matched {
                let cap = crate::framework::MAX_DISPLAY_POLYLINE_POINTS;
                self.display_y = crate::framework::decimate_xy_series_minmax(
                    &self.time_series,
                    &self.y_series,
                    cap,
                );
                self.display_v = crate::framework::decimate_xy_series_minmax(
                    &self.time_series,
                    &self.v_series,
                    cap,
                );
            }
        }
    }
}
