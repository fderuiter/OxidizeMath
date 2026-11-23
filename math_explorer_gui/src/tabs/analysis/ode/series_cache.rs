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
        egui_plot::poi::interpolate_threshold_crossings(y_points, 0.0)
    } else {
        let filtered: Vec<[f64; 2]> = y_points
            .iter()
            .copied()
            .filter(|[t, y]| t.is_finite() && y.is_finite())
            .collect();
        egui_plot::poi::interpolate_threshold_crossings(&filtered, 0.0)
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
                self.display_phase = crate::framework::decimate_xy_series_minmax(
                    &self.y_series,
                    &self.v_series,
                    cap,
                );
            }
        }
    }

    pub(crate) fn collect_full_xy_pairs(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if self
            .series_cache
            .matches_series(&self.time_series, &self.y_series, &self.v_series)
        {
            let (py, pv) = self.series_cache.shared_points();
            return (py.to_vec(), pv.to_vec());
        }
        let mut plot_points_y = Vec::with_capacity(self.time_series.len());
        let mut plot_points_v = if self.v_series.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.v_series.len())
        };
        for (i, &t) in self.time_series.iter().enumerate() {
            if let Some(&y) = self.y_series.get(i) {
                plot_points_y.push([t, y]);
            }
            if let Some(&v) = self.v_series.get(i) {
                plot_points_v.push([t, v]);
            }
        }
        (plot_points_y, plot_points_v)
    }

    pub(crate) fn build_current_trace_series(&self) -> Vec<crate::framework::TraceSeries> {
        let (py, pv) =
            if self
                .series_cache
                .matches_series(&self.time_series, &self.y_series, &self.v_series)
            {
                self.series_cache.shared_points()
            } else {
                let (y, v) = self.collect_full_xy_pairs();
                (y.into(), v.into())
            };
        let mut series = vec![crate::framework::TraceSeries::new("y(t)", py)];
        if self.has_second_state() && !pv.is_empty() {
            series.push(crate::framework::TraceSeries::new(
                self.second_state_label(),
                pv,
            ));
        }
        series
    }

    pub(crate) fn build_full_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        if self.comparison_mode && !self.comparison_manager.is_empty() {
            datasets.extend(self.comparison_manager.export_datasets());
        } else {
            let (plot_points_y, plot_points_v) = self.collect_full_xy_pairs();
            datasets.push(("y(t)".to_string(), plot_points_y));
            if self.has_second_state() && !plot_points_v.is_empty() {
                datasets.push((self.second_state_label().to_string(), plot_points_v));
            }
            if self.is_2d_phase_portrait() && !self.v_series.is_empty() {
                let phase_pts: Vec<[f64; 2]> = self
                    .y_series
                    .iter()
                    .copied()
                    .zip(self.v_series.iter().copied())
                    .map(|(y, v)| [y, v])
                    .collect();
                datasets.push(("Phase Orbit (y, v)".to_string(), phase_pts));
            }
        }
        datasets.extend(self.snapshot_manager.export_datasets());
        for (name, pts) in egui_plot::get_last_plot_datasets() {
            if !datasets.iter().any(|(existing, _)| existing == &name) {
                datasets.push((name, pts));
            }
        }
        datasets
    }
}
