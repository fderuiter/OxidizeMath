//! CSV series calibration (`estimate_ou_params`) and zero-allocation `EulerMaruyama` fan bands.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    OuBandRow, OuProcessState, OuProcessTool, MAX_IMPORTED_CSV_POINTS, MAX_OU_PATHS, MAX_OU_STEPS,
};
use crate::web_io::strip_utf8_bom;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::ou_process::{
    estimate_ou_params, EulerMaruyama, OuAnalyzer, OuParams, TimeStep,
};
use oxidize_core::rng::OxidizeRng;

fn parse_csv_line_pair(line: &str) -> Result<Option<(Option<f64>, f64)>, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }
    let cols: Vec<&str> = trimmed.split([',', ';', '\t']).map(str::trim).collect();
    if cols.len() == 1 {
        return match cols[0].parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(Some((None, v))),
            Ok(_) => Err("CSV contains non-finite numeric value".into()),
            Err(_) => Ok(None),
        };
    }
    let first = cols[0].parse::<f64>();
    let second = cols[1].parse::<f64>();
    match (first, second) {
        (Ok(t), Ok(v)) if t.is_finite() && v.is_finite() => Ok(Some((Some(t), v))),
        (Ok(_), Ok(_)) => Err("CSV contains non-finite numeric value".into()),
        _ => Ok(None),
    }
}

impl OuProcessTool {
    pub(crate) fn validate_state(s: &OuProcessState) -> Result<(), String> {
        if ![s.mu, s.theta, s.sigma, s.x0, s.dt, s.sample_dt, s.threshold]
            .iter()
            .all(|v| v.is_finite())
            || !(-1_000.0..=1_000.0).contains(&s.mu)
            || !(0.001..=100.0).contains(&s.theta)
            || !(0.0..=100.0).contains(&s.sigma)
            || !(-1_000.0..=1_000.0).contains(&s.x0)
            || !(0.001..=5.0).contains(&s.dt)
            || !(0.001..=100.0).contains(&s.sample_dt)
            || !(-1_000.0..=1_000.0).contains(&s.threshold)
            || !(10..=MAX_OU_STEPS).contains(&s.n_steps)
            || !(8..=MAX_OU_PATHS).contains(&s.n_paths)
            || s.observed_series.len() > MAX_IMPORTED_CSV_POINTS
            || s.observed_series.iter().any(|v| !v.is_finite())
        {
            return Err("OU process state parameters out of valid bounds".into());
        }
        Ok(())
    }

    pub(crate) fn apply_state(&mut self, s: OuProcessState) -> Result<(), String> {
        Self::validate_state(&s)?;
        self.mu = s.mu;
        self.theta = s.theta;
        self.sigma = s.sigma;
        self.x0 = s.x0;
        self.dt = s.dt;
        self.sample_dt = s.sample_dt;
        self.threshold = s.threshold;
        self.n_steps = s.n_steps;
        self.n_paths = s.n_paths;
        self.rng_seed = s.rng_seed;
        self.observed_series = s.observed_series;
        self.bump_revision();
        self.recalculate();
        Ok(())
    }

    pub fn import_csv_series(&mut self, raw: &str) -> Result<usize, String> {
        let body = strip_utf8_bom(raw);
        let mut times = Vec::new();
        let mut values = Vec::new();
        let mut non_header_checked = false;

        for (line_idx, line) in body.lines().enumerate() {
            match parse_csv_line_pair(line)? {
                Some((t_opt, v)) => {
                    non_header_checked = true;
                    if values.len() >= MAX_IMPORTED_CSV_POINTS {
                        return Err(format!(
                            "CSV exceeds maximum {} observations",
                            MAX_IMPORTED_CSV_POINTS
                        ));
                    }
                    if let Some(t) = t_opt {
                        times.push(t);
                    }
                    values.push(v);
                }
                None => {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && !trimmed.starts_with('#') && non_header_checked {
                        return Err(format!("Malformed numeric row at line {}", line_idx + 1));
                    }
                    non_header_checked = true;
                }
            }
        }

        if values.len() < 3 {
            return Err("CSV calibration requires at least 3 numeric observations".into());
        }
        if times.len() == values.len() && times.len() >= 2 {
            let inferred_dt = times[1] - times[0];
            if inferred_dt.is_finite() && inferred_dt > 1e-4 {
                self.sample_dt = inferred_dt.clamp(0.001, 100.0);
            }
        }

        let fitted = estimate_ou_params(&values, self.sample_dt)
            .map_err(|e| format!("OU parameter calibration failed: {e}"))?;
        self.mu = fitted.mu.value();
        self.theta = fitted.theta.value();
        self.sigma = fitted.sigma.value();
        self.x0 = values[0];
        self.threshold = self.mu;
        self.observed_series = values;
        self.bump_revision();
        self.recalculate();
        Ok(self.observed_series.len())
    }

    pub fn recalibrate_from_observed(&mut self) -> Result<(), String> {
        if self.observed_series.len() < 3 {
            return Err("At least 3 observed points are required to calibrate".into());
        }
        let fitted = estimate_ou_params(&self.observed_series, self.sample_dt)
            .map_err(|e| format!("OU calibration failed: {e}"))?;
        self.mu = fitted.mu.value();
        self.theta = fitted.theta.value();
        self.sigma = fitted.sigma.value();
        self.x0 = self.observed_series[0];
        self.bump_revision();
        self.recalculate();
        Ok(())
    }

    fn compute_step_quantiles(&mut self, step: usize, stride: usize) -> OuBandRow {
        self.col_scratch.clear();
        for path_slice in self.traj_scratch[..self.n_paths * stride].chunks_exact(stride) {
            self.col_scratch.push(path_slice[step]);
        }
        self.col_scratch.sort_unstable_by(f64::total_cmp);
        let len = self.col_scratch.len();
        let mean = self.col_scratch.iter().sum::<f64>() / len as f64;
        let idx = |q: f64| ((q * (len.saturating_sub(1) as f64)).round() as usize).min(len - 1);
        OuBandRow {
            step,
            time: step as f64 * self.dt,
            mean,
            p05: self.col_scratch[idx(0.05)],
            p25: self.col_scratch[idx(0.25)],
            p75: self.col_scratch[idx(0.75)],
            p95: self.col_scratch[idx(0.95)],
        }
    }

    fn clear_cached_series(&mut self) {
        self.calc_error = None;
        self.mean_points.clear();
        self.p05_points.clear();
        self.p25_points.clear();
        self.p75_points.clear();
        self.p95_points.clear();
        self.sample_path_points.clear();
        self.observed_points.clear();
        self.band_rows.clear();
    }

    pub fn recalculate(&mut self) {
        self.clear_cached_series();
        let params = match OuParams::from_values(self.mu, self.theta, self.sigma) {
            Ok(p) => p,
            Err(e) => {
                self.calc_error = Some(format!("Invalid OU parameters: {e}"));
                return;
            }
        };
        let dt = match TimeStep::new(self.dt) {
            Ok(d) => d,
            Err(e) => {
                self.calc_error = Some(format!("Invalid time step: {e}"));
                return;
            }
        };

        self.momentum_class = OuAnalyzer::new(params, dt).momentum_type();
        self.half_life = std::f64::consts::LN_2 / params.theta.value();
        let stride = self.n_steps + 1;
        let needed = self.n_paths * stride;
        if self.traj_scratch.len() < needed {
            self.traj_scratch.resize(needed, 0.0);
        }

        let solver = EulerMaruyama::new(params, dt);
        let mut rng = OxidizeRng::new(self.rng_seed);
        for path_buf in self.traj_scratch[..needed].chunks_exact_mut(stride) {
            solver.simulate_into(self.x0, path_buf, &mut rng);
        }

        for (step, &val) in self.traj_scratch[..stride].iter().enumerate() {
            self.sample_path_points
                .push(PlotPoint::new(step as f64 * self.dt, val));
        }

        for step in 0..stride {
            let row = self.compute_step_quantiles(step, stride);
            self.mean_points.push(PlotPoint::new(row.time, row.mean));
            self.p05_points.push(PlotPoint::new(row.time, row.p05));
            self.p25_points.push(PlotPoint::new(row.time, row.p25));
            self.p75_points.push(PlotPoint::new(row.time, row.p75));
            self.p95_points.push(PlotPoint::new(row.time, row.p95));
            self.band_rows.push(row);
        }

        let above = self
            .col_scratch
            .iter()
            .filter(|&&v| v > self.threshold)
            .count();
        self.exceedance_prob = above as f64 / self.n_paths.max(1) as f64;
        let horizon = self.n_steps as f64 * self.dt;
        self.eq_line_points = [
            PlotPoint::new(0.0, self.mu),
            PlotPoint::new(horizon, self.mu),
        ];
        for (idx, &obs) in self.observed_series.iter().enumerate() {
            self.observed_points
                .push(PlotPoint::new(idx as f64 * self.sample_dt, obs));
        }
    }
}

// [cite:clinical_trials_statistics]
