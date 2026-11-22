//! Empirical cycling CSV calibration and analytical log-linear warm-starting for [`super::CapacityFadeTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CapacityFadeTool, MAX_ALPHA, MAX_BETA, MIN_ALPHA, MIN_BETA};
use crate::framework::empirical_csv::{CalibrationReport, EmpiricalCsvTable, EmpiricalTripleSeries};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, ClosureLeastSquaresObjective, LevenbergMarquardtSolver,
    LevenbergMarquardtWorkspace, ParameterBounds,
};

impl CapacityFadeTool {
    /// Runs damped Levenberg-Marquardt calibration for `alpha` and `beta` in [`PowerLawModel::new`].
    pub fn calibrate_from_loaded_csv(&mut self, ctx: Option<&egui::Context>) -> Result<(), String> {
        let Some(ref table) = self.empirical_csv.table else {
            return Err("Load a battery cycling CSV log before fitting parameters".to_string());
        };
        let (cycles, cap_obs, dod_series) = self.extract_cycling_series(table)?;
        if cycles.iter().any(|&c| c < 0.0) || cap_obs.iter().any(|&y| y <= 0.0) {
            let err = "Cycle counts must be >= 0 and capacities must be > 0".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        let y_max = cap_obs.iter().copied().fold(0.0_f64, f64::max);
        let scale = if y_max > 1.5 { 100.0 } else { 1.0 };

        let start = PlatformInstant::now();
        let summary = match fit_power_law_series(
            &cycles,
            &cap_obs,
            &dod_series,
            scale,
            self.alpha,
            self.beta,
        ) {
            Ok(s) => s,
            Err(err) => {
                self.empirical_csv.record_error(ctx, err.clone());
                return Err(err);
            }
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.commit_fade_fit(ctx, &summary, &cycles, &cap_obs, elapsed_ms)
    }

    fn extract_cycling_series(
        &self,
        table: &EmpiricalCsvTable,
    ) -> Result<EmpiricalTripleSeries, String> {
        if let Some(aux_idx) = self.empirical_csv.aux_col {
            let (cyc, cap, dods) = table.extract_xyz_sorted(
                self.empirical_csv.x_col,
                self.empirical_csv.y_col,
                aux_idx,
            )?;
            if dods.iter().any(|&d| !(1.0..=100.0).contains(&d)) {
                return Err("Mapped DoD (%) column must remain within 1..=100%".to_string());
            }
            Ok((cyc, cap, dods))
        } else {
            let (cyc, cap) =
                table.extract_xy_sorted(self.empirical_csv.x_col, self.empirical_csv.y_col)?;
            let fixed_dod = self.dod.clamp(1.0, 100.0);
            let dods = vec![fixed_dod; cyc.len()];
            Ok((cyc, cap, dods))
        }
    }

    fn commit_fade_fit(
        &mut self,
        ctx: Option<&egui::Context>,
        summary: &CalibrationSummary,
        cycles: &[f64],
        cap_obs: &[f64],
        elapsed_ms: f64,
    ) -> Result<(), String> {
        let fitted_alpha = summary.parameters[0];
        let fitted_beta = summary.parameters[1];
        if !fitted_alpha.is_finite() || !fitted_beta.is_finite() {
            let err = "Rejected non-finite calibrated battery fade parameters".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        self.alpha = fitted_alpha.clamp(MIN_ALPHA, MAX_ALPHA);
        self.beta = fitted_beta.clamp(MIN_BETA, MAX_BETA);
        if let Some(&c_last) = cycles.last() {
            if c_last > self.cycles_to_simulate {
                self.cycles_to_simulate = c_last.clamp(100.0, 10_000.0);
            }
        }
        self.revision = self.revision.wrapping_add(1).max(1);

        let mut report = CalibrationReport::from_summary(
            &["alpha", "beta"],
            summary,
            cycles,
            cap_obs,
            elapsed_ms,
        )?;
        let model = PowerLawModel::new(self.alpha, self.beta);
        let n70 = DepthOfDischarge::new(self.dod.clamp(1.0, 100.0))
            .map(|d| model.n70(d).as_f64())
            .unwrap_or(0.0);
        report.derived_metrics.push((
            "Calibrated N70 Cycle Life".to_string(),
            format!("{n70:.0} cycles"),
        ));
        let announce = format!(
            "Calibrated Capacity Fade: alpha {:.0}, beta {:.4}, N70 {:.0} cycles",
            self.alpha, self.beta, n70
        );
        self.empirical_csv.record_fit_success(ctx, report, announce);
        Ok(())
    }
}

fn fit_power_law_series(
    cycles: &[f64],
    cap_obs: &[f64],
    dods: &[f64],
    scale: f64,
    init_alpha: f64,
    init_beta: f64,
) -> Result<CalibrationSummary, String> {
    let idx_grid: Vec<f64> = (0..cycles.len()).map(|i| i as f64).collect();
    let mut objective = ClosureLeastSquaresObjective::new(&idx_grid, cap_obs, |idx_f, p| {
        let k = idx_f as usize;
        let model = PowerLawModel::new(p[0], p[1]);
        let dod = DepthOfDischarge::new_clamped(dods[k]);
        let cyc = Cycles::new_clamped(cycles[k]);
        scale * model.capacity(cyc, dod).as_f64()
    })
    .map_err(|e| e.to_string())?;

    let bounds = [
        ParameterBounds::new(MIN_ALPHA, MAX_ALPHA).unwrap(),
        ParameterBounds::new(MIN_BETA, MAX_BETA).unwrap(),
    ];
    let init = warm_start_power_law(cycles, cap_obs, dods, scale, init_alpha, init_beta);
    let mut ws = LevenbergMarquardtWorkspace::new(cycles.len(), 2).map_err(|e| e.to_string())?;
    LevenbergMarquardtSolver::default()
        .fit(&mut objective, &init, &bounds, &mut ws)
        .map_err(|e| e.to_string())
}

fn warm_start_power_law(
    cycles: &[f64],
    cap_obs: &[f64],
    dods: &[f64],
    scale: f64,
    fallback_alpha: f64,
    fallback_beta: f64,
) -> [f64; 2] {
    let ln_07 = 0.7_f64.ln();
    let mut pairs = Vec::with_capacity(cycles.len());
    for k in 0..cycles.len() {
        let soh = cap_obs[k] / scale;
        if cycles[k] > 0.0 && (1e-4..0.9999).contains(&soh) && dods[k] > 0.0 {
            let est_n70 = cycles[k] * ln_07 / soh.ln();
            if est_n70 > 1.0 && est_n70.is_finite() {
                pairs.push((dods[k].ln(), est_n70.ln()));
            }
        }
    }
    if pairs.is_empty() {
        return [
            fallback_alpha.clamp(MIN_ALPHA, MAX_ALPHA),
            fallback_beta.clamp(MIN_BETA, MAX_BETA),
        ];
    }
    let inv_m = 1.0 / (pairs.len() as f64);
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() * inv_m;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() * inv_m;
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let sxy: f64 = pairs
        .iter()
        .map(|p| (p.0 - mean_x) * (p.1 - mean_y))
        .sum();
    let beta_0 = if pairs.len() >= 2 && sxx > 1e-8 {
        (sxy / sxx).clamp(MIN_BETA, MAX_BETA)
    } else {
        fallback_beta.clamp(MIN_BETA, MAX_BETA)
    };
    let alpha_0 = (mean_y - beta_0 * mean_x)
        .exp()
        .clamp(MIN_ALPHA, MAX_ALPHA);
    [alpha_0, beta_0]
}
