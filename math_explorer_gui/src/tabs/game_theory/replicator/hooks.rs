//! Cooperative sensitivity hooks, dataset exports, and CSV calibration for [`super::ReplicatorDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ReplicatorDynamicsTool;
use crate::framework::{
    extract_calibration_series, fit_bounded_parameters, CalibratableParamSpec, SensitivityParamSpec,
};
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;

impl ReplicatorDynamicsTool {
    pub(super) fn export_replicator_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let plot_ds = egui_plot::get_last_plot_datasets();
        if plot_ds.iter().any(|(n, _)| n.contains("Distribution")) {
            return plot_ds;
        }
        let mut datasets = Vec::with_capacity(self.strategy_names.len() + 2);
        for (idx, name) in self.strategy_names.iter().enumerate() {
            let pts: Vec<[f64; 2]> = self
                .trajectory
                .iter()
                .filter_map(|(t, x)| x.get(idx).map(|&v| [*t, v]))
                .collect();
            datasets.push((name.clone(), pts));
        }
        if !self.mean_fitness_series.is_empty() {
            datasets.push(("Mean Fitness".to_string(), self.mean_fitness_series.clone()));
        }
        if !self.trajectory.is_empty() {
            let idx = ((self.playback_time / self.dt.max(1e-6)).round() as usize)
                .min(self.trajectory.len() - 1);
            let dist_pts: Vec<[f64; 2]> = self.trajectory[idx]
                .1
                .iter()
                .enumerate()
                .map(|(i, &val)| [i as f64, val])
                .collect();
            datasets.push(("Distribution".to_string(), dist_pts));
        }
        datasets
    }

    pub(super) fn replicator_sensitivity_specs(&self) -> Vec<SensitivityParamSpec> {
        if self.payoff_matrix.nrows() < 2 || self.payoff_matrix.ncols() < 2 {
            return Vec::new();
        }
        vec![
            SensitivityParamSpec::new(
                "a_00",
                "Payoff A(1,1)",
                self.payoff_matrix[(0, 0)],
                -5.0,
                10.0,
            ),
            SensitivityParamSpec::new(
                "a_01",
                "Payoff A(1,2)",
                self.payoff_matrix[(0, 1)],
                -5.0,
                10.0,
            ),
            SensitivityParamSpec::new(
                "a_10",
                "Payoff A(2,1)",
                self.payoff_matrix[(1, 0)],
                -5.0,
                10.0,
            ),
            SensitivityParamSpec::new(
                "a_11",
                "Payoff A(2,2)",
                self.payoff_matrix[(1, 1)],
                -5.0,
                10.0,
            ),
        ]
    }

    pub(super) fn eval_replicator_sensitivity(
        &self,
        params: &[(&str, f64)],
    ) -> Option<(&'static str, f64)> {
        if self.payoff_matrix.nrows() < 2 || self.payoff_matrix.ncols() < 2 {
            return None;
        }
        let mut mat = self.payoff_matrix.clone();
        for &(k, v) in params {
            if !v.is_finite() {
                return None;
            }
            match k {
                "a_00" => mat[(0, 0)] = v.clamp(-20.0, 20.0),
                "a_01" => mat[(0, 1)] = v.clamp(-20.0, 20.0),
                "a_10" => mat[(1, 0)] = v.clamp(-20.0, 20.0),
                "a_11" => mat[(1, 1)] = v.clamp(-20.0, 20.0),
                _ => {}
            }
        }
        let sys = ReplicatorDynamics::new(mat.clone()).ok()?;
        let dt = self.dt.clamp(0.01, 0.2);
        let horizon = self.time_horizon.clamp(1.0, 40.0);
        let traj = sys.simulate(self.initial_population.clone(), horizon, dt);
        let (_, x_end) = traj.last()?;
        let fit = x_end.dot(&(&mat * x_end));
        fit.is_finite().then_some(("Terminal Mean Fitness", fit))
    }

    pub(super) fn calibrate_replicator_from_csv(
        &mut self,
        datasets: &[egui_plot::ImportedDataset],
    ) -> Result<bool, String> {
        if self.payoff_matrix.nrows() < 2 || self.payoff_matrix.ncols() < 2 {
            return Err("Replicator calibration requires at least 2 strategies".into());
        }
        let points = extract_calibration_series(datasets)?;
        let base_mat = self.payoff_matrix.clone();
        let init_pop = self.initial_population.clone();
        let max_t = points
            .iter()
            .fold(self.time_horizon.max(1.0), |acc, p| acc.max(p[0]));
        let dt = self.dt.clamp(0.01, 0.1);
        let specs = [
            CalibratableParamSpec::new("a_00", base_mat[(0, 0)].clamp(-5.0, 5.0), -5.0, 5.0),
            CalibratableParamSpec::new("a_01", base_mat[(0, 1)].clamp(-5.0, 5.0), -5.0, 5.0),
        ];
        let fitted = fit_bounded_parameters(points, &specs, |t_query, theta| {
            let mut m = base_mat.clone();
            m[(0, 0)] = theta[0];
            m[(0, 1)] = theta[1];
            let sys = ReplicatorDynamics::new(m).ok()?;
            let traj = sys.simulate(init_pop.clone(), max_t.min(60.0), dt);
            if traj.is_empty() {
                return None;
            }
            let idx = ((t_query.max(0.0) / dt).round() as usize).min(traj.len() - 1);
            traj[idx].1.get(0).copied().filter(|v| v.is_finite())
        })?;
        self.symbolic_mode = false;
        self.payoff_matrix[(0, 0)] = fitted[0];
        self.payoff_matrix[(0, 1)] = fitted[1];
        self.run_simulation();
        Ok(true)
    }
}
