#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CorrelatedRiskScenarioTool, RiskAnalysisMode};
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::copula::{
    sgp_joint_probability, Correlation, CorrelationMatrix, GaussianCopula, Probability,
};
use math_explorer::pure_math::statistics::ou_process::{
    EulerMaruyama, MomentumType, OuParams, TimeStep,
};

pub const MAX_OU_PATHS: usize = 200;
pub const MAX_OU_STEPS: usize = 2_000;
pub const MAX_DISPLAYED_OU_PATHS: usize = 20;
pub const MAX_SCENARIO_LABEL_BYTES: usize = 96;
pub const COPULA_SWEEP_POINTS: usize = 41;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopulaCacheKey {
    pub prob_a_bits: u64,
    pub prob_b_bits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OuCacheKey {
    pub mu_bits: u64,
    pub theta_bits: u64,
    pub sigma_bits: u64,
    pub x0_bits: u64,
    pub dt_bits: u64,
    pub steps: usize,
    pub paths: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct CopulaSweepRow {
    pub rho: f64,
    pub joint_prob: f64,
    pub naive_prob: f64,
    pub edge_delta: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OuSummaryStats {
    pub mean_final: f64,
    pub std_final: f64,
    pub p05_final: f64,
    pub p95_final: f64,
    pub horizon: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorrelatedRiskState {
    pub prob_a: f64,
    pub prob_b: f64,
    pub rho: f64,
    pub ou_mu: f64,
    pub ou_theta: f64,
    pub ou_sigma: f64,
    pub ou_x0: f64,
    pub ou_dt: f64,
    pub ou_steps: usize,
    pub ou_paths: usize,
    pub seed: u64,
    pub mode: RiskAnalysisMode,
    pub scenario_label: String,
}

impl CorrelatedRiskState {
    pub fn validate(&self) -> Result<(), String> {
        let floats = [
            self.prob_a,
            self.prob_b,
            self.rho,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_x0,
            self.ou_dt,
        ];
        if !floats.iter().all(|v| v.is_finite()) {
            return Err("All correlated risk parameters must be finite".into());
        }
        if !(0.01..=0.99).contains(&self.prob_a) || !(0.01..=0.99).contains(&self.prob_b) {
            return Err("Marginal probabilities prob_a and prob_b must lie in [0.01, 0.99]".into());
        }
        if !(-0.99..=0.99).contains(&self.rho) {
            return Err("Correlation rho must lie in [-0.99, 0.99]".into());
        }
        if !(-10.0..=10.0).contains(&self.ou_mu) || !(-10.0..=10.0).contains(&self.ou_x0) {
            return Err("OU mean mu and initial state x0 must lie in [-10.0, 10.0]".into());
        }
        if !(0.05..=20.0).contains(&self.ou_theta) || !(0.0..=5.0).contains(&self.ou_sigma) {
            return Err("OU theta must lie in [0.05, 20.0] and sigma in [0.0, 5.0]".into());
        }
        if !(0.001..=0.5).contains(&self.ou_dt) {
            return Err("OU time step dt must lie in [0.001, 0.5]".into());
        }
        if !(1..=MAX_OU_STEPS).contains(&self.ou_steps)
            || !(1..=MAX_OU_PATHS).contains(&self.ou_paths)
        {
            return Err(format!(
                "OU bounds exceeded: steps must be 1..={MAX_OU_STEPS}, paths 1..={MAX_OU_PATHS}"
            ));
        }
        if self.scenario_label.trim().is_empty()
            || self.scenario_label.len() > MAX_SCENARIO_LABEL_BYTES
        {
            return Err("Scenario label must be 1..=96 UTF-8 bytes".into());
        }
        OuParams::from_values(self.ou_mu, self.ou_theta, self.ou_sigma)
            .map_err(|e| format!("Invalid OU parameters: {e}"))?;
        TimeStep::new(self.ou_dt).map_err(|e| format!("Invalid OU time step: {e}"))?;
        Ok(())
    }
}

impl CorrelatedRiskScenarioTool {
    pub(crate) fn apply_validated_state(&mut self, s: CorrelatedRiskState) {
        self.prob_a = s.prob_a;
        self.prob_b = s.prob_b;
        self.rho = s.rho;
        self.ou_mu = s.ou_mu;
        self.ou_theta = s.ou_theta;
        self.ou_sigma = s.ou_sigma;
        self.ou_x0 = s.ou_x0;
        self.ou_dt = s.ou_dt;
        self.ou_steps = s.ou_steps;
        self.ou_paths = s.ou_paths;
        self.seed = s.seed;
        self.mode = s.mode;
        self.scenario_label = s.scenario_label;
        self.last_copula_key = None;
        self.last_ou_key = None;
        self.recalculate_if_needed();
    }

    pub(crate) fn reset_defaults(&mut self) {
        let mode = self.mode;
        *self = Self::default();
        self.mode = mode;
    }

    pub(crate) fn recalculate_if_needed(&mut self) {
        self.update_copula_cache();
        self.update_ou_cache();
    }

    fn update_copula_cache(&mut self) {
        self.copula_error = None;
        let pa = match Probability::new(self.prob_a.clamp(0.01, 0.99)) {
            Ok(p) => p,
            Err(e) => {
                self.copula_error = Some(format!("Invalid P(A): {e}"));
                return;
            }
        };
        let pb = match Probability::new(self.prob_b.clamp(0.01, 0.99)) {
            Ok(p) => p,
            Err(e) => {
                self.copula_error = Some(format!("Invalid P(B): {e}"));
                return;
            }
        };

        let key = CopulaCacheKey {
            prob_a_bits: pa.value().to_bits(),
            prob_b_bits: pb.value().to_bits(),
        };
        if self.last_copula_key != Some(key) {
            self.rebuild_copula_sweep(pa, pb);
            self.last_copula_key = Some(key);
        }
        self.update_active_sgp_point(pa, pb);
    }

    fn rebuild_copula_sweep(&mut self, pa: Probability, pb: Probability) {
        self.copula_curve_points.clear();
        self.copula_rows.clear();
        let naive = pa.value() * pb.value();
        self.naive_prob = naive;
        self.naive_line_points = [PlotPoint::new(-0.99, naive), PlotPoint::new(0.99, naive)];

        let half = (COPULA_SWEEP_POINTS - 1) as f64 / 2.0;
        for idx in 0..COPULA_SWEEP_POINTS {
            let r = ((idx as f64 - half) / half) * 0.99;
            let joint_res = Correlation::new(r)
                .and_then(GaussianCopula::bivariate)
                .and_then(|copula| copula.cdf(&[pa, pb]));
            if let Ok(joint) = joint_res {
                let jp = joint.value();
                self.copula_curve_points.push(PlotPoint::new(r, jp));
                self.copula_rows.push(CopulaSweepRow {
                    rho: r,
                    joint_prob: jp,
                    naive_prob: naive,
                    edge_delta: jp - naive,
                });
            }
        }
    }

    fn update_active_sgp_point(&mut self, pa: Probability, pb: Probability) {
        self.active_rho_marker.clear();
        let clamped_rho = self.rho.clamp(-0.99, 0.99);
        let res = Correlation::new(clamped_rho)
            .and_then(CorrelationMatrix::bivariate)
            .and_then(|mat| sgp_joint_probability(&[pa, pb], &mat));
        match res {
            Ok(joint) => {
                self.active_sgp_prob = joint.value();
                self.active_rho_marker
                    .push(PlotPoint::new(clamped_rho, joint.value()));
            }
            Err(e) => self.copula_error = Some(format!("Copula evaluation failed: {e}")),
        }
    }

    fn update_ou_cache(&mut self) {
        let steps = self.ou_steps.clamp(1, MAX_OU_STEPS);
        let paths = self.ou_paths.clamp(1, MAX_OU_PATHS);
        let key = OuCacheKey {
            mu_bits: self.ou_mu.to_bits(),
            theta_bits: self.ou_theta.to_bits(),
            sigma_bits: self.ou_sigma.to_bits(),
            x0_bits: self.ou_x0.to_bits(),
            dt_bits: self.ou_dt.to_bits(),
            steps,
            paths,
            seed: self.seed,
        };
        if self.last_ou_key == Some(key) {
            return;
        }
        self.ou_error = None;
        self.ou_momentum = MomentumType::classify(self.ou_theta);

        let params = match OuParams::from_values(self.ou_mu, self.ou_theta, self.ou_sigma) {
            Ok(p) => p,
            Err(e) => {
                self.ou_error = Some(format!("Invalid OU parameters: {e}"));
                return;
            }
        };
        let dt = match TimeStep::new(self.ou_dt) {
            Ok(d) => d,
            Err(e) => {
                self.ou_error = Some(format!("Invalid OU time step: {e}"));
                return;
            }
        };

        let solver = EulerMaruyama::new(params, dt);
        let mut rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        let trajectories = solver.simulate_paths(self.ou_x0, steps, paths, &mut rng);
        self.populate_ou_series_and_stats(&trajectories, steps, dt.value());
        self.last_ou_key = Some(key);
    }

    fn populate_ou_series_and_stats(
        &mut self,
        trajectories: &[Vec<f64>],
        steps: usize,
        dt_val: f64,
    ) {
        self.ou_sample_paths.clear();
        self.ou_mean_path.clear();
        let n_paths = trajectories.len().max(1);
        let horizon = steps as f64 * dt_val;
        self.ou_equilibrium_line = [
            PlotPoint::new(0.0, self.ou_mu),
            PlotPoint::new(horizon, self.ou_mu),
        ];

        for (p_idx, traj) in trajectories.iter().take(MAX_DISPLAYED_OU_PATHS).enumerate() {
            let pts: Vec<PlotPoint> = traj
                .iter()
                .enumerate()
                .map(|(s, &val)| PlotPoint::new(s as f64 * dt_val, val))
                .collect();
            self.ou_sample_paths
                .push((format!("Path {}", p_idx + 1), pts));
        }

        for s in 0..=steps {
            let mean_t = trajectories.iter().map(|p| p[s]).sum::<f64>() / n_paths as f64;
            self.ou_mean_path
                .push(PlotPoint::new(s as f64 * dt_val, mean_t));
        }

        let mut finals: Vec<f64> = trajectories
            .iter()
            .filter_map(|p| p.last().copied())
            .collect();
        finals.sort_unstable_by(|a, b| a.total_cmp(b));
        let mean_final = finals.iter().sum::<f64>() / finals.len() as f64;
        let var = finals
            .iter()
            .map(|&x| (x - mean_final).powi(2))
            .sum::<f64>()
            / finals.len() as f64;
        let idx05 = ((finals.len() - 1) as f64 * 0.05) as usize;
        let idx95 = ((finals.len() - 1) as f64 * 0.95) as usize;
        self.ou_stats = Some(OuSummaryStats {
            mean_final,
            std_final: var.sqrt(),
            p05_final: finals[idx05],
            p95_final: finals[idx95],
            horizon,
        });
    }
}
