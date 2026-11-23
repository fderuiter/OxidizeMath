//! Interactive Monte Carlo variance-reduction, randomized QMC, and `BCa` bootstrap workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;
mod state;
#[cfg(test)]
mod tests;
mod ui;

pub use state::{BootstrapStatKind, WorkbenchBenchmark, WorkbenchViewMode};
use state::{evaluate_control, evaluate_iid, evaluate_snis, WorkbenchState};

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::error::MonteCarloError;
use math_explorer::pure_math::statistics::bootstrap::{
    sample_mean, sample_median_in_place, BootstrapComparison, BootstrapWorkspace,
};
use math_explorer::pure_math::statistics::monte_carlo::{MonteCarloEstimate, MonteCarloWorkspace};
use math_explorer::registry::MAX_ITERATIONS;
use oxidize_core::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;

pub(super) const MIN_SAMPLES: usize = 32;
pub(super) const MAX_SAMPLES: usize = MAX_ITERATIONS;
pub(super) const DEFAULT_SAMPLES: usize = 512;
pub(super) const MIN_REPLICATES: usize = 64;
pub(super) const MAX_REPLICATES: usize = MAX_ITERATIONS;
pub(super) const DEFAULT_REPLICATES: usize = 600;
pub(super) const DEFAULT_SEED: u64 = 42;
pub(super) const HISTOGRAM_BINS: usize = 24;
pub(super) const DECAY_BUDGETS: [usize; 7] = [32, 64, 128, 256, 512, 1024, 2048];
const MICRO_REPEATS: usize = 4;

/// Interactive `FinancialMathTab` workbench comparing variance reduction, QMC, and `BCa` bootstrap.
pub struct MonteCarloWorkbenchTool {
    pub(super) benchmark: WorkbenchBenchmark,
    pub(super) view_mode: WorkbenchViewMode,
    pub(super) num_samples: usize,
    pub(super) bootstrap_replicates: usize,
    pub(super) confidence_level: f64,
    pub(super) importance_tilt: f64,
    pub(super) bootstrap_statistic: BootstrapStatKind,
    pub(super) seed: u64,

    mc_ws_1d: MonteCarloWorkspace,
    mc_ws_3d: MonteCarloWorkspace,
    mc_ws_4d: MonteCarloWorkspace,
    boot_ws: BootstrapWorkspace,
    payoff_samples: Vec<f64>,

    pub(super) exact_reference: f64,
    pub(super) est_iid: Option<MonteCarloEstimate>,
    pub(super) est_antithetic: Option<MonteCarloEstimate>,
    pub(super) est_control: Option<MonteCarloEstimate>,
    pub(super) est_lhs: Option<MonteCarloEstimate>,
    pub(super) est_halton: Option<MonteCarloEstimate>,
    pub(super) est_sobol: Option<MonteCarloEstimate>,
    pub(super) est_snis: Result<MonteCarloEstimate, MonteCarloError>,
    pub(super) boot_comparison: Option<BootstrapComparison>,

    pub(super) decay_iid: Vec<PlotPoint>,
    pub(super) decay_antithetic: Vec<PlotPoint>,
    pub(super) decay_control: Vec<PlotPoint>,
    pub(super) decay_lhs: Vec<PlotPoint>,
    pub(super) decay_halton: Vec<PlotPoint>,
    pub(super) decay_sobol: Vec<PlotPoint>,
    pub(super) slope_half_ref: Vec<PlotPoint>,
    pub(super) slope_one_ref: Vec<PlotPoint>,

    pub(super) hist_bins: [(f64, f64); HISTOGRAM_BINS],
    pub(super) hist_bin_width: f64,
    pub(super) hist_max_density: f64,
    pub(super) line_pct_lo: [PlotPoint; 2],
    pub(super) line_pct_hi: [PlotPoint; 2],
    pub(super) line_bca_lo: [PlotPoint; 2],
    pub(super) line_bca_hi: [PlotPoint; 2],
    pub(super) line_point_est: [PlotPoint; 2],
    pub(super) last_compute_ms: f64,
}

impl Default for MonteCarloWorkbenchTool {
    fn default() -> Self {
        let z_line = [PlotPoint::new(0.0, 0.0), PlotPoint::new(0.0, 1.0)];
        let mut tool = Self {
            benchmark: WorkbenchBenchmark::DiscountedBasketCall4D,
            view_mode: WorkbenchViewMode::SplitOverview,
            num_samples: DEFAULT_SAMPLES,
            bootstrap_replicates: DEFAULT_REPLICATES,
            confidence_level: 0.95,
            importance_tilt: 0.85,
            bootstrap_statistic: BootstrapStatKind::Mean,
            seed: DEFAULT_SEED,
            mc_ws_1d: MonteCarloWorkspace::new(MAX_SAMPLES, 1).expect("valid 1D MC workspace"),
            mc_ws_3d: MonteCarloWorkspace::new(MAX_SAMPLES, 3).expect("valid 3D MC workspace"),
            mc_ws_4d: MonteCarloWorkspace::new(MAX_SAMPLES, 4).expect("valid 4D MC workspace"),
            boot_ws: BootstrapWorkspace::new(MAX_SAMPLES, MAX_REPLICATES)
                .expect("valid bootstrap workspace"),
            payoff_samples: vec![0.0; MAX_SAMPLES],
            exact_reference: 0.0,
            est_iid: None,
            est_antithetic: None,
            est_control: None,
            est_lhs: None,
            est_halton: None,
            est_sobol: None,
            est_snis: Err(MonteCarloError::InvalidSampleCount { samples: 0 }),
            boot_comparison: None,
            decay_iid: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            decay_antithetic: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            decay_control: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            decay_lhs: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            decay_halton: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            decay_sobol: Vec::with_capacity(DECAY_BUDGETS.len() + 1),
            slope_half_ref: Vec::with_capacity(2),
            slope_one_ref: Vec::with_capacity(2),
            hist_bins: [(0.0, 0.0); HISTOGRAM_BINS],
            hist_bin_width: 1.0,
            hist_max_density: 1.0,
            line_pct_lo: z_line,
            line_pct_hi: z_line,
            line_bca_lo: z_line,
            line_bca_hi: z_line,
            line_point_est: z_line,
            last_compute_ms: 0.0,
        };
        tool.recompute();
        tool
    }
}

impl MonteCarloWorkbenchTool {
    /// Restores canonical default benchmark, sample budget, and seed parameters.
    pub fn reset_defaults(&mut self) {
        self.benchmark = WorkbenchBenchmark::DiscountedBasketCall4D;
        self.view_mode = WorkbenchViewMode::SplitOverview;
        self.num_samples = DEFAULT_SAMPLES;
        self.bootstrap_replicates = DEFAULT_REPLICATES;
        self.confidence_level = 0.95;
        self.importance_tilt = 0.85;
        self.bootstrap_statistic = BootstrapStatKind::Mean;
        self.seed = DEFAULT_SEED;
        self.recompute();
    }

    /// Recomputes all estimators, log-decay curves, and bootstrap intervals with timing.
    pub fn recompute(&mut self) {
        let start = PlatformInstant::now();
        self.recompute_in_place();
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Zero-allocation hot evaluation of Monte Carlo, QMC, SNIS `ESS`, and `BCa` bootstrap.
    pub fn recompute_in_place(&mut self) {
        let n = self.num_samples.clamp(MIN_SAMPLES, MAX_SAMPLES);
        let b_reps = self
            .bootstrap_replicates
            .clamp(MIN_REPLICATES, MAX_REPLICATES);
        let conf = self.confidence_level.clamp(0.80, 0.99);
        let bench = self.benchmark;
        self.exact_reference = bench.exact_value();

        self.compute_active_estimates(bench, n, conf);
        self.compute_error_decay_curves(bench, n, conf);
        self.compute_bootstrap_histogram(bench, n, b_reps, conf);
    }

    fn active_mc_ws_mut(&mut self, bench: WorkbenchBenchmark) -> &mut MonteCarloWorkspace {
        match bench {
            WorkbenchBenchmark::SkewedWealthReturn => &mut self.mc_ws_1d,
            WorkbenchBenchmark::SmoothSeparable3D => &mut self.mc_ws_3d,
            WorkbenchBenchmark::DiscountedBasketCall4D => &mut self.mc_ws_4d,
        }
    }

    fn compute_active_estimates(&mut self, bench: WorkbenchBenchmark, n: usize, conf: f64) {
        let tilt = self.importance_tilt;
        let seed = self.seed;
        let ws = self.active_mc_ws_mut(bench);
        let _ = ws.set_confidence_level(conf);

        let mut rng_iid = OxidizeRng::new(seed ^ 0x1111_A001);
        let mut rng_anti = OxidizeRng::new(seed ^ 0x2222_B002);
        let mut rng_ctrl = OxidizeRng::new(seed ^ 0x3333_C003);
        let mut rng_lhs = OxidizeRng::new(seed ^ 0x4444_D004);
        let mut rng_halton = OxidizeRng::new(seed ^ 0x5555_E005);
        let mut rng_sobol = OxidizeRng::new(seed ^ 0x6666_F006);
        let mut rng_snis = OxidizeRng::new(seed ^ 0x7777_0007);

        let iid = ws
            .estimate_iid(n, &mut rng_iid, |u| evaluate_iid(bench, u))
            .ok();
        let anti = ws
            .estimate_antithetic(n, &mut rng_anti, |u| evaluate_iid(bench, u))
            .ok();
        let ctrl = ws
            .estimate_control_variate(n, bench.control_mean(), &mut rng_ctrl, |u| {
                evaluate_control(bench, u)
            })
            .ok();
        let lhs = ws
            .estimate_latin_hypercube(n, &mut rng_lhs, |u| evaluate_iid(bench, u))
            .ok();
        let halton = ws
            .estimate_qmc_halton(n, &mut rng_halton, |u| evaluate_iid(bench, u))
            .ok();
        let sobol = ws
            .estimate_qmc_sobol(n, &mut rng_sobol, |u| evaluate_iid(bench, u))
            .ok();
        let snis =
            ws.estimate_importance_sampling(n, &mut rng_snis, |u| evaluate_snis(bench, u, tilt));

        self.est_iid = iid;
        self.est_antithetic = anti;
        self.est_control = ctrl;
        self.est_lhs = lhs;
        self.est_halton = halton;
        self.est_sobol = sobol;
        self.est_snis = snis;
    }

    fn compute_error_decay_curves(&mut self, bench: WorkbenchBenchmark, max_n: usize, conf: f64) {
        self.decay_iid.clear();
        self.decay_antithetic.clear();
        self.decay_control.clear();
        self.decay_lhs.clear();
        self.decay_halton.clear();
        self.decay_sobol.clear();
        self.slope_half_ref.clear();
        self.slope_one_ref.clear();

        let exact = bench.exact_value();
        for &budget in &DECAY_BUDGETS {
            if budget > max_n {
                break;
            }
            self.push_decay_point(bench, budget, conf, exact);
        }
        if max_n > DECAY_BUDGETS[DECAY_BUDGETS.len() - 1] {
            self.push_decay_point(bench, max_n, conf, exact);
        }
        if let (Some(first), Some(last)) = (self.decay_iid.first(), self.decay_iid.last()) {
            let (x0, x1) = (first.x, last.x);
            self.slope_half_ref.push(PlotPoint::new(x0, first.y));
            self.slope_half_ref
                .push(PlotPoint::new(x1, first.y - 0.5 * (x1 - x0)));
            let sob_y0 = self.decay_sobol.first().map_or(first.y, |p| p.y);
            self.slope_one_ref.push(PlotPoint::new(x0, sob_y0));
            self.slope_one_ref
                .push(PlotPoint::new(x1, sob_y0 - (x1 - x0)));
        }
    }

    fn push_decay_point(&mut self, bench: WorkbenchBenchmark, budget: usize, conf: f64, exact: f64) {
        let log_n = (budget as f64).log10();
        let seed = self.seed;
        let mut sq = [0.0_f64; 6];
        {
            let ws = self.active_mc_ws_mut(bench);
            let _ = ws.set_confidence_level(conf);
            for rep in 0..MICRO_REPEATS {
                let base = seed.wrapping_add((rep as u64).wrapping_mul(0x9E37_79B9))
                    ^ ((budget as u64).wrapping_mul(0x517C_C1B7));
                let mut r0 = OxidizeRng::new(base ^ 0x101);
                let mut r1 = OxidizeRng::new(base ^ 0x202);
                let mut r2 = OxidizeRng::new(base ^ 0x303);
                let mut r3 = OxidizeRng::new(base ^ 0x404);
                let mut r4 = OxidizeRng::new(base ^ 0x505);
                let mut r5 = OxidizeRng::new(base ^ 0x606);

                if let Ok(e) = ws.estimate_iid(budget, &mut r0, |u| evaluate_iid(bench, u)) {
                    sq[0] += (e.mean - exact).powi(2);
                }
                if let Ok(e) = ws.estimate_antithetic(budget, &mut r1, |u| evaluate_iid(bench, u)) {
                    sq[1] += (e.mean - exact).powi(2);
                }
                if let Ok(e) =
                    ws.estimate_control_variate(budget, bench.control_mean(), &mut r2, |u| {
                        evaluate_control(bench, u)
                    })
                {
                    sq[2] += (e.mean - exact).powi(2);
                }
                if let Ok(e) =
                    ws.estimate_latin_hypercube(budget, &mut r3, |u| evaluate_iid(bench, u))
                {
                    sq[3] += (e.mean - exact).powi(2);
                }
                if let Ok(e) = ws.estimate_qmc_halton(budget, &mut r4, |u| evaluate_iid(bench, u)) {
                    sq[4] += (e.mean - exact).powi(2);
                }
                if let Ok(e) = ws.estimate_qmc_sobol(budget, &mut r5, |u| evaluate_iid(bench, u)) {
                    sq[5] += (e.mean - exact).powi(2);
                }
            }
        }
        let inv_r = 1.0 / (MICRO_REPEATS as f64);
        let to_log_rmse = |sum_sq: f64| ((sum_sq * inv_r).sqrt().max(1e-12)).log10();
        self.decay_iid.push(PlotPoint::new(log_n, to_log_rmse(sq[0])));
        self.decay_antithetic
            .push(PlotPoint::new(log_n, to_log_rmse(sq[1])));
        self.decay_control
            .push(PlotPoint::new(log_n, to_log_rmse(sq[2])));
        self.decay_lhs.push(PlotPoint::new(log_n, to_log_rmse(sq[3])));
        self.decay_halton
            .push(PlotPoint::new(log_n, to_log_rmse(sq[4])));
        self.decay_sobol
            .push(PlotPoint::new(log_n, to_log_rmse(sq[5])));
    }

    fn compute_bootstrap_histogram(
        &mut self,
        bench: WorkbenchBenchmark,
        n: usize,
        b_reps: usize,
        conf: f64,
    ) {
        let mut rng_data = OxidizeRng::new(self.seed ^ 0xABCD_1234);
        let mut unit_point = [0.0_f64; 4];
        let dim = bench.dimension();
        for slot in self.payoff_samples.iter_mut().take(n) {
            for coord in unit_point.iter_mut().take(dim) {
                *coord = rand::Rng::r#gen::<f64>(&mut rng_data).clamp(1e-12, 1.0 - 1e-12);
            }
            *slot = evaluate_iid(bench, &unit_point[..dim]);
        }

        let mut rng_boot = OxidizeRng::new(self.seed ^ 0xDCBA_9876);
        let data_slice = &self.payoff_samples[..n];
        self.boot_comparison = match self.bootstrap_statistic {
            BootstrapStatKind::Mean => self
                .boot_ws
                .compare_intervals(data_slice, b_reps, conf, &mut rng_boot, sample_mean)
                .ok(),
            BootstrapStatKind::Median => self
                .boot_ws
                .compare_intervals_with_scratch(
                    data_slice,
                    b_reps,
                    conf,
                    &mut rng_boot,
                    sample_median_in_place,
                )
                .ok(),
        };
        self.populate_histogram_bins();
    }

    fn populate_histogram_bins(&mut self) {
        let reps = self.boot_ws.last_replicates();
        if reps.is_empty() {
            self.hist_bins = [(0.0, 0.0); HISTOGRAM_BINS];
            self.hist_bin_width = 1.0;
            self.hist_max_density = 1.0;
            return;
        }
        let (lo, hi) = (reps[0], reps[reps.len() - 1]);
        let bin_w = ((hi - lo).max(1e-9)) / (HISTOGRAM_BINS as f64);
        let mut counts = [0usize; HISTOGRAM_BINS];
        for &val in reps {
            let idx = (((val - lo) / bin_w).floor() as usize).min(HISTOGRAM_BINS - 1);
            counts[idx] += 1;
        }
        let norm = 1.0 / ((reps.len() as f64) * bin_w);
        let mut max_d = 1e-9_f64;
        for (i, &c) in counts.iter().enumerate() {
            let center = lo + ((i as f64) + 0.5) * bin_w;
            let density = (c as f64) * norm;
            self.hist_bins[i] = (center, density);
            max_d = max_d.max(density);
        }
        self.hist_bin_width = bin_w;
        self.hist_max_density = max_d;
        if let Some(comp) = self.boot_comparison {
            let top = max_d * 1.08;
            let mk = |x: f64| [PlotPoint::new(x, 0.0), PlotPoint::new(x, top)];
            self.line_pct_lo = mk(comp.percentile.ci_lower);
            self.line_pct_hi = mk(comp.percentile.ci_upper);
            self.line_bca_lo = mk(comp.bca.ci_lower);
            self.line_bca_hi = mk(comp.bca.ci_upper);
            self.line_point_est = mk(comp.bca.point_estimate);
        }
    }
}

impl InteractiveTool for MonteCarloWorkbenchTool {
    fn name(&self) -> &'static str {
        "Monte Carlo & Bootstrap Workbench"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        self.confidence_level.is_finite() && self.importance_tilt.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.num_samples = self.num_samples.clamp(MIN_SAMPLES, MAX_SAMPLES);
        self.bootstrap_replicates = self
            .bootstrap_replicates
            .clamp(MIN_REPLICATES, MAX_REPLICATES);
        self.confidence_level = self.confidence_level.clamp(0.80, 0.99);
        self.importance_tilt = self.importance_tilt.clamp(0.0, 3.5);
        self.recompute();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&WorkbenchState {
            benchmark: self.benchmark,
            view_mode: self.view_mode,
            num_samples: self.num_samples,
            bootstrap_replicates: self.bootstrap_replicates,
            confidence_level: self.confidence_level,
            importance_tilt: self.importance_tilt,
            bootstrap_statistic: self.bootstrap_statistic,
            seed: self.seed,
        })
        .ok()
    }

    fn state_bounds(&self) -> crate::framework::ToolStateBounds {
        crate::framework::ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 8, 64)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: WorkbenchState =
            serde_json::from_str(state).map_err(|e| format!("Invalid workbench state: {e}"))?;
        parsed.validate()?;
        self.benchmark = parsed.benchmark;
        self.view_mode = parsed.view_mode;
        self.num_samples = parsed.num_samples;
        self.bootstrap_replicates = parsed.bootstrap_replicates;
        self.confidence_level = parsed.confidence_level;
        self.importance_tilt = parsed.importance_tilt;
        self.bootstrap_statistic = parsed.bootstrap_statistic;
        self.seed = parsed.seed;
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        self.render_controls_panel(ui);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.recompute();
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// [cite:clinical_trials_statistics]
