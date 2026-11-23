//! Parrondo's Paradox interactive Markov ratchet laboratory for `FinancialMathTab`.

mod plots;
mod state;
#[cfg(test)]
mod tests;
mod ui;

pub use state::{KellyEdgeBadge, ParrondoViewMode};
use state::ParrondoToolState;

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::markov::parrondo::{
    ALLOWED_MODULI, MAX_MODULUS, ParrondoEnsemble, ParrondoGame, ParrondoParams,
    ParrondoPhaseGridConfig, ParrondoPhasePoint, ParrondoReport, ParrondoSchedule,
    scan_parrondo_phase_region, solve_parrondo_ratchet,
};
use oxidize_core::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;

const DEFAULT_WALKERS: usize = 256;
const DEFAULT_STEPS: usize = 180;
const DEFAULT_SEED: u64 = 42;
pub(super) const MIN_WALKERS: usize = 32;
pub(super) const MAX_WALKERS: usize = 2048;
pub(super) const MIN_STEPS: usize = 20;
pub(super) const MAX_STEPS: usize = 500;

/// Interactive Markov ratchet lab exploring Harmer-Abbott / Parrondo's paradox.
pub struct ParrondoParadoxTool {
    pub(super) p_a_base: f64,
    pub(super) p_b0_base: f64,
    pub(super) p_b1_base: f64,
    pub(super) epsilon: f64,
    pub(super) modulus: usize,
    pub(super) schedule_input: String,
    pub(super) active_schedule: ParrondoSchedule,
    pub(super) num_walkers: usize,
    pub(super) num_steps: usize,
    pub(super) seed: u64,
    pub(super) view_mode: ParrondoViewMode,

    ensemble_a: ParrondoEnsemble,
    ensemble_b: ParrondoEnsemble,
    ensemble_mix: ParrondoEnsemble,
    ensemble_schedule: ParrondoEnsemble,
    ensemble_cap_walkers: usize,
    ensemble_cap_steps: usize,

    pub(super) report: Option<ParrondoReport>,
    pub(super) kelly_a: KellyEdgeBadge,
    pub(super) kelly_b: KellyEdgeBadge,
    pub(super) kelly_mix: KellyEdgeBadge,
    pub(super) kelly_schedule: KellyEdgeBadge,

    pub(super) traj_a: Vec<PlotPoint>,
    pub(super) traj_b: Vec<PlotPoint>,
    pub(super) traj_mix: Vec<PlotPoint>,
    pub(super) traj_schedule: Vec<PlotPoint>,
    pub(super) drift_a: Vec<PlotPoint>,
    pub(super) drift_b: Vec<PlotPoint>,
    pub(super) drift_mix: Vec<PlotPoint>,
    pub(super) drift_schedule: Vec<PlotPoint>,
    pub(super) emp_occ_schedule: [f64; MAX_MODULUS],

    pub(super) phase_points: Vec<ParrondoPhasePoint>,
    pub(super) paradox_scatter: Vec<PlotPoint>,
    pub(super) non_paradox_scatter: Vec<PlotPoint>,
    pub(super) boundary_game_a: Vec<PlotPoint>,
    pub(super) boundary_game_b: Vec<PlotPoint>,
    pub(super) boundary_schedule: Vec<PlotPoint>,

    pub(super) schedule_error: Option<String>,
    pub(super) error_msg: Option<String>,
    pub(super) last_compute_ms: f64,
}

impl Default for ParrondoParadoxTool {
    fn default() -> Self {
        let canonical = ParrondoParams::canonical();
        let mut tool = Self {
            p_a_base: canonical.p_a_base(),
            p_b0_base: canonical.p_b0_base(),
            p_b1_base: canonical.p_b1_base(),
            epsilon: canonical.epsilon(),
            modulus: canonical.modulus(),
            schedule_input: "ABB".to_string(),
            active_schedule: ParrondoSchedule::abb(),
            num_walkers: DEFAULT_WALKERS,
            num_steps: DEFAULT_STEPS,
            seed: DEFAULT_SEED,
            view_mode: ParrondoViewMode::SplitOverview,
            ensemble_a: ParrondoEnsemble::with_capacity(DEFAULT_WALKERS, MAX_STEPS)
                .expect("valid default capacity"),
            ensemble_b: ParrondoEnsemble::with_capacity(DEFAULT_WALKERS, MAX_STEPS)
                .expect("valid default capacity"),
            ensemble_mix: ParrondoEnsemble::with_capacity(DEFAULT_WALKERS, MAX_STEPS)
                .expect("valid default capacity"),
            ensemble_schedule: ParrondoEnsemble::with_capacity(DEFAULT_WALKERS, MAX_STEPS)
                .expect("valid default capacity"),
            ensemble_cap_walkers: DEFAULT_WALKERS,
            ensemble_cap_steps: MAX_STEPS,
            report: None,
            kelly_a: KellyEdgeBadge::NoEdge { win_prob: 0.495 },
            kelly_b: KellyEdgeBadge::NoEdge { win_prob: 0.495 },
            kelly_mix: KellyEdgeBadge::NoEdge { win_prob: 0.5 },
            kelly_schedule: KellyEdgeBadge::NoEdge { win_prob: 0.5 },
            traj_a: Vec::with_capacity(MAX_STEPS + 1),
            traj_b: Vec::with_capacity(MAX_STEPS + 1),
            traj_mix: Vec::with_capacity(MAX_STEPS + 1),
            traj_schedule: Vec::with_capacity(MAX_STEPS + 1),
            drift_a: Vec::with_capacity(2),
            drift_b: Vec::with_capacity(2),
            drift_mix: Vec::with_capacity(2),
            drift_schedule: Vec::with_capacity(2),
            emp_occ_schedule: [0.0; MAX_MODULUS],
            phase_points: Vec::new(),
            paradox_scatter: Vec::new(),
            non_paradox_scatter: Vec::new(),
            boundary_game_a: Vec::new(),
            boundary_game_b: Vec::new(),
            boundary_schedule: Vec::new(),
            schedule_error: None,
            error_msg: None,
            last_compute_ms: 0.0,
        };
        tool.recompute_all();
        tool
    }
}

impl ParrondoParadoxTool {
    /// Resets all parameters and schedule selection to the canonical Harmer-Abbott configuration.
    pub fn reset_defaults(&mut self) {
        let canonical = ParrondoParams::canonical();
        self.p_a_base = canonical.p_a_base();
        self.p_b0_base = canonical.p_b0_base();
        self.p_b1_base = canonical.p_b1_base();
        self.epsilon = canonical.epsilon();
        self.modulus = canonical.modulus();
        self.schedule_input = "ABB".to_string();
        self.active_schedule = ParrondoSchedule::abb();
        self.num_walkers = DEFAULT_WALKERS;
        self.num_steps = DEFAULT_STEPS;
        self.seed = DEFAULT_SEED;
        self.view_mode = ParrondoViewMode::SplitOverview;
        self.schedule_error = None;
        self.recompute_all();
    }

    pub(super) fn apply_schedule_text(&mut self, candidate: &str) -> bool {
        let normalized = candidate.trim().to_ascii_uppercase();
        match ParrondoSchedule::parse(&normalized) {
            Ok(parsed) => {
                self.schedule_input = normalized;
                self.active_schedule = parsed;
                self.schedule_error = None;
                true
            }
            Err(err) => {
                self.schedule_error = Some(err.to_string());
                false
            }
        }
    }

    fn ensure_ensemble_capacity(&mut self) {
        let req_steps = self.num_steps.max(MAX_STEPS);
        if self.ensemble_cap_walkers != self.num_walkers || self.ensemble_cap_steps < req_steps {
            if let (Ok(ea), Ok(eb), Ok(em), Ok(es)) = (
                ParrondoEnsemble::with_capacity(self.num_walkers, req_steps),
                ParrondoEnsemble::with_capacity(self.num_walkers, req_steps),
                ParrondoEnsemble::with_capacity(self.num_walkers, req_steps),
                ParrondoEnsemble::with_capacity(self.num_walkers, req_steps),
            ) {
                self.ensemble_a = ea;
                self.ensemble_b = eb;
                self.ensemble_mix = em;
                self.ensemble_schedule = es;
                self.ensemble_cap_walkers = self.num_walkers;
                self.ensemble_cap_steps = req_steps;
            }
        }
    }

    pub(super) fn recompute_all(&mut self) {
        let start = PlatformInstant::now();
        self.error_msg = None;
        let params = match ParrondoParams::new(
            self.modulus,
            self.p_a_base,
            self.p_b0_base,
            self.p_b1_base,
            self.epsilon,
        ) {
            Ok(p) => p,
            Err(err) => {
                self.error_msg = Some(format!("Parameter error: {err}"));
                return;
            }
        };
        let report = match solve_parrondo_ratchet(&params, &self.active_schedule) {
            Ok(r) => r,
            Err(err) => {
                self.error_msg = Some(format!("Solver error: {err}"));
                return;
            }
        };
        self.kelly_a = KellyEdgeBadge::evaluate(report.gain_a);
        self.kelly_b = KellyEdgeBadge::evaluate(report.gain_b);
        self.kelly_mix = KellyEdgeBadge::evaluate(report.gain_mix);
        self.kelly_schedule = KellyEdgeBadge::evaluate(report.gain_schedule);
        self.populate_drift_lines(&report);
        self.report = Some(report);
        self.run_ensembles(&params);
        self.rebuild_phase_wedge();
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    pub(super) fn recompute_ensembles_only(&mut self) {
        let start = PlatformInstant::now();
        if let Ok(params) = ParrondoParams::new(
            self.modulus,
            self.p_a_base,
            self.p_b0_base,
            self.p_b1_base,
            self.epsilon,
        ) {
            self.run_ensembles(&params);
            if let Some(ref report) = self.report.clone() {
                self.populate_drift_lines(report);
            }
        }
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn populate_drift_lines(&mut self, report: &ParrondoReport) {
        let horizon = self.num_steps as f64;
        for (buf, slope) in [
            (&mut self.drift_a, report.gain_a),
            (&mut self.drift_b, report.gain_b),
            (&mut self.drift_mix, report.gain_mix),
            (&mut self.drift_schedule, report.gain_schedule),
        ] {
            buf.clear();
            buf.push(PlotPoint::new(0.0, 0.0));
            buf.push(PlotPoint::new(horizon, horizon * slope));
        }
    }

    fn run_ensembles(&mut self, params: &ParrondoParams) {
        self.ensure_ensemble_capacity();
        let mut rng_a = OxidizeRng::new(self.seed ^ 0x000A_11CE_0001);
        let mut rng_b = OxidizeRng::new(self.seed ^ 0x000B_22DF_0002);
        let mut rng_mix = OxidizeRng::new(self.seed ^ 0x000C_33E0_0003);
        let mut rng_sched = OxidizeRng::new(self.seed);

        self.ensemble_a
            .simulate_pure_game(params, ParrondoGame::GameA, self.num_steps, &mut rng_a);
        self.ensemble_b
            .simulate_pure_game(params, ParrondoGame::GameB, self.num_steps, &mut rng_b);
        self.ensemble_mix
            .simulate_mixture(params, self.num_steps, &mut rng_mix);
        self.ensemble_schedule.simulate_schedule(
            params,
            &self.active_schedule,
            self.num_steps,
            &mut rng_sched,
        );

        fill_plot_points(&mut self.traj_a, self.ensemble_a.mean_trajectory());
        fill_plot_points(&mut self.traj_b, self.ensemble_b.mean_trajectory());
        fill_plot_points(&mut self.traj_mix, self.ensemble_mix.mean_trajectory());
        fill_plot_points(
            &mut self.traj_schedule,
            self.ensemble_schedule.mean_trajectory(),
        );

        self.emp_occ_schedule = [0.0; MAX_MODULUS];
        let counts = self.ensemble_schedule.occupancy_counts(params.modulus());
        let total: u64 = counts.iter().sum();
        if total > 0 {
            let inv = 1.0 / (total as f64);
            for (idx, &c) in counts.iter().enumerate() {
                self.emp_occ_schedule[idx] = (c as f64) * inv;
            }
        }
    }

    fn rebuild_phase_wedge(&mut self) {
        let cfg = ParrondoPhaseGridConfig {
            modulus: self.modulus,
            p_b0_base: self.p_b0_base,
            p_b1_base: self.p_b1_base,
            p_a_min: 0.42,
            p_a_max: 0.51,
            epsilon_min: -0.02,
            epsilon_max: 0.05,
            p_a_steps: 29,
            epsilon_steps: 25,
        };
        self.paradox_scatter.clear();
        self.non_paradox_scatter.clear();
        if let Ok(pts) = scan_parrondo_phase_region(&cfg, &self.active_schedule) {
            for pt in &pts {
                let plot_pt = PlotPoint::new(pt.p_a, pt.epsilon);
                if pt.is_paradox {
                    self.paradox_scatter.push(plot_pt);
                } else {
                    self.non_paradox_scatter.push(plot_pt);
                }
            }
            plots::extract_phase_boundaries(
                &cfg,
                &pts,
                &mut self.boundary_game_a,
                &mut self.boundary_game_b,
                &mut self.boundary_schedule,
            );
            self.phase_points = pts;
        }
    }
}

fn fill_plot_points(dst: &mut Vec<PlotPoint>, values: &[f64]) {
    dst.clear();
    for (idx, &val) in values.iter().enumerate() {
        dst.push(PlotPoint::new(idx as f64, val));
    }
}

impl InteractiveTool for ParrondoParadoxTool {
    fn name(&self) -> &'static str {
        "Parrondo's Paradox"
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
        [self.p_a_base, self.p_b0_base, self.p_b1_base, self.epsilon]
            .into_iter()
            .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) -> bool {
        if !ALLOWED_MODULI.contains(&self.modulus) {
            self.modulus = 3;
        }
        self.epsilon = self.epsilon.clamp(-0.02, 0.05);
        let lo = (0.02 + self.epsilon).max(0.02);
        let hi = (0.98 + self.epsilon).min(0.98);
        self.p_a_base = self.p_a_base.clamp(lo, hi);
        self.p_b0_base = self.p_b0_base.clamp(lo, hi);
        self.p_b1_base = self.p_b1_base.clamp(lo, hi);
        self.num_walkers = self.num_walkers.clamp(MIN_WALKERS, MAX_WALKERS);
        self.num_steps = self.num_steps.clamp(MIN_STEPS, MAX_STEPS);
        self.recompute_all();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&ParrondoToolState {
            p_a_base: self.p_a_base,
            p_b0_base: self.p_b0_base,
            p_b1_base: self.p_b1_base,
            epsilon: self.epsilon,
            modulus: self.modulus,
            schedule: self.active_schedule.as_string(),
            num_walkers: self.num_walkers,
            num_steps: self.num_steps,
            seed: self.seed,
            view_mode: self.view_mode,
        })
        .ok()
    }

    fn state_bounds(&self) -> crate::framework::ToolStateBounds {
        crate::framework::ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 8, 64)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: ParrondoToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Parrondo state: {e}"))?;
        if !(MIN_WALKERS..=MAX_WALKERS).contains(&parsed.num_walkers)
            || !(MIN_STEPS..=MAX_STEPS).contains(&parsed.num_steps)
            || !(-0.05..=0.10).contains(&parsed.epsilon)
        {
            return Err("Parrondo simulation bounds exceeded".to_string());
        }
        let sched = ParrondoSchedule::parse(&parsed.schedule)
            .map_err(|e| format!("Invalid Parrondo schedule: {e}"))?;
        ParrondoParams::new(
            parsed.modulus,
            parsed.p_a_base,
            parsed.p_b0_base,
            parsed.p_b1_base,
            parsed.epsilon,
        )
        .map_err(|e| format!("Invalid Parrondo parameters: {e}"))?;

        self.p_a_base = parsed.p_a_base;
        self.p_b0_base = parsed.p_b0_base;
        self.p_b1_base = parsed.p_b1_base;
        self.epsilon = parsed.epsilon;
        self.modulus = parsed.modulus;
        self.schedule_input = sched.as_string();
        self.active_schedule = sched;
        self.num_walkers = parsed.num_walkers;
        self.num_steps = parsed.num_steps;
        self.seed = parsed.seed;
        self.view_mode = parsed.view_mode;
        self.schedule_error = None;
        self.recompute_all();
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
        self.recompute_ensembles_only();
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
