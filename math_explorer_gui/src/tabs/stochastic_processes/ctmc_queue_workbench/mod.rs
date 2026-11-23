#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Continuous-time Markov chain (`ContinuousMarkovChain`), inhomogeneous Poisson thinning,
//! and $M/M/c$ Erlang-C birth-death queueing workbench.

mod render;
pub mod state;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
use math_explorer::pure_math::statistics::poisson::{InhomogeneousPoisson, IntensityProfile};
use math_explorer::pure_math::statistics::queueing::{MmcQueue, QueueMetrics};
use math_explorer::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use state::{ContinuousModelMode, CtmcQueueState, PoissonVariant};

pub(super) const MIN_SERVERS: usize = 1;
pub(super) const MAX_SERVERS: usize = 6;
pub(super) const MIN_CAPACITY: usize = 8;
pub(super) const MAX_CAPACITY: usize = 36;
const MAX_TRAJ_EVENTS: usize = 512;

/// Interactive workbench for rate-preconditioned CTMC QR steady states, $M/M/c$ Erlang-C queues,
/// and Lewis-Shedler inhomogeneous Poisson thinning.
pub struct CtmcQueueWorkbenchTool {
    pub(super) mode: ContinuousModelMode,
    pub(super) poisson_variant: PoissonVariant,
    pub(super) rate_log10: f64,
    pub(super) forward_weight: f64,
    pub(super) backward_weight: f64,
    pub(super) servers: usize,
    pub(super) arrival_rate: f64,
    pub(super) service_rate: f64,
    pub(super) capacity: usize,
    pub(super) horizon: f64,
    pub(super) seed: u64,

    pub(super) ctmc_steady: Option<[f64; 3]>,
    pub(super) ctmc_empirical: [f64; 3],
    pub(super) queue_metrics: Option<QueueMetrics>,
    pub(super) queue_analytic_pk: Vec<f64>,
    pub(super) queue_ctmc_pk: Vec<f64>,
    pub(super) primary_curve: Vec<PlotPoint>,
    pub(super) secondary_curve: Vec<PlotPoint>,
    pub(super) expected_arrivals: f64,
    pub(super) realized_arrivals: usize,
    pub(super) error_msg: Option<String>,
    pub(super) last_compute_ms: f64,
}

impl Default for CtmcQueueWorkbenchTool {
    fn default() -> Self {
        let mut tool = Self {
            mode: ContinuousModelMode::SlowRateCtmc,
            poisson_variant: PoissonVariant::Sinusoidal,
            rate_log10: -4.0,
            forward_weight: 1.0,
            backward_weight: 2.0,
            servers: 3,
            arrival_rate: 4.5,
            service_rate: 2.0,
            capacity: 20,
            horizon: 25.0,
            seed: 42,
            ctmc_steady: None,
            ctmc_empirical: [0.0; 3],
            queue_metrics: None,
            queue_analytic_pk: Vec::new(),
            queue_ctmc_pk: Vec::new(),
            primary_curve: Vec::new(),
            secondary_curve: Vec::new(),
            expected_arrivals: 0.0,
            realized_arrivals: 0,
            error_msg: None,
            last_compute_ms: 0.0,
        };
        tool.recompute_all();
        tool
    }
}

impl CtmcQueueWorkbenchTool {
    /// Switches the active continuous-time regime and recomputes all curves and diagnostics.
    pub fn set_mode(&mut self, mode: ContinuousModelMode) {
        self.mode = mode;
        self.recompute_all();
    }

    pub(super) fn enforce_queue_stability(&mut self) {
        let max_lambda = (self.servers as f64 * self.service_rate * 0.95).clamp(0.2, 25.0);
        if self.arrival_rate > max_lambda {
            self.arrival_rate = max_lambda;
        }
        if self.capacity < self.servers.max(MIN_CAPACITY) {
            self.capacity = self.servers.max(MIN_CAPACITY);
        }
    }

    pub(super) fn recompute_all(&mut self) {
        let start = PlatformInstant::now();
        self.error_msg = None;
        self.enforce_queue_stability();

        match self.mode {
            ContinuousModelMode::SlowRateCtmc => self.recompute_slow_ctmc(),
            ContinuousModelMode::MmcBirthDeathQueue => self.recompute_mmc_queue(),
            ContinuousModelMode::InhomogeneousPoisson => self.recompute_poisson(),
        }
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn recompute_slow_ctmc(&mut self) {
        let scale = 10.0_f64.powf(self.rate_log10.clamp(-7.0, 0.0));
        let wf = scale * self.forward_weight.clamp(0.2, 5.0);
        let wb = scale * self.backward_weight.clamp(0.2, 5.0);
        let g = DMatrix::from_row_slice(
            3,
            3,
            &[-wf, wf, 0.0, wb, -(wf + wb), wf, 0.0, wb, -wb],
        );

        let chain = match ContinuousMarkovChain::new(g) {
            Ok(c) => c,
            Err(e) => {
                self.error_msg = Some(format!("CTMC error: {e}"));
                return;
            }
        };
        self.ctmc_steady = chain.steady_state().map(|v| [v[0], v[1], v[2]]);

        let sim_horizon = (450.0 / (wf + wb).max(1e-12)).max(1.0);
        let mut times = [0.0_f64; MAX_TRAJ_EVENTS];
        let mut states = [0_usize; MAX_TRAJ_EVENTS];
        let mut sojourns = [0.0_f64; 3];
        let mut rng = OxidizeRng::new(self.seed);

        if let Ok(events) =
            chain.simulate_gillespie_into(0, sim_horizon, &mut times, &mut states, &mut sojourns, &mut rng)
        {
            let total = sojourns.iter().sum::<f64>().max(f64::MIN_POSITIVE);
            self.ctmc_empirical = [
                sojourns[0] / total,
                sojourns[1] / total,
                sojourns[2] / total,
            ];
            self.primary_curve = build_step_curve(&times[..events], &states[..events], scale);
            self.secondary_curve.clear();
        }
    }

    fn recompute_mmc_queue(&mut self) {
        let queue = match MmcQueue::new(self.arrival_rate, self.service_rate, self.servers) {
            Ok(q) => q,
            Err(e) => {
                self.error_msg = Some(format!("Queue error: {e}"));
                return;
            }
        };
        let metrics = queue.metrics();
        self.queue_metrics = Some(metrics);

        let display_k = 12.min(self.capacity);
        self.queue_analytic_pk = (0..=display_k).map(|k| queue.state_probability(k)).collect();
        self.queue_ctmc_pk = queue
            .to_truncated_ctmc(self.capacity)
            .ok()
            .and_then(|ctmc| ctmc.steady_state())
            .map(|v| v.iter().take(display_k + 1).copied().collect())
            .unwrap_or_default();

        let mut times = [0.0_f64; MAX_TRAJ_EVENTS];
        let mut counts = [0_usize; MAX_TRAJ_EVENTS];
        let mut rng = OxidizeRng::new(self.seed);
        let events = queue.simulate_trajectory_into(0, self.horizon, &mut times, &mut counts, &mut rng);
        let n_pts = events.min(MAX_TRAJ_EVENTS);
        self.primary_curve = build_step_curve(&times[..n_pts], &counts[..n_pts], 1.0);
        self.secondary_curve = vec![
            PlotPoint::new(0.0, metrics.mean_in_system),
            PlotPoint::new(self.horizon, metrics.mean_in_system),
        ];
    }

    fn recompute_poisson(&mut self) {
        let profile = match self.poisson_variant {
            PoissonVariant::Sinusoidal => IntensityProfile::Sinusoidal {
                base_rate: self.arrival_rate,
                amplitude: 0.65 * self.arrival_rate,
                angular_freq: 0.8,
                phase: 0.0,
            },
            PoissonVariant::StepBurst => IntensityProfile::StepBurst {
                base_rate: (0.45 * self.arrival_rate).max(0.05),
                burst_rate: 1.85 * self.arrival_rate,
                burst_start: 0.30 * self.horizon,
                burst_end: 0.65 * self.horizon,
            },
        };
        let proc = match InhomogeneousPoisson::new(profile) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(format!("Poisson error: {e}"));
                return;
            }
        };
        self.expected_arrivals = proc.expected_count(self.horizon).unwrap_or(0.0);

        let mut arrivals = [0.0_f64; MAX_TRAJ_EVENTS];
        let mut rng = OxidizeRng::new(self.seed);
        let total = proc
            .simulate_arrivals_into(self.horizon, &mut arrivals, &mut rng)
            .unwrap_or(0);
        self.realized_arrivals = total;

        let kept = total.min(MAX_TRAJ_EVENTS);
        self.primary_curve = build_counting_curve(&arrivals[..kept], self.horizon);
        self.secondary_curve = (0..=80)
            .map(|i| {
                let t = self.horizon * (i as f64 / 80.0);
                PlotPoint::new(t, profile.integrated_mean(t))
            })
            .collect();
    }
}

fn build_step_curve(times: &[f64], states: &[usize], time_scale: f64) -> Vec<PlotPoint> {
    let mut pts = Vec::with_capacity(times.len() * 2);
    for i in 0..times.len() {
        let x = times[i] * time_scale;
        let y = states[i] as f64;
        if i > 0 {
            pts.push(PlotPoint::new(x, states[i - 1] as f64));
        }
        pts.push(PlotPoint::new(x, y));
    }
    pts
}

fn build_counting_curve(arrivals: &[f64], horizon: f64) -> Vec<PlotPoint> {
    let mut pts = Vec::with_capacity(arrivals.len() * 2 + 2);
    pts.push(PlotPoint::new(0.0, 0.0));
    for (idx, &t) in arrivals.iter().enumerate() {
        pts.push(PlotPoint::new(t, idx as f64));
        pts.push(PlotPoint::new(t, (idx + 1) as f64));
    }
    pts.push(PlotPoint::new(horizon, arrivals.len() as f64));
    pts
}

impl InteractiveTool for CtmcQueueWorkbenchTool {
    fn name(&self) -> &'static str {
        "CTMC, Poisson & M/M/c Queues"
    }

    fn domain(&self) -> &'static str {
        "stochastic_processes"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.rate_log10,
            self.forward_weight,
            self.backward_weight,
            self.arrival_rate,
            self.service_rate,
            self.horizon,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.rate_log10 = self.rate_log10.clamp(-7.0, 0.0);
        self.forward_weight = self.forward_weight.clamp(0.2, 5.0);
        self.backward_weight = self.backward_weight.clamp(0.2, 5.0);
        self.servers = self.servers.clamp(MIN_SERVERS, MAX_SERVERS);
        self.service_rate = self.service_rate.clamp(0.5, 10.0);
        self.arrival_rate = self.arrival_rate.clamp(0.2, 25.0);
        self.capacity = self.capacity.clamp(MIN_CAPACITY, MAX_CAPACITY);
        self.horizon = self.horizon.clamp(5.0, 100.0);
        self.recompute_all();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CtmcQueueState {
            mode: self.mode,
            poisson_variant: self.poisson_variant,
            rate_log10: self.rate_log10,
            forward_weight: self.forward_weight,
            backward_weight: self.backward_weight,
            servers: self.servers,
            arrival_rate: self.arrival_rate,
            service_rate: self.service_rate,
            capacity: self.capacity,
            horizon: self.horizon,
            seed: self.seed,
        })
        .ok()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 8, 64)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: CtmcQueueState =
            serde_json::from_str(state).map_err(|e| format!("Invalid CTMC/Queue state: {e}"))?;
        parsed.validate()?;

        self.mode = parsed.mode;
        self.poisson_variant = parsed.poisson_variant;
        self.rate_log10 = parsed.rate_log10;
        self.forward_weight = parsed.forward_weight;
        self.backward_weight = parsed.backward_weight;
        self.servers = parsed.servers;
        self.arrival_rate = parsed.arrival_rate;
        self.service_rate = parsed.service_rate;
        self.capacity = parsed.capacity;
        self.horizon = parsed.horizon;
        self.seed = parsed.seed;
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
        self.recompute_all();
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
