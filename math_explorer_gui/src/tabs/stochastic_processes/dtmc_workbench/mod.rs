#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Discrete-time Markov chain (`MarkovChain`) and non-stationary `TransitionTensor` workbench.

mod render;
pub mod state;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::markov::{
    CommunicatingClass, MarkovChain, StateType, TimeIndex, TransitionTensor,
};
use math_explorer::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use state::{DtmcModelPreset, DtmcWorkbenchState};

pub(super) const MIN_MC_STEPS: usize = 50;
pub(super) const MAX_MC_STEPS: usize = 5_000;
const DEFAULT_MC_STEPS: usize = 1_200;
const DEFAULT_SEED: u64 = 42;

/// Interactive workbench for DTMC communicating classes, QR stationary nullspaces,
/// `B = N * R` limiting distributions, hitting times, and `TransitionTensor` slices.
pub struct DtmcTensorWorkbenchTool {
    pub(super) preset: DtmcModelPreset,
    pub(super) row0_stay: f64,
    pub(super) row0_to1: f64,
    pub(super) row1_to0: f64,
    pub(super) row1_stay: f64,
    pub(super) shot_clock_time: f64,
    pub(super) initial_state: usize,
    pub(super) target_state: usize,
    pub(super) mc_steps: usize,
    pub(super) seed: u64,

    pub(super) active_matrix: DMatrix<f64>,
    pub(super) classes: Vec<CommunicatingClass>,
    pub(super) chain_period: Option<usize>,
    pub(super) is_aperiodic: bool,
    pub(super) stationary: Option<[f64; 3]>,
    pub(super) limiting: Option<[f64; 3]>,
    pub(super) empirical_freq: [f64; 3],
    pub(super) hitting_times: Option<[f64; 3]>,
    pub(super) tensor_curve_s1: Vec<PlotPoint>,
    pub(super) tensor_curve_s2: Vec<PlotPoint>,
    pub(super) error_msg: Option<String>,
    pub(super) last_compute_ms: f64,
}

impl Default for DtmcTensorWorkbenchTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: DtmcModelPreset::PeriodTwoWalk,
            row0_stay: 0.50,
            row0_to1: 0.475,
            row1_to0: 0.50,
            row1_stay: 0.00,
            shot_clock_time: 12.0,
            initial_state: 0,
            target_state: 2,
            mc_steps: DEFAULT_MC_STEPS,
            seed: DEFAULT_SEED,
            active_matrix: DMatrix::identity(3, 3),
            classes: Vec::new(),
            chain_period: None,
            is_aperiodic: false,
            stationary: None,
            limiting: None,
            empirical_freq: [0.0; 3],
            hitting_times: None,
            tensor_curve_s1: Vec::new(),
            tensor_curve_s2: Vec::new(),
            error_msg: None,
            last_compute_ms: 0.0,
        };
        tool.recompute_all();
        tool
    }
}

impl DtmcTensorWorkbenchTool {
    /// Applies a fixed canonical DTMC / `TransitionTensor` preset and recomputes diagnostics.
    pub fn apply_model_preset(&mut self, preset: DtmcModelPreset) {
        self.preset = preset;
        match preset {
            DtmcModelPreset::ErgodicChain => {
                self.row0_stay = 0.55;
                self.row0_to1 = 0.30;
                self.row1_to0 = 0.25;
                self.row1_stay = 0.45;
            }
            DtmcModelPreset::PeriodTwoWalk => {
                self.row0_stay = 0.0;
                self.row0_to1 = 1.0;
                self.row1_to0 = 0.50;
                self.row1_stay = 0.0;
            }
            DtmcModelPreset::AsymmetricAbsorbing => {
                self.row0_stay = 0.50;
                self.row0_to1 = 0.475;
                self.row1_to0 = 0.0;
                self.row1_stay = 1.0;
            }
            DtmcModelPreset::ShotClockTensor => {
                self.shot_clock_time = 12.0;
                self.row0_stay = 0.60;
                self.row0_to1 = 0.30;
            }
        }
        self.recompute_all();
    }

    pub(super) fn recompute_all(&mut self) {
        let start = PlatformInstant::now();
        self.error_msg = None;

        let (matrix, state_types) = match self.build_chain_definition() {
            Ok(pair) => pair,
            Err(err) => {
                self.error_msg = Some(err);
                return;
            }
        };
        self.active_matrix = matrix.clone();

        let chain = match MarkovChain::new(matrix, state_types) {
            Ok(c) => c,
            Err(err) => {
                self.error_msg = Some(format!("MarkovChain error: {err}"));
                return;
            }
        };

        self.classes = chain.communicating_classes();
        self.chain_period = chain.period();
        self.is_aperiodic = chain.is_aperiodic();
        self.stationary = chain.stationary_distribution().map(|v| vec3_to_arr(&v));
        self.limiting = chain
            .limiting_distribution_from_state(self.initial_state.min(2))
            .map(|v| vec3_to_arr(&v));
        self.hitting_times = chain
            .mean_hitting_times(self.target_state.min(2))
            .ok()
            .map(|v| vec3_to_arr(&v));

        let mut visits = [0_usize; 3];
        let mut rng = OxidizeRng::new(self.seed);
        if chain
            .simulate_visits_into(self.initial_state.min(2), self.mc_steps, &mut visits, &mut rng)
            .is_ok()
            && self.mc_steps > 0
        {
            let inv = 1.0 / self.mc_steps as f64;
            self.empirical_freq = [
                visits[0] as f64 * inv,
                visits[1] as f64 * inv,
                visits[2] as f64 * inv,
            ];
        }
        self.rebuild_tensor_curves();
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn build_chain_definition(&self) -> Result<(DMatrix<f64>, Vec<StateType>), String> {
        match self.preset {
            DtmcModelPreset::ErgodicChain => {
                let (r00, r01, r02) = normalize_row3(self.row0_stay, self.row0_to1);
                let (r10, r11, r12) = normalize_row3(self.row1_to0, self.row1_stay);
                let mat = DMatrix::from_row_slice(3, 3, &[r00, r01, r02, r10, r11, r12, 0.25, 0.35, 0.40]);
                Ok((mat, vec![StateType::Transient; 3]))
            }
            DtmcModelPreset::PeriodTwoWalk => {
                let p_left = self.row1_to0.clamp(0.05, 0.95);
                let mat = DMatrix::from_row_slice(
                    3,
                    3,
                    &[0.0, 1.0, 0.0, p_left, 0.0, 1.0 - p_left, 0.0, 1.0, 0.0],
                );
                Ok((mat, vec![StateType::Transient; 3]))
            }
            DtmcModelPreset::AsymmetricAbsorbing => {
                let (r00, r01, r02) = normalize_absorbing_row(self.row0_stay, self.row0_to1);
                let mat = DMatrix::from_row_slice(
                    3,
                    3,
                    &[r00, r01, r02, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                );
                Ok((
                    mat,
                    vec![
                        StateType::Transient,
                        StateType::Absorbing,
                        StateType::Absorbing,
                    ],
                ))
            }
            DtmcModelPreset::ShotClockTensor => {
                let tensor = build_shot_clock_tensor()?;
                let t = TimeIndex::new(self.shot_clock_time.clamp(3.0, 24.0))
                    .map_err(|e| format!("Invalid shot-clock time: {e}"))?;
                let mat = tensor
                    .transition_matrix_at(t)
                    .map_err(|e| format!("Tensor interpolation error: {e}"))?;
                Ok((
                    mat,
                    vec![
                        StateType::Transient,
                        StateType::Absorbing,
                        StateType::Absorbing,
                    ],
                ))
            }
        }
    }

    fn rebuild_tensor_curves(&mut self) {
        self.tensor_curve_s1.clear();
        self.tensor_curve_s2.clear();
        let Ok(tensor) = build_shot_clock_tensor() else {
            return;
        };
        let states = vec![
            StateType::Transient,
            StateType::Absorbing,
            StateType::Absorbing,
        ];
        for step in 0..=21 {
            let t_val = 3.0 + step as f64;
            let lim_opt = TimeIndex::new(t_val)
                .ok()
                .and_then(|ti| tensor.transition_matrix_at(ti).ok())
                .and_then(|mat| MarkovChain::new(mat, states.clone()).ok())
                .and_then(|chain| chain.limiting_distribution_from_state(0));
            if let Some(lim) = lim_opt {
                self.tensor_curve_s1.push(PlotPoint::new(t_val, lim[1]));
                self.tensor_curve_s2.push(PlotPoint::new(t_val, lim[2]));
            }
        }
    }
}

fn vec3_to_arr(v: &DVector<f64>) -> [f64; 3] {
    [
        v.get(0).copied().unwrap_or(0.0),
        v.get(1).copied().unwrap_or(0.0),
        v.get(2).copied().unwrap_or(0.0),
    ]
}

pub(super) fn normalize_row3(a: f64, b: f64) -> (f64, f64, f64) {
    let a_c = a.clamp(0.0, 0.96);
    let b_c = b.clamp(0.0, (0.98 - a_c).max(0.0));
    let rem = (1.0 - a_c - b_c).max(0.0);
    let sum = a_c + b_c + rem;
    (a_c / sum, b_c / sum, rem / sum)
}

fn normalize_absorbing_row(stay: f64, to_abs1: f64) -> (f64, f64, f64) {
    let s = stay.clamp(0.0, 0.90);
    let rem_budget = 1.0 - s;
    let a1 = to_abs1.clamp(0.0, rem_budget);
    let a2 = (rem_budget - a1).max(0.0);
    (s, a1, a2)
}

fn build_shot_clock_tensor() -> Result<TransitionTensor<f64>, String> {
    let t0 = TimeIndex::new(0.0).map_err(|e| e.to_string())?;
    let t24 = TimeIndex::new(24.0).map_err(|e| e.to_string())?;
    let mut tensor = TransitionTensor::new(3, t0, t24);

    let slices = [
        (3.0, [0.20, 0.48, 0.32]),
        (12.0, [0.55, 0.34, 0.11]),
        (24.0, [0.80, 0.16, 0.04]),
    ];
    for (sec, r0) in slices {
        let ti = TimeIndex::new(sec).map_err(|e| e.to_string())?;
        let mat = DMatrix::from_row_slice(
            3,
            3,
            &[r0[0], r0[1], r0[2], 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        tensor.add_time_slice(ti, mat).map_err(|e| e.to_string())?;
    }
    Ok(tensor)
}

impl InteractiveTool for DtmcTensorWorkbenchTool {
    fn name(&self) -> &'static str {
        "DTMC & Transition Tensor"
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
            self.row0_stay,
            self.row0_to1,
            self.row1_to0,
            self.row1_stay,
            self.shot_clock_time,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.row0_stay = self.row0_stay.clamp(0.0, 0.90);
        self.row0_to1 = self.row0_to1.clamp(0.0, 1.0 - self.row0_stay);
        self.row1_to0 = self.row1_to0.clamp(0.0, 0.95);
        self.row1_stay = self.row1_stay.clamp(0.0, (0.98 - self.row1_to0).max(0.0));
        self.shot_clock_time = self.shot_clock_time.clamp(3.0, 24.0);
        self.initial_state = self.initial_state.min(2);
        self.target_state = self.target_state.min(2);
        self.mc_steps = self.mc_steps.clamp(MIN_MC_STEPS, MAX_MC_STEPS);
        self.recompute_all();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&DtmcWorkbenchState {
            preset: self.preset,
            row0_stay: self.row0_stay,
            row0_to1: self.row0_to1,
            row1_to0: self.row1_to0,
            row1_stay: self.row1_stay,
            shot_clock_time: self.shot_clock_time,
            initial_state: self.initial_state,
            target_state: self.target_state,
            mc_steps: self.mc_steps,
            seed: self.seed,
        })
        .ok()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(crate::url_state::MAX_DECODED_URL_STATE_BYTES, 8, 64)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: DtmcWorkbenchState =
            serde_json::from_str(state).map_err(|e| format!("Invalid DTMC state: {e}"))?;
        parsed.validate()?;

        self.preset = parsed.preset;
        self.row0_stay = parsed.row0_stay;
        self.row0_to1 = parsed.row0_to1;
        self.row1_to0 = parsed.row1_to0;
        self.row1_stay = parsed.row1_stay;
        self.shot_clock_time = parsed.shot_clock_time;
        self.initial_state = parsed.initial_state;
        self.target_state = parsed.target_state;
        self.mc_steps = parsed.mc_steps;
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
