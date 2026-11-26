#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State model, parameter validation, and `InteractiveTool` implementation for [`FractranMachineTool`].

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::math_types::Integer;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranProgram, FractranState, FIRST_64_PRIMES,
};
use math_explorer::pure_math::algorithmic_information::geometry::{distance, Point2D};
use math_explorer::pure_math::algorithmic_information::kolmogorov::{
    bdm_complexity, exponents_to_bits, integer_to_bits, k_floor_r, lz76_complexity,
    prefix_kolmogorov_approx, shannon_entropy_bits,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Hard per-frame stepping cap preventing browser/UI stalls on `wasm32` and desktop.
pub const MAX_STEPS_PER_FRAME: usize = 256;

/// Maximum retained trajectory points in the rolling compressibility chart.
pub const MAX_TRAJECTORY_POINTS: usize = 1_024;

/// Maximum initial 64-bit integer accepted by UI and preset state deserialization.
pub const MAX_INITIAL_VALUE: u64 = 1_000_000_000;

/// Maximum Warm-up steps executed deterministically upon preset hydration.
pub const MAX_WARMUP_STEPS: usize = 512;

/// Maximum byte length accepted for custom fraction rule strings.
pub const MAX_CUSTOM_FRACTIONS_LEN: usize = 2_048;

/// Built-in or user-defined FRACTRAN program selection.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FractranPreset {
    /// Conway's 14-fraction `PRIMEGAME` streaming $2^2, 2^3, 2^5, 2^7, \dots$
    Primegame,
    /// 9-fraction Fibonacci register machine mapping $2^1 5^n \to 2^{F_{n+1}} 3^{F_n}$.
    Fibonacci,
    /// 6-fraction Minsky multiplier mapping $2^a 3^b \to 5^{a \cdot b}$.
    Multiplier,
    /// 1-fraction Minsky adder mapping $2^a 3^b \to 3^{a+b}$.
    Adder,
    /// User-editable fraction list validated over the first 64 primes ($p \le 311$).
    Custom,
}

impl FractranPreset {
    /// Short display label for UI buttons and selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Primegame => "PRIMEGAME",
            Self::Fibonacci => "Fibonacci",
            Self::Multiplier => "Multiplier",
            Self::Adder => "Adder",
            Self::Custom => "Custom",
        }
    }

    /// Canonical comma-separated fraction string for this preset.
    #[must_use]
    pub const fn canonical_fractions_text(self) -> &'static str {
        match self {
            Self::Primegame | Self::Custom => {
                "17/91, 78/85, 19/51, 23/38, 29/33, 77/29, 95/23, 77/19, 1/17, 11/13, 13/11, 15/2, 1/7, 55/1"
            }
            Self::Fibonacci => "1547/22, 11/17, 119/33, 19/11, 46/133, 19/23, 69/247, 1/19, 11/5",
            Self::Multiplier => "455/33, 11/13, 1/11, 3/7, 11/2, 1/3",
            Self::Adder => "3/2",
        }
    }

    /// Recommended initial state integer $N_0$ demonstrating each canonical program.
    #[must_use]
    pub const fn default_initial_value(self) -> u64 {
        match self {
            Self::Primegame | Self::Custom => 2,
            Self::Fibonacci => 2 * 3_125, // 2^1 * 5^5 -> 2^8 * 3^5
            Self::Multiplier => 8 * 27,   // 2^3 * 3^3 -> 5^9
            Self::Adder => 16 * 81,       // 2^4 * 3^4 -> 3^8
        }
    }
}

/// Bounded serializable configuration for [`FractranMachineTool`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FractranToolConfig {
    /// Playback speed multiplier (`0.25..=16.0`).
    pub playback_rate: f64,
    /// Transitions executed per frame while playing (`1..=256`).
    pub steps_per_frame: usize,
    /// Deterministic initial transitions pre-stepped on load (`0..=512`).
    pub warmup_steps: usize,
    /// Initial integer encoded into the 64 prime registers (`1..=1_000_000_000`).
    pub initial_value: u64,
    /// Active canonical or custom program mode.
    pub preset: FractranPreset,
    /// Editable fraction list used when `preset == FractranPreset::Custom`.
    pub custom_fractions: String,
}

impl Default for FractranToolConfig {
    fn default() -> Self {
        Self {
            playback_rate: 1.0,
            steps_per_frame: 16,
            warmup_steps: 24,
            initial_value: 2,
            preset: FractranPreset::Primegame,
            custom_fractions: FractranPreset::Primegame
                .canonical_fractions_text()
                .to_string(),
        }
    }
}

impl FractranToolConfig {
    /// Validates numeric ranges, finiteness, prime factor caps, and custom fraction syntax.
    pub fn validate(&self) -> Result<(FractranProgram, FractranState), String> {
        if !self.playback_rate.is_finite() || !(0.25..=16.0).contains(&self.playback_rate) {
            return Err("playback_rate must be finite within [0.25, 16.0]".to_string());
        }
        if self.steps_per_frame == 0 || self.steps_per_frame > MAX_STEPS_PER_FRAME {
            return Err(format!(
                "steps_per_frame must be within 1..={MAX_STEPS_PER_FRAME}"
            ));
        }
        if self.warmup_steps > MAX_WARMUP_STEPS {
            return Err(format!("warmup_steps must be <= {MAX_WARMUP_STEPS}"));
        }
        if self.initial_value == 0 || self.initial_value > MAX_INITIAL_VALUE {
            return Err(format!(
                "initial_value must be within 1..={MAX_INITIAL_VALUE}"
            ));
        }
        if self.custom_fractions.len() > MAX_CUSTOM_FRACTIONS_LEN {
            return Err("custom_fractions exceeds maximum length budget".to_string());
        }
        let program = match self.preset {
            FractranPreset::Primegame => FractranProgram::primegame(),
            FractranPreset::Fibonacci => FractranProgram::fibonacci(),
            FractranPreset::Multiplier => FractranProgram::multiplier(),
            FractranPreset::Adder => FractranProgram::adder(),
            FractranPreset::Custom => {
                FractranProgram::parse(&self.custom_fractions).map_err(|e| e.to_string())?
            }
        };
        let state = FractranState::from_u64(self.initial_value).map_err(|e| e.to_string())?;
        Ok((program, state))
    }
}

/// Single step telemetry sample comparing Shannon entropy against Kolmogorov upper bounds.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryPoint {
    /// Transition step index $t$.
    pub step: usize,
    /// Empirical 0th-order Shannon information $N \cdot H_0(x_t)$ in bits.
    pub shannon_bits: f64,
    /// Lempel-Ziv (`LZ76`) exhaustive phrase upper bound in bits.
    pub lz76_bits: f64,
    /// 1D Block Decomposition Method (`BDM`) complexity bound in bits.
    pub bdm_bits: f64,
    /// Elias prefix Kolmogorov upper bound $\log_2(N_t) + 2\log_2\log_2(N_t) + 1$ in bits.
    pub prefix_k_bits: f64,
}

/// Interactive zero-allocation FRACTRAN virtual machine and compressibility explorer.
pub struct FractranMachineTool {
    pub(crate) config: FractranToolConfig,
    pub(crate) program: FractranProgram,
    pub(crate) live: FractranState,
    pub(crate) is_playing: bool,
    pub(crate) last_rule_index: Option<usize>,
    pub(crate) next_rule_index: Option<usize>,
    pub(crate) emitted_primes: Vec<u32>,
    pub(crate) trajectory: Vec<TrajectoryPoint>,
    pub(crate) validation_error: Option<String>,
    pub(crate) status_banner: Option<String>,
    pub(crate) initial_input_text: String,
    pub(crate) dyadic_demo_k: f64,
    pub(crate) revision: u64,
}

impl Default for FractranMachineTool {
    fn default() -> Self {
        let cfg = FractranToolConfig::default();
        let (program, live) = cfg
            .validate()
            .expect("default FractranToolConfig must be valid");
        let mut tool = Self {
            initial_input_text: cfg.initial_value.to_string(),
            config: cfg,
            program,
            live,
            is_playing: false,
            last_rule_index: None,
            next_rule_index: None,
            emitted_primes: Vec::new(),
            trajectory: Vec::new(),
            validation_error: None,
            status_banner: None,
            dyadic_demo_k: 0.0,
            revision: 1,
        };
        tool.rebuild_machine(tool.config.warmup_steps);
        tool
    }
}

impl FractranMachineTool {
    /// Estimates $\sum r_i \log_2(p_i)$ to decide whether exact integer bit extraction is cheap.
    pub(crate) fn estimated_state_bit_length(&self) -> f64 {
        self.live
            .exponents
            .iter()
            .enumerate()
            .filter(|(_, exp)| **exp > 0)
            .map(|(idx, &exp)| f64::from(exp) * (FIRST_64_PRIMES[idx] as f64).log2())
            .sum()
    }

    fn sample_current_point(&self) -> TrajectoryPoint {
        let approx_bits = self.estimated_state_bit_length();
        let (bits, prefix_k) = if approx_bits <= 2_048.0 {
            let state_int = FractranProgram::decode_to_integer(&self.live.exponents);
            let k = prefix_kolmogorov_approx(&state_int);
            (integer_to_bits(&state_int), k)
        } else {
            let log_n = approx_bits.max(2.0);
            let k = log_n + 2.0 * log_n.log2() + 1.0;
            (exponents_to_bits(&self.live.exponents), k)
        };
        TrajectoryPoint {
            step: self.live.steps,
            shannon_bits: shannon_entropy_bits(&bits),
            lz76_bits: lz76_complexity(&bits),
            bdm_bits: bdm_complexity(&bits),
            prefix_k_bits: prefix_k,
        }
    }

    fn refresh_dyadic_summary(&mut self) {
        let r2 = i128::from(self.live.exponents[0]) + 1;
        let r3 = i128::from(self.live.exponents[1]) + 1;
        let pt = Point2D::new(
            math_explorer::math_types::Rational::from((r2, r2 + r3)),
            math_explorer::math_types::Rational::from((r3, r2 + r3 + 1)),
        );
        let dist = distance(&Point2D::from_ints(0, 0), &pt);
        let _ = Integer::from(dist.numer().round() as i64);
        self.dyadic_demo_k = k_floor_r(&pt, 12);
    }

    /// Re-initializes the register file from `self.config` and runs `warmup` transitions.
    pub(crate) fn rebuild_machine(&mut self, warmup: usize) {
        if let Ok((program, live)) = self.config.validate() {
            self.program = program;
            self.live = live;
            self.validation_error = None;
        }
        self.is_playing = false;
        self.last_rule_index = None;
        self.next_rule_index = self.program.matching_rule(&self.live.exponents);
        self.emitted_primes.clear();
        self.trajectory.clear();
        self.status_banner = None;
        self.initial_input_text = self.config.initial_value.to_string();
        self.trajectory.push(self.sample_current_point());
        self.refresh_dyadic_summary();
        for _ in 0..warmup.min(MAX_WARMUP_STEPS) {
            if !self.step_once() {
                break;
            }
        }
        self.bump_revision();
    }

    #[inline]
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Advances the VM by one transition, updating prime tape, compressibility curves, and cycle guard.
    pub fn step_once(&mut self) -> bool {
        match self.live.step(&self.program) {
            Ok(Some(rule_idx)) => {
                self.last_rule_index = Some(rule_idx);
                self.next_rule_index = self.program.matching_rule(&self.live.exponents);
                if let Some(p) = FractranProgram::power_of_two_exponent(&self.live.exponents) {
                    if self.emitted_primes.last().copied() != Some(p) {
                        self.emitted_primes.push(p);
                    }
                }
                if self.trajectory.len() >= MAX_TRAJECTORY_POINTS {
                    self.trajectory.remove(0);
                }
                self.trajectory.push(self.sample_current_point());
                self.refresh_dyadic_summary();
                self.bump_revision();
                if let Some(info) = self.live.cycle {
                    self.stop_live_playback_and_mute();
                    self.status_banner = Some(format!(
                        "Floyd cycle detected (start step {}, period {}) — playback paused.",
                        info.start_step, info.period
                    ));
                    return false;
                }
                true
            }
            Ok(None) => {
                self.stop_live_playback_and_mute();
                self.next_rule_index = None;
                self.status_banner = Some(format!(
                    "Machine halted cleanly after {} steps.",
                    self.live.steps
                ));
                self.bump_revision();
                false
            }
            Err(err) => {
                self.stop_live_playback_and_mute();
                self.validation_error = Some(err.to_string());
                self.bump_revision();
                false
            }
        }
    }

    /// Executes one frame's bounded batch (`1..=256` steps) while live playback is active.
    pub fn step_frame_batch(&mut self) {
        let scaled = ((self.config.steps_per_frame as f64) * self.config.playback_rate).round();
        let budget = if scaled.is_finite() && scaled >= 1.0 {
            (scaled as usize).clamp(1, MAX_STEPS_PER_FRAME)
        } else {
            self.config.steps_per_frame.clamp(1, MAX_STEPS_PER_FRAME)
        };
        for _ in 0..budget {
            if !self.step_once() {
                break;
            }
        }
    }

    /// Validates and applies a new [`FractranToolConfig`].
    pub fn apply_config(&mut self, cfg: FractranToolConfig) -> Result<(), String> {
        let (program, live) = cfg.validate()?;
        let warmup = cfg.warmup_steps;
        self.config = cfg;
        self.program = program;
        self.live = live;
        self.rebuild_machine(warmup);
        Ok(())
    }
}

impl TheoryDescribable for FractranMachineTool {
    fn theory_description(&self) -> String {
        "John Horton Conway's FRACTRAN encodes universal Minsky register machines in an ordered \
         list of positive fractions f_1, ..., f_k acting on prime-factored integers N = prod p_i^{r_i}. \
         At each transition, the first fraction with N * f_j in Z updates register exponents by \
         signed deltas without BigInt heap churn. Trajectory states are analyzed via prefix \
         Kolmogorov complexity bounds, Lempel-Ziv (LZ76) parsing, and 1D Block Decomposition (BDM)."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "FRAC-tran prime register virtual machine with Lempel-Ziv and Block Decomposition \
         Kolmogorov complexity bounds"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Conway, J. H. (1987). FRACTRAN: A Simple Universal Programming Language for Arithmetic; \
         Cholak et al., Algorithmic Information Bounds for Distances and Orthogonal Projections \
         [algorithmic_information_rust]."
            .to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("overview".to_string(), self.theory_description());
        map.insert("citation".to_string(), self.theory_citation());
        map
    }
}

impl InteractiveTool for FractranMachineTool {
    fn name(&self) -> &'static str {
        "FRACTRAN Virtual Machine"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn domain(&self) -> &'static str {
        "algorithmic_information"
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 16 * 1024,
            max_depth: 8,
            max_array_len: 128,
        }
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.playback_rate.is_finite()
            && (0.25..=16.0).contains(&self.config.playback_rate)
            && (1..=MAX_STEPS_PER_FRAME).contains(&self.config.steps_per_frame)
    }

    fn clamp_parameters(&mut self) -> bool {
        if !self.config.playback_rate.is_finite() {
            self.config.playback_rate = 1.0;
        } else {
            self.config.playback_rate = self.config.playback_rate.clamp(0.25, 16.0);
        }
        self.config.steps_per_frame = self.config.steps_per_frame.clamp(1, MAX_STEPS_PER_FRAME);
        self.config.warmup_steps = self.config.warmup_steps.min(MAX_WARMUP_STEPS);
        self.config.initial_value = self.config.initial_value.clamp(1, MAX_INITIAL_VALUE);
        self.bump_revision();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: FractranToolConfig =
            serde_json::from_str(state).map_err(|e| format!("Invalid FRACTRAN state: {e}"))?;
        self.apply_config(parsed)
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn step_simulation(&mut self) {
        let _ = self.step_once();
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.is_playing {
            self.step_frame_batch();
            parent_ui.ctx().request_repaint();
        }
        let side_id = crate::framework::scoped_panel_id(parent_ui, "fractran_controls_panel");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls_panel(ui);
        });
        self.show_complexity_panel(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_registers_and_rules(ui);
        });
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

// [cite:algorithmic_information_rust]
