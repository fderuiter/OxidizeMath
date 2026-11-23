//! Interactive FRACTRAN prime-exponent virtual machine and Minsky counter-machine explorer.

pub mod diagram;
mod panels;

use crate::framework::{ExportCapabilities, InteractiveTool};
use diagram::{draw_minsky_diagram, MinskyDiagramModel};
use eframe::egui;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranCompiledVm, FractranError, FractranPreset, FractranProgram, FractranState,
    MinskyTransition,
};
use math_explorer::pure_math::algorithmic_information::kolmogorov::{
    bdm_complexity, exponents_to_bits, lz76_complexity,
};
use panels::{
    build_export_datasets, build_trace_row, draw_fraction_rules_strip, draw_register_bar_plot,
    draw_status_banner, draw_trace_inspector, TraceRow,
};
use std::collections::VecDeque;

/// Maximum VM steps executed in a single UI frame during `"Run Until Power of 2"` breakpoints.
pub const MAX_BREAKPOINT_STEPS_PER_FRAME: usize = 4096;
const MAX_STEPS_PER_FRAME: usize = 512;
const MAX_TRACE_ENTRIES: usize = 64;
const MAX_FRACTIONS_INPUT_BYTES: usize = 4096;

/// Selectable FRACTRAN preset or user-edited fraction list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PresetChoice {
    /// Conway's 14-fraction `PRIMEGAME`.
    PrimeGame,
    /// Conway's 6-fraction multiplication machine (`MULT`).
    Mult,
    /// Single-fraction addition machine (`ADD`).
    Add,
    /// Nine-fraction Fibonacci pair machine (`FIBONACCI`).
    Fibonacci,
    /// User-supplied positive rational sequence.
    Custom,
}

impl PresetChoice {
    const ALL: [Self; 5] = [
        Self::PrimeGame,
        Self::Mult,
        Self::Add,
        Self::Fibonacci,
        Self::Custom,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::PrimeGame => "PRIMEGAME",
            Self::Mult => "MULT",
            Self::Add => "ADD",
            Self::Fibonacci => "FIBONACCI",
            Self::Custom => "Custom",
        }
    }

    const fn to_core(self) -> Option<FractranPreset> {
        match self {
            Self::PrimeGame => Some(FractranPreset::PrimeGame),
            Self::Mult => Some(FractranPreset::Mult),
            Self::Add => Some(FractranPreset::Add),
            Self::Fibonacci => Some(FractranPreset::Fibonacci),
            Self::Custom => None,
        }
    }
}

/// Terminal or active execution status of [`FractranMachineTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineStopReason {
    /// Actively advancing across frames.
    Running,
    /// Paused awaiting user transport input.
    Paused,
    /// Paused on hitting a pure prime power of two $2^k$.
    HitPowerOfTwo(u32),
    /// Halted because no fraction denominator divides the current register state.
    Halted,
    /// Paused after exhausting [`MAX_BREAKPOINT_STEPS_PER_FRAME`] steps in one frame.
    FrameCapReached,
    /// Paused because stepping `rule_index` would overflow `u32::MAX` on `prime`.
    Overflow {
        /// Zero-based rule index that triggered overflow protection.
        rule_index: usize,
        /// Prime register that reached `u32::MAX`.
        prime: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FractranMachineState {
    preset: PresetChoice,
    fractions_input: String,
    initial_value: u64,
    steps_per_frame: usize,
    stop_on_power_of_two: bool,
}

/// Interactive FRACTRAN register machine and Minsky counter-machine visualizer.
pub struct FractranMachineTool {
    preset: PresetChoice,
    fractions_input: String,
    initial_value_input: String,
    initial_value: u64,
    steps_per_frame: usize,
    stop_on_power_of_two: bool,
    program: FractranProgram,
    vm: FractranCompiledVm,
    minsky: FractranState,
    transitions: Vec<MinskyTransition>,
    registers: Vec<u32>,
    step_count: u64,
    last_fired_rule: Option<usize>,
    primes_found: Vec<u32>,
    trace: VecDeque<TraceRow>,
    is_playing: bool,
    run_until_pow2_active: bool,
    stop_reason: MachineStopReason,
    validation_error: Option<String>,
    complexity_bounds: (f64, f64),
}

impl Default for FractranMachineTool {
    fn default() -> Self {
        Self::from_core_preset(FractranPreset::PrimeGame, PresetChoice::PrimeGame)
    }
}

impl FractranMachineTool {
    fn from_core_preset(core: FractranPreset, choice: PresetChoice) -> Self {
        let (program, minsky, initial_value) = core.build();
        let fractions_input = program.format_fractions();
        let (vm, registers) = program
            .compile_for_initial(initial_value)
            .expect("built-in FRACTRAN preset compiles");
        let transitions = minsky.transitions(&vm);
        let mut tool = Self {
            preset: choice,
            fractions_input,
            initial_value_input: initial_value.to_string(),
            initial_value,
            steps_per_frame: 1,
            stop_on_power_of_two: true,
            program,
            vm,
            minsky,
            transitions,
            registers,
            step_count: 0,
            last_fired_rule: None,
            primes_found: Vec::new(),
            trace: VecDeque::with_capacity(MAX_TRACE_ENTRIES),
            is_playing: false,
            run_until_pow2_active: false,
            stop_reason: MachineStopReason::Paused,
            validation_error: None,
            complexity_bounds: (0.0, 0.0),
        };
        tool.record_trace_row();
        tool
    }

    /// Selects a built-in preset or switches to custom editing mode.
    pub fn select_preset(&mut self, choice: PresetChoice) {
        if let Some(core) = choice.to_core() {
            let (steps, stop_pow2) = (self.steps_per_frame, self.stop_on_power_of_two);
            *self = Self::from_core_preset(core, choice);
            self.steps_per_frame = steps;
            self.stop_on_power_of_two = stop_pow2;
        } else {
            self.preset = PresetChoice::Custom;
        }
    }

    /// Recompiles the VM from the current fraction list and starting integer text fields.
    pub fn apply_custom_inputs(&mut self) -> Result<(), String> {
        if self.fractions_input.len() > MAX_FRACTIONS_INPUT_BYTES {
            let err = format!("Fraction input exceeds {MAX_FRACTIONS_INPUT_BYTES} bytes");
            self.validation_error = Some(err.clone());
            self.is_playing = false;
            return Err(err);
        }
        let Ok(initial_value) = self.initial_value_input.trim().parse::<u64>() else {
            let err = "Initial integer N0 must be at least 1.".to_string();
            self.validation_error = Some(err.clone());
            self.is_playing = false;
            return Err(err);
        };
        let text = self.fractions_input.clone();
        self.compile_and_install(self.preset, &text, initial_value)
            .inspect_err(|e| {
                self.validation_error = Some(e.clone());
                self.is_playing = false;
            })
    }

    fn compile_and_install(
        &mut self,
        choice: PresetChoice,
        fractions_text: &str,
        initial_value: u64,
    ) -> Result<(), String> {
        let program =
            FractranProgram::parse_fractions(fractions_text).map_err(|e| e.to_string())?;
        let (vm, registers) = program
            .compile_for_initial(initial_value)
            .map_err(|e| e.to_string())?;
        let minsky = match choice.to_core() {
            Some(core) if program.format_fractions() == core.program().format_fractions() => {
                core.minsky_state()
            }
            _ => FractranState::from_vm_heuristic(&vm),
        };
        self.transitions = minsky.transitions(&vm);
        self.preset = choice;
        self.fractions_input = fractions_text.to_string();
        self.initial_value = initial_value;
        self.initial_value_input = initial_value.to_string();
        self.program = program;
        self.vm = vm;
        self.minsky = minsky;
        self.registers = registers;
        self.validation_error = None;
        self.reset_runtime_counters();
        Ok(())
    }

    /// Resets registers, trace log, and playback flags back to `initial_value`.
    pub fn reset_machine(&mut self) {
        if let Ok(regs) = self.vm.encode_value(self.initial_value) {
            self.registers = regs;
        }
        self.reset_runtime_counters();
    }

    fn reset_runtime_counters(&mut self) {
        self.step_count = 0;
        self.last_fired_rule = None;
        self.primes_found.clear();
        self.trace.clear();
        self.is_playing = false;
        self.run_until_pow2_active = false;
        self.stop_reason = MachineStopReason::Paused;
        self.record_trace_row();
    }

    fn pause_with(&mut self, reason: MachineStopReason) -> MachineStopReason {
        let was_playing = std::mem::take(&mut self.is_playing);
        self.run_until_pow2_active = false;
        self.stop_reason = reason;
        if was_playing
            && matches!(
                reason,
                MachineStopReason::Halted | MachineStopReason::Overflow { .. }
            )
        {
            crate::audio::notify_playback_paused();
        }
        reason
    }

    /// Executes a single deterministic FRACTRAN step and updates trace & $2^k$ state.
    pub fn step_once(&mut self) -> MachineStopReason {
        match self.vm.step_registers(&mut self.registers) {
            Ok(Some(rule_idx)) => {
                self.step_count += 1;
                self.last_fired_rule = Some(rule_idx);
                let pow2 = self.vm.power_of_two_exponent(&self.registers);
                if let Some(k) = pow2 {
                    self.primes_found.push(k);
                }
                self.record_trace_row();
                if let Some(k) =
                    pow2.filter(|_| self.stop_on_power_of_two || self.run_until_pow2_active)
                {
                    return self.pause_with(MachineStopReason::HitPowerOfTwo(k));
                }
                MachineStopReason::Running
            }
            Ok(None) => self.pause_with(MachineStopReason::Halted),
            Err(FractranError::RegisterOverflow { rule_index, prime }) => {
                self.pause_with(MachineStopReason::Overflow { rule_index, prime })
            }
            Err(_) => self.pause_with(MachineStopReason::Paused),
        }
    }

    /// Advances up to [`MAX_BREAKPOINT_STEPS_PER_FRAME`] (`4096`) steps until $2^k$, halt, overflow, or frame cap.
    pub fn run_until_power_of_two(&mut self) -> MachineStopReason {
        self.run_until_pow2_active = true;
        self.is_playing = true;
        for _ in 0..MAX_BREAKPOINT_STEPS_PER_FRAME {
            let outcome = self.step_once();
            if outcome != MachineStopReason::Running {
                return outcome;
            }
        }
        self.pause_with(MachineStopReason::FrameCapReached)
    }

    fn advance_playback_frame(&mut self, ctx: &egui::Context) {
        if self.run_until_pow2_active {
            let _ = self.run_until_power_of_two();
            return;
        }
        if !self.is_playing {
            return;
        }
        for _ in 0..self.steps_per_frame.clamp(1, MAX_STEPS_PER_FRAME) {
            if self.step_once() != MachineStopReason::Running {
                return;
            }
        }
        if self.is_playing {
            ctx.request_repaint();
        }
    }

    fn record_trace_row(&mut self) {
        let bits = exponents_to_bits(&self.registers);
        self.complexity_bounds = (bdm_complexity(&bits), lz76_complexity(&bits));
        if self.trace.len() >= MAX_TRACE_ENTRIES {
            self.trace.pop_front();
        }
        self.trace.push_back(build_trace_row(
            &self.vm,
            &self.minsky,
            &self.registers,
            self.step_count,
            self.last_fired_rule,
        ));
    }
}

impl InteractiveTool for FractranMachineTool {
    fn name(&self) -> &'static str {
        "FRACTRAN Machine"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        build_export_datasets(&self.vm, &self.registers, &self.primes_found)
    }

    fn step_simulation(&mut self) {
        let _ = self.step_once();
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset, &self.fractions_input, self.initial_value).hash(&mut h);
        (self.steps_per_frame, self.stop_on_power_of_two).hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&FractranMachineState {
            preset: self.preset,
            fractions_input: self.fractions_input.clone(),
            initial_value: self.initial_value,
            steps_per_frame: self.steps_per_frame,
            stop_on_power_of_two: self.stop_on_power_of_two,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: FractranMachineState =
            serde_json::from_str(state).map_err(|e| format!("Invalid FRACTRAN state: {e}"))?;
        if parsed.fractions_input.len() > MAX_FRACTIONS_INPUT_BYTES {
            return Err("FRACTRAN fractions_input exceeds byte limit".to_string());
        }
        if !(1..=MAX_STEPS_PER_FRAME).contains(&parsed.steps_per_frame) {
            return Err("steps_per_frame out of bounds (1..=512)".to_string());
        }
        self.compile_and_install(parsed.preset, &parsed.fractions_input, parsed.initial_value)?;
        self.steps_per_frame = parsed.steps_per_frame;
        self.stop_on_power_of_two = parsed.stop_on_power_of_two;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.advance_playback_frame(ui.ctx());
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("FRACTRAN Prime-Exponent Virtual Machine");
                self.render_controls_header(ui);
                self.render_transport_bar(ui);
                self.render_sonification_bar(ui);
                draw_status_banner(
                    ui,
                    self.step_count,
                    self.stop_reason,
                    self.validation_error.as_deref(),
                    &self.primes_found,
                );
                draw_fraction_rules_strip(ui, &self.program.fractions, self.last_fired_rule);
                ui.add_space(4.0);
                draw_register_bar_plot(ui, &self.vm, &self.registers);
                ui.add_space(6.0);
                ui.strong("Minsky Counter-Machine State Graph");
                draw_minsky_diagram(
                    ui,
                    &MinskyDiagramModel {
                        vm: &self.vm,
                        minsky: &self.minsky,
                        registers: &self.registers,
                        transitions: &self.transitions,
                        last_fired_rule: self.last_fired_rule,
                        step_count: self.step_count,
                    },
                );
                ui.add_space(4.0);
                draw_trace_inspector(ui, &self.trace);
            });
        });
    }
}

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

// [cite:algorithmic_information_rust]
