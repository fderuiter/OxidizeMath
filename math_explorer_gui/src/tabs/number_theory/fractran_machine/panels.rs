//! Control toolbars, status banners, register bar plot, trace view, and tool metadata.

use super::{
    FractranMachineTool, MachineStopReason, PresetChoice, MAX_BREAKPOINT_STEPS_PER_FRAME,
    MAX_STEPS_PER_FRAME,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use math_explorer::math_types::Rational;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranCompiledVm, FractranState,
};
use std::collections::VecDeque;

/// Bounded execution trace entry recorded per VM step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRow {
    /// Cumulative step number.
    pub step: u64,
    /// Fired zero-based rule index (`None` on step 0).
    pub fired_rule: Option<usize>,
    /// Active Minsky state label.
    pub state_label: String,
    /// Exponent $k$ when current state equals $2^k$.
    pub power_of_two: Option<u32>,
    /// Compact non-zero prime-exponent factorization string.
    pub register_summary: String,
}

/// Builds a [`TraceRow`] from the current compiled VM and register slice.
pub fn build_trace_row(
    vm: &FractranCompiledVm,
    minsky: &FractranState,
    registers: &[u32],
    step: u64,
    fired_rule: Option<usize>,
) -> TraceRow {
    let decoded = minsky.decode(vm, registers);
    let state_label = if decoded.active_state_primes.is_empty() {
        "q0".to_string()
    } else {
        decoded
            .active_state_primes
            .iter()
            .map(|(p, e)| {
                if *e == 1 {
                    format!("q{p}")
                } else {
                    format!("q{p}^{e}")
                }
            })
            .collect::<Vec<_>>()
            .join("*")
    };
    let non_zero: Vec<String> = vm
        .primes()
        .iter()
        .zip(registers)
        .filter(|(_, &e)| e > 0)
        .map(|(&p, &e)| format!("{p}^{e}"))
        .collect();
    let register_summary = if non_zero.is_empty() {
        "1".into()
    } else {
        non_zero.join(" * ")
    };
    TraceRow {
        step,
        fired_rule,
        state_label,
        power_of_two: vm.power_of_two_exponent(registers),
        register_summary,
    }
}

impl FractranMachineTool {
    pub(super) fn render_controls_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Preset:");
            for choice in PresetChoice::ALL {
                let selected = self.preset == choice;
                if ui
                    .selectable_label(selected, choice.label())
                    .accessible_hover_text("Select FRACTRAN counter-machine preset")
                    .clicked()
                    && !selected
                {
                    self.select_preset(choice);
                }
            }
            ui.separator();
            ui.label("N0:");
            if ui
                .add(egui::TextEdit::singleline(&mut self.initial_value_input).desired_width(72.0))
                .changed()
            {
                self.preset = PresetChoice::Custom;
            }
            if ui
                .button("✓ Compile")
                .accessible_hover_text("Compile FRACTRAN fractions and reset initial value N0")
                .clicked()
            {
                let _ = self.apply_custom_inputs();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Fractions:");
            let width = ui.available_width().max(180.0);
            if ui
                .add(egui::TextEdit::singleline(&mut self.fractions_input).desired_width(width))
                .changed()
            {
                self.preset = PresetChoice::Custom;
            }
        });
    }

    pub(super) fn render_transport_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let transport = PlaybackTransport::in_thread(
                "fractran_transport",
                "FRACTRAN Machine",
                self.is_playing,
            );
            if let Some(action) = transport.show(ui) {
                match action {
                    PlaybackAction::Play => {
                        self.is_playing = true;
                        self.stop_reason = MachineStopReason::Running;
                    }
                    PlaybackAction::Pause => {
                        let _ = self.pause_with(MachineStopReason::Paused);
                    }
                    PlaybackAction::Step => {
                        if self.step_once() == MachineStopReason::Running {
                            self.stop_reason = MachineStopReason::Paused;
                        }
                    }
                    PlaybackAction::Reset => self.reset_machine(),
                    PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
                }
            }
            if ui
                .button("⏩ Run Until Power of 2")
                .accessible_hover_text(
                    "Advance up to 4096 steps in this frame until reaching a pure power of two 2^k",
                )
                .clicked()
            {
                let _ = self.run_until_power_of_two();
            }
            ui.checkbox(&mut self.stop_on_power_of_two, "Break on 2^k")
                .accessible_hover_text("Pause playback automatically whenever N = 2^k");
            ui.add(
                egui::Slider::new(&mut self.steps_per_frame, 1..=MAX_STEPS_PER_FRAME)
                    .text("Steps/frame"),
            );
        });
    }
}

/// Renders execution status, validation error alerts, and discovered $2^k$ prime exponents.
pub fn draw_status_banner(
    ui: &mut egui::Ui,
    step_count: u64,
    stop_reason: MachineStopReason,
    validation_error: Option<&str>,
    primes_found: &[u32],
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    if let Some(err) = validation_error {
        ui.colored_label(palette.error, format!("Validation error: {err}"));
    }
    let status_text = match stop_reason {
        MachineStopReason::Running => format!("Step {step_count} — running"),
        MachineStopReason::Paused => format!("Step {step_count} — paused"),
        MachineStopReason::HitPowerOfTwo(k) => {
            format!("Step {step_count} — hit 2^{k} (exponent k = {k})")
        }
        MachineStopReason::Halted => format!("Step {step_count} — halted (no divisible rule)"),
        MachineStopReason::FrameCapReached => {
            format!("Step {step_count} — paused at {MAX_BREAKPOINT_STEPS_PER_FRAME}-step frame cap")
        }
        MachineStopReason::Overflow { rule_index, prime } => format!(
            "Step {step_count} — paused on u32::MAX overflow at R{} (prime {prime})",
            rule_index + 1
        ),
    };
    ui.horizontal_wrapped(|ui| {
        ui.strong(status_text);
        if !primes_found.is_empty() {
            let tail: Vec<String> = primes_found
                .iter()
                .rev()
                .take(12)
                .rev()
                .map(u32::to_string)
                .collect();
            ui.colored_label(
                palette.success,
                format!("  2^k hits: [{}]", tail.join(", ")),
            );
        }
    });
}

/// Renders the fraction multiplier rules strip, highlighting `last_fired_rule`.
pub fn draw_fraction_rules_strip(
    ui: &mut egui::Ui,
    fractions: &[Rational],
    last_fired_rule: Option<usize>,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal_wrapped(|ui| {
        for (idx, frac) in fractions.iter().enumerate() {
            let text = format!(
                "R{}: {}/{}",
                idx + 1,
                frac.0.numerator(),
                frac.0.denominator()
            );
            let rich = if last_fired_rule == Some(idx) {
                egui::RichText::new(text).strong().color(palette.accent)
            } else {
                egui::RichText::new(text)
                    .monospace()
                    .color(palette.muted_text)
            };
            ui.label(rich);
        }
    });
}

/// Renders prime-exponent register bars inside `egui_plot::Plot`.
pub fn draw_register_bar_plot(ui: &mut egui::Ui, vm: &FractranCompiledVm, registers: &[u32]) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let bars: Vec<Bar> = vm
        .primes()
        .iter()
        .zip(registers)
        .enumerate()
        .map(|(idx, (&prime, &exp))| {
            Bar::new(idx as f64, f64::from(exp))
                .width(0.66)
                .name(format!("p = {prime}"))
                .fill(palette.series_color(idx))
        })
        .collect();
    let primes_snapshot = vm.primes().to_vec();
    Plot::new("fractran_register_plot")
        .height(190.0)
        .include_y(0.0)
        .include_y(4.0)
        .x_axis_formatter(move |mark, _range| {
            let idx = mark.value.round() as isize;
            if idx >= 0 && (mark.value - idx as f64).abs() < 1e-3 {
                if let Some(&p) = primes_snapshot.get(idx as usize) {
                    return format!("p={p}");
                }
            }
            String::new()
        })
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Prime Register Exponents", bars));
        });
}

/// Renders the collapsible execution trace inspector table.
pub fn draw_trace_inspector(ui: &mut egui::Ui, trace: &VecDeque<TraceRow>) {
    ui.collapsing("Execution Trace Inspector (recent steps)", |ui| {
        egui::ScrollArea::vertical()
            .max_height(130.0)
            .show(ui, |ui| {
                for row in trace.iter().rev().take(24) {
                    let rule_txt = row
                        .fired_rule
                        .map_or_else(|| "init".to_string(), |r| format!("R{}", r + 1));
                    let pow_txt = row
                        .power_of_two
                        .map_or_else(String::new, |k| format!(" [2^{k}]"));
                    ui.monospace(format!(
                        "#{:<5} {:<4} {:<8} {}{}",
                        row.step, rule_txt, row.state_label, row.register_summary, pow_txt
                    ));
                }
            });
    });
}

/// Builds exportable `[prime, exponent]` and `[index, 2^k]` plot datasets.
pub fn build_export_datasets(
    vm: &FractranCompiledVm,
    registers: &[u32],
    primes_found: &[u32],
) -> Vec<(String, Vec<[f64; 2]>)> {
    let regs = vm
        .primes()
        .iter()
        .zip(registers)
        .map(|(&p, &e)| [p as f64, f64::from(e)])
        .collect();
    let mut datasets = vec![("Prime Register Exponents".to_string(), regs)];
    if !primes_found.is_empty() {
        let pows = primes_found
            .iter()
            .enumerate()
            .map(|(i, &k)| [(i + 1) as f64, f64::from(k)])
            .collect();
        datasets.push(("Discovered 2^k Exponents".to_string(), pows));
    }
    datasets
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {"schema_version": "1.0", "title": "Conway PRIMEGAME Machine", "domain": "number_theory", "timestamp": "2026-11-21T00:00:00Z"},
  "config": {"preset": "PrimeGame", "fractions_input": "17/91, 78/85, 19/51, 23/38, 29/33, 77/29, 95/23, 77/19, 1/17, 11/13, 13/11, 15/2, 1/7, 55/1", "initial_value": 2, "steps_per_frame": 4, "stop_on_power_of_two": true}
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FractranMachineTool",
        display_name: "FRACTRAN Machine",
        domain: "number_theory",
        tags: &["fractran", "minsky", "primes", "register-machine"],
        build: || Box::new(FractranMachineTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractran_primegame_starter",
        title: "Conway PRIMEGAME Machine",
        description: "Conway's 14-fraction universal prime-generating register machine decoded into Minsky state flags and counters.",
        category: "Computability",
        icon: "🧮",
        domain: "number_theory",
        tool_name: Some("FractranMachineTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

impl scientific_metadata::theory::TheoryDescribable for FractranMachineTool {
    fn theory_description(&self) -> String {
        "Conway's FRACTRAN universal register language encodes machine configurations as prime-exponent vectors $N = \\prod_{i=1}^m p_i^{e_i}$ and rational multipliers $f_j = A_j / B_j$ as Minsky counter-machine transitions subtracting $\\nu_p(B_j)$ when divisible and adding $\\nu_p(A_j)$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Conway FRACTRAN prime exponent virtual machine and Minsky counter machine universality"
            .into()
    }
    fn theory_citation(&self) -> String {
        "FRACTRAN: A Simple Universal Programming Language for Arithmetic (Conway, 1987); Computation: Finite and Infinite Machines (Minsky, 1967)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

// [cite:algorithmic_information_rust]
