//! Accessible Minsky counter-machine state graph, register bar plot, and trace view.

use super::{MachineStopReason, MAX_BREAKPOINT_STEPS_PER_FRAME};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use math_explorer::math_types::Rational;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranCompiledVm, FractranState, MinskyDecodedState, MinskyTransition,
};
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Accessible label exposed to `AccessKit` for the Minsky counter-machine graph canvas.
pub const MINSKY_CANVAS_LABEL: &str = "FRACTRAN Minsky Counter-Machine Graph";

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
            .map(|(p, e)| if *e == 1 { format!("q{p}") } else { format!("q{p}^{e}") })
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
    let register_summary = if non_zero.is_empty() { "1".into() } else { non_zero.join(" * ") };
    TraceRow {
        step,
        fired_rule,
        state_label,
        power_of_two: vm.power_of_two_exponent(registers),
        register_summary,
    }
}

/// Read-only view model rendered by [`draw_minsky_diagram`].
pub struct MinskyDiagramModel<'a> {
    /// Compiled prime-exponent VM.
    pub vm: &'a FractranCompiledVm,
    /// Minsky state-prime vs register-prime partition.
    pub minsky: &'a FractranState,
    /// Current prime-exponent register values.
    pub registers: &'a [u32],
    /// Extracted Minsky state transitions across all rules.
    pub transitions: &'a [MinskyTransition],
    /// Most recently fired zero-based fraction rule index.
    pub last_fired_rule: Option<usize>,
    /// Total executed FRACTRAN steps.
    pub step_count: u64,
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
            let tail: Vec<String> =
                primes_found.iter().rev().take(12).rev().map(u32::to_string).collect();
            ui.colored_label(palette.success, format!("  2^k hits: [{}]", tail.join(", ")));
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
            let text = format!("R{}: {}/{}", idx + 1, frac.0.numerator(), frac.0.denominator());
            let rich = if last_fired_rule == Some(idx) {
                egui::RichText::new(text).strong().color(palette.accent)
            } else {
                egui::RichText::new(text).monospace().color(palette.muted_text)
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
        egui::ScrollArea::vertical().max_height(130.0).show(ui, |ui| {
            for row in trace.iter().rev().take(24) {
                let rule_txt =
                    row.fired_rule.map_or_else(|| "init".to_string(), |r| format!("R{}", r + 1));
                let pow_txt = row.power_of_two.map_or_else(String::new, |k| format!(" [2^{k}]"));
                ui.monospace(format!(
                    "#{:<5} {:<4} {:<8} {}{}",
                    row.step, rule_txt, row.state_label, row.register_summary, pow_txt
                ));
            }
        });
    });
}

/// Renders the `AccessKit`-focusable Minsky counter-machine state-transition diagram.
pub fn draw_minsky_diagram(ui: &mut egui::Ui, model: &MinskyDiagramModel<'_>) {
    let decoded = model.minsky.decode(model.vm, model.registers);
    let summary = format_accessibility_summary(&decoded, model.last_fired_rule, model.step_count);
    let width = ui.available_width().max(260.0);
    let (response, painter) =
        ui.allocate_painter(egui::vec2(width, 228.0), egui::Sense::focusable_noninteractive());
    register_canvas_accessibility(
        ui,
        &response,
        egui::accesskit::Role::GraphicsDocument,
        MINSKY_CANVAS_LABEL,
        Some(&summary),
    );

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let text_color = ui.visuals().strong_text_color();
    painter.rect_filled(response.rect, 6.0_f32, palette.surface_fill);
    painter.rect_stroke(
        response.rect,
        6.0_f32,
        egui::Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );

    let nodes = compute_state_nodes(response.rect, &model.minsky.state_primes);
    draw_transition_edges(&painter, &nodes, model.transitions, model.last_fired_rule, &palette);
    draw_state_nodes(&painter, &nodes, &decoded.active_state_primes, &palette, text_color);
    draw_counter_strip(&painter, response.rect, &decoded.counters, &palette, text_color);
    paint_canvas_focus_ring(ui, &response);
}

fn format_accessibility_summary(
    decoded: &MinskyDecodedState,
    last_fired_rule: Option<usize>,
    step_count: u64,
) -> String {
    let state_desc = if decoded.active_state_primes.is_empty() {
        "q0 (resting)".to_string()
    } else {
        decoded
            .active_state_primes
            .iter()
            .map(|(p, e)| if *e == 1 { format!("q{p}") } else { format!("q{p}^{e}") })
            .collect::<Vec<_>>()
            .join("+")
    };
    let counters_desc =
        decoded.counters.iter().map(|(p, e)| format!("r{p}={e}")).collect::<Vec<_>>().join(", ");
    let rule_desc = last_fired_rule.map_or_else(|| "none".to_string(), |r| format!("R{}", r + 1));
    format!(
        "Step {step_count}, active state {state_desc}, fired {rule_desc}, counters [{counters_desc}]"
    )
}

fn compute_state_nodes(rect: egui::Rect, state_primes: &[u64]) -> Vec<(Option<u64>, egui::Pos2)> {
    let total = 1 + state_primes.len();
    let graph_rect = egui::Rect::from_min_max(
        rect.min + egui::vec2(24.0, 14.0),
        egui::pos2(rect.max.x - 24.0, rect.max.y - 42.0),
    );
    let center = graph_rect.center();
    if total == 1 {
        return vec![(None, center)];
    }
    let (rx, ry) = ((graph_rect.width() * 0.38).max(42.0), (graph_rect.height() * 0.36).max(34.0));
    (0..total)
        .map(|idx| {
            let angle = -TAU * 0.25 + TAU * (idx as f32) / (total as f32);
            let pos = egui::pos2(center.x + rx * angle.cos(), center.y + ry * angle.sin());
            (if idx == 0 { None } else { Some(state_primes[idx - 1]) }, pos)
        })
        .collect()
}

fn draw_transition_edges(
    painter: &egui::Painter,
    nodes: &[(Option<u64>, egui::Pos2)],
    transitions: &[MinskyTransition],
    last_fired: Option<usize>,
    palette: &SemanticPalette,
) {
    let find_pos = |st: Option<u64>| {
        nodes.iter().find_map(|&(s, pos)| (s == st).then_some(pos)).unwrap_or(nodes[0].1)
    };
    for tr in transitions {
        if Some(tr.rule_index) != last_fired {
            let stroke = egui::Stroke::new(1.25_f32, palette.graph_edge);
            paint_single_transition(
                painter,
                find_pos(tr.from_state),
                find_pos(tr.to_state),
                tr,
                stroke,
                None,
            );
        }
    }
    if let Some(tr) = last_fired.and_then(|idx| transitions.iter().find(|t| t.rule_index == idx)) {
        let tag = format!("R{}", tr.rule_index + 1);
        paint_single_transition(
            painter,
            find_pos(tr.from_state),
            find_pos(tr.to_state),
            tr,
            egui::Stroke::new(2.6_f32, palette.accent),
            Some((&tag, palette.accent)),
        );
    }
}

fn paint_single_transition(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    tr: &MinskyTransition,
    stroke: egui::Stroke,
    badge: Option<(&str, egui::Color32)>,
) {
    let radius = 17.0_f32;
    if (to - from).length_sq() < 4.0 {
        let loop_center = from + egui::vec2(0.0, -radius - 10.0);
        painter.circle_stroke(loop_center, 10.0, stroke);
        if let Some((label, color)) = badge {
            painter.text(
                loop_center + egui::vec2(0.0, -14.0),
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(11.5),
                color,
            );
        }
        return;
    }
    let dir = (to - from).normalized();
    let perp = egui::vec2(-dir.y, dir.x);
    let offset = perp * (((tr.rule_index % 3) as f32 - 1.0) * 4.5);
    let (start, end) = (from + dir * radius + offset, to - dir * radius + offset);
    painter.line_segment([start, end], stroke);
    painter.line_segment([end, end - dir * 7.5 + perp * 4.0], stroke);
    painter.line_segment([end, end - dir * 7.5 - perp * 4.0], stroke);
    if let Some((label, color)) = badge {
        painter.text(
            start.lerp(end, 0.5) + perp * 9.0,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(11.5),
            color,
        );
    }
}

fn draw_state_nodes(
    painter: &egui::Painter,
    nodes: &[(Option<u64>, egui::Pos2)],
    active_primes: &[(u64, u32)],
    palette: &SemanticPalette,
    text_color: egui::Color32,
) {
    for &(state_prime, pos) in nodes {
        let is_active = state_prime.map_or(active_primes.is_empty(), |p| {
            active_primes.iter().any(|&(ap, _)| ap == p)
        });
        let fill = if is_active { palette.info.gamma_multiply(0.28) } else { palette.surface_fill };
        let border = if is_active {
            egui::Stroke::new(2.5_f32, palette.success)
        } else {
            egui::Stroke::new(1.4_f32, palette.node_stroke)
        };
        painter.circle(pos, 17.0, fill, border);
        let label = state_prime.map_or_else(|| "q0".to_string(), |p| format!("q{p}"));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::monospace(11.5),
            text_color,
        );
    }
}

fn draw_counter_strip(
    painter: &egui::Painter,
    rect: egui::Rect,
    counters: &[(u64, u32)],
    palette: &SemanticPalette,
    text_color: egui::Color32,
) {
    let strip_top = rect.max.y - 30.0;
    painter.line_segment(
        [egui::pos2(rect.min.x + 10.0, strip_top), egui::pos2(rect.max.x - 10.0, strip_top)],
        egui::Stroke::new(1.0_f32, palette.graph_edge),
    );
    let summary = if counters.is_empty() {
        "Minsky Data Registers: none".to_string()
    } else {
        let items: Vec<String> =
            counters.iter().take(8).map(|(p, e)| format!("r_{p} = {e}")).collect();
        format!("Minsky Data Registers:  {}", items.join("   |   "))
    };
    painter.text(
        egui::pos2(rect.min.x + 14.0, strip_top + 15.0),
        egui::Align2::LEFT_CENTER,
        summary,
        egui::FontId::monospace(11.5),
        text_color,
    );
}

/// Builds exportable `[prime, exponent]` and `[index, 2^k]` plot datasets.
pub fn build_export_datasets(
    vm: &FractranCompiledVm,
    registers: &[u32],
    primes_found: &[u32],
) -> Vec<(String, Vec<[f64; 2]>)> {
    let regs = vm.primes().iter().zip(registers).map(|(&p, &e)| [p as f64, f64::from(e)]).collect();
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
        build: || Box::new(super::FractranMachineTool::default()),
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

impl scientific_metadata::theory::TheoryDescribable for super::FractranMachineTool {
    fn theory_description(&self) -> String {
        "Conway's FRACTRAN universal register language encodes machine configurations as prime-exponent vectors $N = \\prod_{i=1}^m p_i^{e_i}$ and rational multipliers $f_j = A_j / B_j$ as Minsky counter-machine transitions subtracting $\\nu_p(B_j)$ when divisible and adding $\\nu_p(A_j)$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Conway FRACTRAN prime exponent virtual machine and Minsky counter machine universality".into()
    }
    fn theory_citation(&self) -> String {
        "FRACTRAN: A Simple Universal Programming Language for Arithmetic (Conway, 1987); Computation: Finite and Infinite Machines (Minsky, 1967)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

// [cite:algorithmic_information_rust]
