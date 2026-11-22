//! Accessible Minsky counter-machine state-transition graph canvas.

use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranCompiledVm, FractranState, MinskyDecodedState, MinskyTransition,
};
use std::f32::consts::TAU;

/// Accessible label exposed to `AccessKit` for the Minsky counter-machine graph canvas.
pub const MINSKY_CANVAS_LABEL: &str = "FRACTRAN Minsky Counter-Machine Graph";

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

/// Renders the `AccessKit`-focusable Minsky counter-machine state-transition diagram.
pub fn draw_minsky_diagram(ui: &mut egui::Ui, model: &MinskyDiagramModel<'_>) {
    let decoded = model.minsky.decode(model.vm, model.registers);
    let summary = format_accessibility_summary(&decoded, model.last_fired_rule, model.step_count);
    let width = ui.available_width().max(260.0);
    let (response, painter) = ui.allocate_painter(
        egui::vec2(width, 228.0),
        egui::Sense::focusable_noninteractive(),
    );
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
    draw_transition_edges(
        &painter,
        &nodes,
        model.transitions,
        model.last_fired_rule,
        &palette,
    );
    draw_state_nodes(
        &painter,
        &nodes,
        &decoded.active_state_primes,
        &palette,
        text_color,
    );
    draw_counter_strip(
        &painter,
        response.rect,
        &decoded.counters,
        &palette,
        text_color,
    );
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
            .map(|(p, e)| {
                if *e == 1 {
                    format!("q{p}")
                } else {
                    format!("q{p}^{e}")
                }
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    let counters_desc = decoded
        .counters
        .iter()
        .map(|(p, e)| format!("r{p}={e}"))
        .collect::<Vec<_>>()
        .join(", ");
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
    let (rx, ry) = (
        (graph_rect.width() * 0.38).max(42.0),
        (graph_rect.height() * 0.36).max(34.0),
    );
    (0..total)
        .map(|idx| {
            let angle = -TAU * 0.25 + TAU * (idx as f32) / (total as f32);
            let pos = egui::pos2(center.x + rx * angle.cos(), center.y + ry * angle.sin());
            (
                if idx == 0 {
                    None
                } else {
                    Some(state_primes[idx - 1])
                },
                pos,
            )
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
        nodes
            .iter()
            .find_map(|&(s, pos)| (s == st).then_some(pos))
            .unwrap_or(nodes[0].1)
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
        let fill = if is_active {
            palette.info.gamma_multiply(0.28)
        } else {
            palette.surface_fill
        };
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
        [
            egui::pos2(rect.min.x + 10.0, strip_top),
            egui::pos2(rect.max.x - 10.0, strip_top),
        ],
        egui::Stroke::new(1.0_f32, palette.graph_edge),
    );
    let summary = if counters.is_empty() {
        "Minsky Data Registers: none".to_string()
    } else {
        let items: Vec<String> = counters
            .iter()
            .take(8)
            .map(|(p, e)| format!("r_{p} = {e}"))
            .collect();
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

// [cite:algorithmic_information_rust]
