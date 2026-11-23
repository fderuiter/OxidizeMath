#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive `(Z, N)` nuclide chart canvas with keyboard arrow navigation and AccessKit support.

use super::nuclear_tool::NuclearStabilityTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::{self, Color32, Stroke};
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::physics::nuclear::{AtomicNumber, LiquidDropModel, MassNumber};

const MAX_CHART_N: u32 = 150;
const MAX_CHART_Z: u32 = 96;
const MAGIC_NUMBERS: [u32; 7] = [2, 8, 20, 28, 50, 82, 126];

/// Chemical symbols for $Z = 1..=96$ displayed in nuclide labels.
pub(crate) fn element_symbol(z: u32) -> &'static str {
    const SYMBOLS: [&str; 97] = [
        "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
        "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm",
    ];
    SYMBOLS.get(z as usize).copied().unwrap_or("Nu")
}

pub(crate) fn render_nuclide_chart_canvas(tool: &mut NuclearStabilityTool, ui: &mut egui::Ui) {
    let desired = egui::vec2(ui.available_width().max(220.0_f32), 235.0_f32);
    let (response, painter) = ui.allocate_painter(desired, egui::Sense::click_and_drag());
    let _ = response
        .clone()
        .accessible_hover_text("Interactive (Z, N) Nuclide Valley of Stability Chart");
    tool.focused_canvas = response.has_focus().then_some(response.id);

    register_nuclide_shortcuts(ui.ctx());
    lock_canvas_arrow_keys(ui, &response);
    if response.has_focus() {
        handle_chart_keyboard(tool, ui);
    }

    let plot_rect = response.rect.shrink2(egui::vec2(30.0_f32, 18.0_f32));
    if response.clicked() || response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            response.request_focus();
            select_nuclide_from_pointer(tool, ui.ctx(), plot_rect, pos);
        }
    }
    tool.focused_canvas = response.has_focus().then_some(response.id);

    let summary = format!(
        "{}-{} (Z={}, N={}), B/A = {:.3} MeV",
        element_symbol(tool.state.selected_z),
        tool.state.selected_z + tool.state.selected_n,
        tool.state.selected_z,
        tool.state.selected_n,
        tool.evaluate_observables()
            .map_or(0.0, |o| o.breakdown.per_nucleon)
    );
    register_canvas_accessibility(
        ui,
        &response,
        egui::accesskit::Role::GraphicsDocument,
        "Nuclide Valley of Stability Chart (N, Z)",
        Some(&summary),
    );

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let high_contrast = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
    painter.rect_filled(response.rect, 4.0_f32, ui.visuals().extreme_bg_color);
    draw_nuclide_cells(&painter, plot_rect, high_contrast);
    draw_magic_shells_and_cursor(tool, &painter, plot_rect, &palette);
    paint_canvas_focus_ring(ui, &response);
}

fn register_nuclide_shortcuts(ctx: &egui::Context) {
    CommandRegistryData::register(
        ctx,
        "Decrease Neutron Count N",
        "Step nuclide cursor left (N - 1) on the (Z, N) chart",
        CommandTrigger::Key(egui::Key::ArrowLeft),
        true,
        "Nuclide Chart Canvas",
    );
    CommandRegistryData::register(
        ctx,
        "Increase Neutron Count N",
        "Step nuclide cursor right (N + 1) on the (Z, N) chart",
        CommandTrigger::Key(egui::Key::ArrowRight),
        true,
        "Nuclide Chart Canvas",
    );
    CommandRegistryData::register(
        ctx,
        "Increase Proton Count Z",
        "Step nuclide cursor up (Z + 1) on the (Z, N) chart",
        CommandTrigger::Key(egui::Key::ArrowUp),
        true,
        "Nuclide Chart Canvas",
    );
    CommandRegistryData::register(
        ctx,
        "Decrease Proton Count Z",
        "Step nuclide cursor down (Z - 1) on the (Z, N) chart",
        CommandTrigger::Key(egui::Key::ArrowDown),
        true,
        "Nuclide Chart Canvas",
    );
}

fn handle_chart_keyboard(tool: &mut NuclearStabilityTool, ui: &mut egui::Ui) {
    let left = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft));
    let right = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight));
    let up = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp));
    let down = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown));
    if !(left || right || up || down) {
        return;
    }
    if left {
        tool.state.selected_n = tool.state.selected_n.saturating_sub(1);
    }
    if right {
        tool.state.selected_n = (tool.state.selected_n + 1).min(MAX_CHART_N);
    }
    if up {
        tool.state.selected_z = (tool.state.selected_z + 1).min(MAX_CHART_Z);
    }
    if down {
        tool.state.selected_z = tool.state.selected_z.saturating_sub(1).max(1);
    }
    tool.clamp_in_place();
    tool.revision = tool.revision.wrapping_add(1);
    announce_selection(tool, ui.ctx());
}

fn select_nuclide_from_pointer(
    tool: &mut NuclearStabilityTool,
    ctx: &egui::Context,
    rect: egui::Rect,
    pos: egui::Pos2,
) {
    let tx = ((pos.x - rect.left()) / rect.width().max(1.0_f32)).clamp(0.0_f32, 1.0_f32);
    let ty = ((rect.bottom() - pos.y) / rect.height().max(1.0_f32)).clamp(0.0_f32, 1.0_f32);
    let new_n = (tx * (MAX_CHART_N as f32)).round() as u32;
    let new_z = 1 + (ty * ((MAX_CHART_Z - 1) as f32)).round() as u32;
    if (new_z, new_n) != (tool.state.selected_z, tool.state.selected_n) {
        tool.state.selected_z = new_z.clamp(1, MAX_CHART_Z);
        tool.state.selected_n = new_n.clamp(0, MAX_CHART_N);
        tool.revision = tool.revision.wrapping_add(1);
        announce_selection(tool, ctx);
    }
}

fn announce_selection(tool: &NuclearStabilityTool, ctx: &egui::Context) {
    let sym = element_symbol(tool.state.selected_z);
    let a = tool.state.selected_z + tool.state.selected_n;
    let be_a = tool
        .evaluate_observables()
        .map_or(0.0, |o| o.breakdown.per_nucleon);
    announce_live_status(
        ctx,
        format!(
            "Selected {sym}-{a} (Z={}, N={}), B/A = {be_a:.3} MeV",
            tool.state.selected_z, tool.state.selected_n
        ),
    );
}

fn draw_nuclide_cells(painter: &egui::Painter, rect: egui::Rect, high_contrast: bool) {
    let ldm = LiquidDropModel::new();
    let step_n = 2u32;
    let step_z = 2u32;
    let cell_w = rect.width() * (step_n as f32) / ((MAX_CHART_N + 1) as f32);
    let cell_h = rect.height() * (step_z as f32) / (MAX_CHART_Z as f32);

    let mut z = 1u32;
    while z <= MAX_CHART_Z {
        let mut n = 0u32;
        while n <= MAX_CHART_N {
            if let Ok(a) = MassNumber::new(z + n) {
                if let Ok(bd) = ldm.binding_energy_breakdown(AtomicNumber::new(z), a) {
                    if bd.per_nucleon > 0.5 {
                        let norm = ((bd.per_nucleon - 2.0) / 6.85).clamp(0.0, 1.0);
                        let rgb = if high_contrast {
                            heatmap_color_high_contrast(norm)
                        } else {
                            heatmap_color(norm)
                        };
                        let x =
                            rect.left() + (n as f32 / MAX_CHART_N as f32) * (rect.width() - cell_w);
                        let y = rect.bottom()
                            - ((z - 1) as f32 / (MAX_CHART_Z - 1) as f32)
                                * (rect.height() - cell_h)
                            - cell_h;
                        let cell =
                            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_w, cell_h));
                        painter.rect_filled(
                            cell,
                            0.0_f32,
                            Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                        );
                    }
                }
            }
            n += step_n;
        }
        z += step_z;
    }
}

fn draw_magic_shells_and_cursor(
    tool: &NuclearStabilityTool,
    painter: &egui::Painter,
    rect: egui::Rect,
    palette: &SemanticPalette,
) {
    painter.rect_stroke(
        rect,
        2.0_f32,
        Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Outside,
    );
    // N = Z reference line
    let nz_end = MAX_CHART_Z as f32 / MAX_CHART_N as f32;
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.bottom()),
            egui::pos2(rect.left() + nz_end * rect.width(), rect.top()),
        ],
        Stroke::new(1.0_f32, palette.muted_text),
    );
    for &m in &MAGIC_NUMBERS {
        if m <= MAX_CHART_N {
            let x = rect.left() + (m as f32 / MAX_CHART_N as f32) * rect.width();
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                Stroke::new(0.75_f32, palette.graph_edge),
            );
        }
        if m <= MAX_CHART_Z {
            let y = rect.bottom() - ((m - 1) as f32 / (MAX_CHART_Z - 1) as f32) * rect.height();
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                Stroke::new(0.75_f32, palette.graph_edge),
            );
        }
    }

    let cx = rect.left() + (tool.state.selected_n as f32 / MAX_CHART_N as f32) * rect.width();
    let cy = rect.bottom()
        - ((tool.state.selected_z - 1) as f32 / (MAX_CHART_Z - 1) as f32) * rect.height();
    let cpos = egui::pos2(cx, cy);
    painter.circle_stroke(cpos, 6.5_f32, Stroke::new(2.2_f32, palette.node_stroke));
    painter.circle_filled(cpos, 3.2_f32, palette.accent);
    painter.text(
        egui::pos2(rect.left() + 6.0_f32, rect.top() + 4.0_f32),
        egui::Align2::LEFT_TOP,
        "Z vs N Valley of Stability (B/A MeV)",
        egui::FontId::proportional(11.5_f32),
        palette.node_stroke,
    );
}
