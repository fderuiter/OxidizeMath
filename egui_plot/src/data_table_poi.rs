//! Critical Points table view and keyboard navigation inside [`crate::data_table`].

use egui::{
    Align, Color32, Context, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense, Stroke,
    TextStyle, Ui, vec2,
};

use crate::{
    data_table::{
        COL_W, DataTableState, HDR_H, IDX_W, MAX_H, ROW_H, RowCells, copy_to_clipboard,
        format::{format_pois_csv, format_pois_tsv, format_table_value},
    },
    emit_plot_live_message,
    poi::CriticalPoint,
};

const SERIES_COL_W: f32 = 188.0;

pub(crate) fn clamp_poi_state(state: &mut DataTableState, count: usize) {
    if count == 0 {
        state.selected_poi_row = 0;
        state.scroll_to_selected = false;
        state.pending_scroll_offset = None;
    } else {
        state.selected_poi_row = state.selected_poi_row.min(count - 1);
    }
}

pub(crate) fn handle_poi_keys(
    ctx: &Context,
    plot_id: Id,
    state: &mut DataTableState,
    pois: &[CriticalPoint],
) {
    if pois.is_empty() {
        return;
    }
    let prev = state.selected_poi_row;
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowDown)) {
        state.selected_poi_row = (prev + 1).min(pois.len() - 1);
        state.scroll_to_selected = true;
    } else if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowUp)) {
        state.selected_poi_row = prev.saturating_sub(1);
        state.scroll_to_selected = true;
    }
    if state.selected_poi_row != prev
        && let Some(poi) = pois.get(state.selected_poi_row)
    {
        let (x, y) = (
            format_table_value(poi.point[0]),
            format_table_value(poi.point[1]),
        );
        let msg = format!(
            "Critical point {} of {}: {} ({}), X = {x}, Y = {y}",
            state.selected_poi_row + 1,
            pois.len(),
            poi.kind.label(),
            poi.series_summary()
        );
        emit_plot_live_message(ctx, msg, false);
    }
    if ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)))
        && let Some(poi) = pois.get(state.selected_poi_row)
    {
        let single = format_pois_tsv(std::slice::from_ref(poi));
        copy_to_clipboard(ctx, plot_id, state, "selected critical point (TSV)", single);
    }
}

pub(crate) fn render_poi_body(
    ui: &mut Ui,
    plot_id: Id,
    state: &mut DataTableState,
    pois: &[CriticalPoint],
) -> bool {
    if pois.is_empty() {
        clamp_poi_state(state, 0);
        ui.label(
            RichText::new("No critical points detected in the current visible viewport.").strong(),
        );
        ui.label(
            RichText::new(
                "Pan or zoom over roots, y-intercepts, local extrema, or curve intersections.",
            )
            .small()
            .weak(),
        );
        return render_poi_footer(ui, plot_id, state, pois);
    }
    clamp_poi_state(state, pois.len());
    render_poi_grid(ui, plot_id, state, pois);
    render_poi_footer(ui, plot_id, state, pois)
}

fn render_poi_grid(ui: &mut Ui, plot_id: Id, state: &mut DataTableState, pois: &[CriticalPoint]) {
    let width = IDX_W + COL_W + SERIES_COL_W + 2.0 * COL_W;
    ScrollArea::horizontal()
        .id_salt(plot_id.with("poi_table_hscroll"))
        .max_width(ui.available_width().max(320.0))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_min_width(width);
            render_poi_header(ui, width);
            ui.separator();
            ui.style_mut().always_scroll_the_only_direction = false;
            let mut vs = ScrollArea::vertical()
                .id_salt(plot_id.with("poi_table_vscroll"))
                .max_height(MAX_H)
                .auto_shrink([false, false]);
            if let Some(off) = state.pending_scroll_offset.take() {
                vs = vs.vertical_scroll_offset(off);
            }
            vs.show_rows(ui, ROW_H, pois.len(), |ui, range| {
                ui.set_min_width(width);
                if state.scroll_to_selected && !range.contains(&state.selected_poi_row) {
                    let pitch = ROW_H + ui.spacing().item_spacing.y;
                    let target = (state.selected_poi_row as f32 * pitch - MAX_H * 0.4).max(0.0);
                    state.pending_scroll_offset = Some(target);
                    state.scroll_to_selected = false;
                    ui.ctx().request_repaint();
                }
                for r in range {
                    paint_poi_row(ui, state, r, &pois[r], width);
                }
            });
        });
}

fn render_poi_header(ui: &mut Ui, width: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, HDR_H), Sense::hover());
    ui.painter()
        .rect_filled(rect, 3.0_f32, ui.visuals().faint_bg_color);
    let cells = RowCells {
        ui,
        rect,
        height: HDR_H,
        font: TextStyle::Body.resolve(ui.style()),
    };
    let fg = ui.visuals().strong_text_color();
    let _ = cells.paint_span(0.0, IDX_W, "#", fg, None);
    let _ = cells.paint_span(IDX_W, COL_W, "Category", fg, None);
    let _ = cells.paint_span(IDX_W + COL_W, SERIES_COL_W, "Series", fg, None);
    let _ = cells.paint_span(IDX_W + COL_W + SERIES_COL_W, COL_W, "X", fg, None);
    let _ = cells.paint_span(IDX_W + 2.0 * COL_W + SERIES_COL_W, COL_W, "Y", fg, None);
}

fn paint_poi_row(
    ui: &mut Ui,
    state: &mut DataTableState,
    row_idx: usize,
    poi: &CriticalPoint,
    width: f32,
) {
    let (rect, resp) = ui.allocate_exact_size(vec2(width, ROW_H), Sense::click());
    if resp.clicked() {
        state.selected_poi_row = row_idx;
    }
    paint_poi_row_bg(ui, state, rect, &resp, row_idx);
    let cells = RowCells {
        ui,
        rect,
        height: ROW_H,
        font: TextStyle::Monospace.resolve(ui.style()),
    };
    let fg = if row_idx == state.selected_poi_row {
        ui.visuals().selection.stroke.color
    } else {
        ui.visuals().text_color()
    };
    let dot_gray = Color32::from_rgb(128, 134, 145);
    let _ = cells.paint_span(
        0.0,
        IDX_W,
        &(row_idx + 1).to_string(),
        ui.visuals().weak_text_color(),
        None,
    );
    let _ = cells.paint_span(IDX_W, COL_W, poi.kind.label(), fg, Some(dot_gray));
    let series_txt = poi.series_summary();
    let series_cell = cells.paint_span(IDX_W + COL_W, SERIES_COL_W, &series_txt, fg, None);
    if resp.hover_pos().is_some_and(|p| series_cell.contains(p)) {
        resp.on_hover_text(&series_txt);
    }
    let _ = cells.paint_span(
        IDX_W + COL_W + SERIES_COL_W,
        COL_W,
        &format_table_value(poi.point[0]),
        fg,
        None,
    );
    let _ = cells.paint_span(
        IDX_W + 2.0 * COL_W + SERIES_COL_W,
        COL_W,
        &format_table_value(poi.point[1]),
        fg,
        None,
    );
}

fn paint_poi_row_bg(
    ui: &Ui,
    state: &mut DataTableState,
    rect: Rect,
    resp: &egui::Response,
    row_idx: usize,
) {
    if row_idx == state.selected_poi_row {
        ui.painter()
            .rect_filled(rect, 2.0_f32, ui.visuals().selection.bg_fill);
        let stroke = Stroke::new(1.0_f32, ui.visuals().selection.stroke.color);
        ui.painter()
            .rect_stroke(rect, 2.0_f32, stroke, egui::StrokeKind::Inside);
        if state.scroll_to_selected {
            resp.scroll_to_me(Some(Align::Center));
            state.scroll_to_selected = false;
        }
    } else if resp.hovered() {
        let bg = ui
            .visuals()
            .widgets
            .hovered
            .bg_fill
            .gamma_multiply(0.35_f32);
        ui.painter().rect_filled(rect, 2.0_f32, bg);
    } else if row_idx % 2 == 1 {
        ui.painter()
            .rect_filled(rect, 0.0_f32, ui.visuals().faint_bg_color);
    }
}

fn render_poi_footer(
    ui: &mut Ui,
    plot_id: Id,
    state: &mut DataTableState,
    pois: &[CriticalPoint],
) -> bool {
    if let Some((ref msg, is_err)) = state.status_message {
        let c = if is_err {
            ui.visuals().error_fg_color
        } else {
            ui.visuals().selection.stroke.color
        };
        ui.colored_label(c, msg);
    }
    ui.separator();
    let has_pois = !pois.is_empty();
    let mut close_clicked = false;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(has_pois, egui::Button::new("Copy TSV"))
            .clicked()
        {
            let tsv = format_pois_tsv(pois);
            copy_to_clipboard(ui.ctx(), plot_id, state, "critical points (TSV)", tsv);
        }
        if ui
            .add_enabled(has_pois, egui::Button::new("Copy CSV"))
            .clicked()
        {
            let csv = format_pois_csv(pois);
            copy_to_clipboard(ui.ctx(), plot_id, state, "critical points (CSV)", csv);
        }
        close_clicked = ui.button("Close").clicked();
        if has_pois {
            let hint = format!(
                "Point {}/{} (↑/↓ to navigate)",
                state.selected_poi_row + 1,
                pois.len()
            );
            ui.label(RichText::new(hint).small().weak());
        }
    });
    close_clicked
}
