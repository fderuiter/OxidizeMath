//! Critical Points table view and keyboard navigation inside [`crate::data_table`].

use egui::{
    Align, Color32, Context, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense, Stroke,
    TextStyle, Ui, UiBuilder, vec2,
};

use crate::{
    a11y::{
        AccessibleTooltipExt as _, accessible_marker_color, composite_over, focus_ring_stroke,
        modal_action_button, register_cell_node, register_table_node, scope_table_row,
        set_accesskit_bounds, status_message_label,
    },
    data_table::{
        COL_W, DataTableState, HDR_H, IDX_W, MAX_H, ROW_H, RowCells, consume_copy_event,
        copy_to_clipboard,
        format::{format_pois_csv, format_pois_tsv, format_table_value},
    },
    emit_plot_live_message,
    poi::CriticalPoint,
};

const SERIES_COL_W: f32 = 188.0;
const POI_DOT_GRAY: Color32 = Color32::from_rgb(128, 134, 145);

pub(crate) fn clamp_poi_state(state: &mut DataTableState, count: usize) {
    if count == 0 {
        state.selected_poi_row = 0;
        state.scroll_to_selected = false;
        state.pending_scroll_offset = None;
    } else {
        state.selected_poi_row = state.selected_poi_row.min(count - 1);
    }
}

fn format_poi_row_summary(row_idx: usize, total: usize, poi: &CriticalPoint) -> String {
    let (x, y) = (
        format_table_value(poi.point[0]),
        format_table_value(poi.point[1]),
    );
    format!(
        "Critical point {} of {total}: {} ({}), X = {x}, Y = {y}",
        row_idx + 1,
        poi.kind.label(),
        poi.series_summary()
    )
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
        let msg = format_poi_row_summary(state.selected_poi_row, pois.len(), poi);
        emit_plot_live_message(ctx, msg, false);
    }
    if consume_copy_event(ctx)
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
    let total = pois.len();
    ui.scope_builder(
        UiBuilder::new().id_salt(plot_id.with("poi_table_a11y")),
        |ui| {
            let table_id = ui.unique_id();
            register_table_node(ui.ctx(), table_id, "Critical points table", total + 1, 5);
            ScrollArea::horizontal()
                .id_salt(plot_id.with("poi_table_hscroll"))
                .max_width(ui.available_width().max(320.0))
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.set_min_width(width);
                    render_poi_header(ui, table_id, width);
                    ui.separator();
                    ui.style_mut().always_scroll_the_only_direction = false;
                    let mut vs = ScrollArea::vertical()
                        .id_salt(plot_id.with("poi_table_vscroll"))
                        .max_height(MAX_H)
                        .auto_shrink([false, false]);
                    if let Some(off) = state.pending_scroll_offset.take() {
                        vs = vs.vertical_scroll_offset(off);
                    }
                    vs.show_rows(ui, ROW_H, total, |ui, range| {
                        ui.set_min_width(width);
                        if state.scroll_to_selected && !range.contains(&state.selected_poi_row) {
                            let pitch = ROW_H + ui.spacing().item_spacing.y;
                            let target =
                                (state.selected_poi_row as f32 * pitch - MAX_H * 0.4).max(0.0);
                            state.pending_scroll_offset = Some(target);
                            state.scroll_to_selected = false;
                            ui.ctx().request_repaint();
                        }
                        for r in range {
                            paint_poi_row(ui, table_id, state, r, total, &pois[r], width);
                        }
                    });
                });
            set_accesskit_bounds(ui.ctx(), table_id, ui.min_rect());
        },
    );
}

fn render_poi_header(ui: &mut Ui, table_id: Id, width: f32) {
    let _ = scope_table_row(
        ui,
        table_id,
        "poi_hdr_row",
        vec2(width, HDR_H),
        Sense::hover(),
        0,
        |hdr_ui, rect, _| {
            let faint = hdr_ui.visuals().faint_bg_color;
            hdr_ui.painter().rect_filled(rect, 3.0_f32, faint);
            let hdr_bg = composite_over(faint, hdr_ui.visuals().window_fill);
            let specs = [
                (0.0, IDX_W, "#"),
                (IDX_W, COL_W, "Category"),
                (IDX_W + COL_W, SERIES_COL_W, "Series"),
                (IDX_W + COL_W + SERIES_COL_W, COL_W, "X"),
                (IDX_W + 2.0 * COL_W + SERIES_COL_W, COL_W, "Y"),
            ];
            let rects: Vec<Rect> = {
                let cells = RowCells {
                    ui: hdr_ui,
                    rect,
                    height: HDR_H,
                    font: TextStyle::Body.resolve(hdr_ui.style()),
                };
                let fg = hdr_ui.visuals().strong_text_color();
                specs
                    .iter()
                    .map(|&(off, w, txt)| cells.paint_span(off, w, txt, fg, None))
                    .collect()
            };
            let row_id = hdr_ui.unique_id();
            for (col_idx, ((_, _, name), cell_rect)) in specs.iter().zip(rects).enumerate() {
                let _ = register_cell_node(
                    hdr_ui,
                    row_id.with(("poi_hdr_col", col_idx)),
                    cell_rect,
                    egui::accesskit::Role::ColumnHeader,
                    col_idx,
                    (*name).to_owned(),
                    false,
                    hdr_bg,
                );
            }
            ("Critical point column headers".to_owned(), None)
        },
    );
}

fn paint_poi_row(
    ui: &mut Ui,
    table_id: Id,
    state: &mut DataTableState,
    row_idx: usize,
    total: usize,
    poi: &CriticalPoint,
    width: f32,
) {
    let _ = scope_table_row(
        ui,
        table_id,
        ("poi_row", row_idx),
        vec2(width, ROW_H),
        Sense::click(),
        row_idx + 1,
        |row_ui, rect, pre_resp| {
            let summary = format_poi_row_summary(row_idx, total, poi);
            if pre_resp.clicked() {
                state.selected_poi_row = row_idx;
                emit_plot_live_message(row_ui.ctx(), summary.clone(), false);
            }
            let sel = row_idx == state.selected_poi_row;
            let row_bg = paint_poi_row_bg(row_ui, state, rect, pre_resp, row_idx);
            let dot_color = accessible_marker_color(row_ui.visuals(), POI_DOT_GRAY, row_bg);
            let series_txt = poi.series_summary();
            let x_txt = format_table_value(poi.point[0]);
            let y_txt = format_table_value(poi.point[1]);
            let (idx_rect, cat_rect, series_rect, x_rect, y_rect) = {
                let cells = RowCells {
                    ui: row_ui,
                    rect,
                    height: ROW_H,
                    font: TextStyle::Monospace.resolve(row_ui.style()),
                };
                let fg = if sel {
                    accessible_marker_color(
                        row_ui.visuals(),
                        row_ui.visuals().selection.stroke.color,
                        row_bg,
                    )
                } else {
                    row_ui.visuals().text_color()
                };
                let idx_rect = cells.paint_span(
                    0.0,
                    IDX_W,
                    &(row_idx + 1).to_string(),
                    row_ui.visuals().weak_text_color(),
                    None,
                );
                let cat_rect =
                    cells.paint_span(IDX_W, COL_W, poi.kind.label(), fg, Some(dot_color));
                let series_rect =
                    cells.paint_span(IDX_W + COL_W, SERIES_COL_W, &series_txt, fg, None);
                let x_rect =
                    cells.paint_span(IDX_W + COL_W + SERIES_COL_W, COL_W, &x_txt, fg, None);
                let y_rect =
                    cells.paint_span(IDX_W + 2.0 * COL_W + SERIES_COL_W, COL_W, &y_txt, fg, None);
                (idx_rect, cat_rect, series_rect, x_rect, y_rect)
            };
            let row_id = row_ui.unique_id();
            let _ = register_cell_node(
                row_ui,
                row_id.with("poi_row_hdr"),
                idx_rect,
                egui::accesskit::Role::RowHeader,
                0,
                format!("Point {}", row_idx + 1),
                false,
                row_bg,
            );
            let _ = register_cell_node(
                row_ui,
                row_id.with("poi_cat"),
                cat_rect,
                egui::accesskit::Role::Cell,
                1,
                format!("Category: {}", poi.kind.label()),
                false,
                row_bg,
            );
            let _ = register_cell_node(
                row_ui,
                row_id.with("poi_series"),
                series_rect,
                egui::accesskit::Role::Cell,
                2,
                format!("Series: {series_txt}"),
                true,
                row_bg,
            )
            .accessible_tooltip(series_txt.as_str());
            let _ = register_cell_node(
                row_ui,
                row_id.with("poi_x"),
                x_rect,
                egui::accesskit::Role::Cell,
                3,
                format!("X: {x_txt}"),
                false,
                row_bg,
            );
            let _ = register_cell_node(
                row_ui,
                row_id.with("poi_y"),
                y_rect,
                egui::accesskit::Role::Cell,
                4,
                format!("Y: {y_txt}"),
                false,
                row_bg,
            );
            (summary, Some(sel))
        },
    );
}

fn paint_poi_row_bg(
    ui: &Ui,
    state: &mut DataTableState,
    rect: Rect,
    resp: &egui::Response,
    row_idx: usize,
) -> Color32 {
    let win_bg = ui.visuals().window_fill;
    let row_bg = if row_idx == state.selected_poi_row {
        let sel_bg = ui.visuals().selection.bg_fill;
        ui.painter().rect_filled(rect, 2.0_f32, sel_bg);
        let stroke = Stroke::new(1.0_f32, ui.visuals().selection.stroke.color);
        ui.painter()
            .rect_stroke(rect, 2.0_f32, stroke, egui::StrokeKind::Inside);
        if state.scroll_to_selected {
            resp.scroll_to_me(Some(Align::Center));
            state.scroll_to_selected = false;
        }
        composite_over(sel_bg, win_bg)
    } else if resp.hovered() {
        let bg = ui
            .visuals()
            .widgets
            .hovered
            .bg_fill
            .gamma_multiply(0.35_f32);
        ui.painter().rect_filled(rect, 2.0_f32, bg);
        composite_over(bg, win_bg)
    } else if row_idx % 2 == 1 {
        let faint = ui.visuals().faint_bg_color;
        ui.painter().rect_filled(rect, 0.0_f32, faint);
        composite_over(faint, win_bg)
    } else {
        win_bg
    };
    if resp.has_focus() {
        let ring = focus_ring_stroke(ui.visuals(), row_bg);
        ui.painter()
            .rect_stroke(rect, 2.0_f32, ring, egui::StrokeKind::Inside);
    }
    row_bg
}

fn render_poi_footer(
    ui: &mut Ui,
    plot_id: Id,
    state: &mut DataTableState,
    pois: &[CriticalPoint],
) -> bool {
    ui.separator();
    let has_pois = !pois.is_empty();
    let disabled_reason = "No critical points available to copy";
    let mut close_clicked = false;
    ui.horizontal(|ui| {
        let copy_tsv = modal_action_button(ui, has_pois, "Copy TSV", disabled_reason);
        if has_pois && copy_tsv.clicked() {
            let tsv = format_pois_tsv(pois);
            copy_to_clipboard(ui.ctx(), plot_id, state, "critical points (TSV)", tsv);
        }
        let copy_csv = modal_action_button(ui, has_pois, "Copy CSV", disabled_reason);
        if has_pois && copy_csv.clicked() {
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
    if let Some((ref msg, is_err)) = state.status_message {
        let _ = status_message_label(ui, msg, is_err);
    }
    close_clicked
}
