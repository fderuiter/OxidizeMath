//! Virtualized 3D `(X, Y, Z)` series-tabbed Data Table modal for [`crate::Plot`].

use egui::{
    Align, Color32, Context, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense, Stroke,
    TextStyle, Ui, UiBuilder, vec2,
};

use super::{
    COL_W, DataTableState, HDR_H, IDX_W, MAX_H, ROW_H, RowCells, TableSeriesRef3D,
    consume_copy_event, copy_to_clipboard, format, format_table_value, format_tsv_3d,
    poll_data_table_clipboard_status,
};
use crate::{
    ModalLifecycle, Plot3DSeries,
    a11y::{
        accessible_marker_color, composite_over, focus_ring_stroke, focusable_tab_button,
        modal_action_button, register_cell_node, register_table_node, scope_table_row,
        set_accesskit_bounds, status_message_label, step_tablist_on_arrow_keys,
    },
    emit_plot_live_message,
    export::format_csv_3d,
};

struct TableView3D<'a> {
    plot_id: Id,
    series: &'a [TableSeriesRef3D<'a>],
    state: &'a mut DataTableState,
}

impl TableView3D<'_> {
    fn clamp(&mut self, rows: usize) {
        if self.series.is_empty() || rows == 0 {
            self.state.selected_series = 0;
            self.state.selected_row = 0;
            self.state.scroll_to_selected = false;
            self.state.pending_scroll_offset = None;
        } else {
            self.state.selected_series = self.state.selected_series.min(self.series.len() - 1);
            self.state.selected_row = self.state.selected_row.min(rows - 1);
        }
    }

    fn selected_row_tsv(&self) -> Option<String> {
        let s = self.series.get(self.state.selected_series)?;
        let pt = s.points.get(self.state.selected_row)?;
        if self.series.len() == 1 {
            Some(format!("X\tY\tZ\n{}\t{}\t{}\n", pt[0], pt[1], pt[2]))
        } else {
            let hdr = format::sanitize_tsv_label(&s.display_name(self.state.selected_series));
            Some(format!(
                "Dataset\tX\tY\tZ\n{hdr}\t{}\t{}\t{}\n",
                pt[0], pt[1], pt[2]
            ))
        }
    }

    fn format_row_summary(&self, row_idx: usize, rows: usize) -> Option<String> {
        let idx = self
            .state
            .selected_series
            .min(self.series.len().saturating_sub(1));
        let s = self.series.get(idx)?;
        let pt = s.points.get(row_idx)?;
        let name = s.display_name(idx);
        let x = format_table_value(pt[0]);
        let y = format_table_value(pt[1]);
        let z = format_table_value(pt[2]);
        Some(format!(
            "Row {} of {rows} ({name}): X = {x}, Y = {y}, Z = {z}",
            row_idx + 1
        ))
    }

    fn handle_keys(&mut self, ctx: &Context, rows: usize) {
        if rows == 0 {
            return;
        }
        let prev = self.state.selected_row;
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowDown)) {
            self.state.selected_row = (prev + 1).min(rows - 1);
            self.state.scroll_to_selected = true;
        } else if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowUp)) {
            self.state.selected_row = prev.saturating_sub(1);
            self.state.scroll_to_selected = true;
        }
        if self.state.selected_row != prev
            && let Some(msg) = self.format_row_summary(self.state.selected_row, rows)
        {
            emit_plot_live_message(ctx, msg, false);
        }
        if consume_copy_event(ctx)
            && let Some(tsv) = self.selected_row_tsv()
        {
            copy_to_clipboard(ctx, self.plot_id, self.state, "selected row (TSV)", tsv);
        }
    }

    fn paint_row_bg(
        &mut self,
        ui: &Ui,
        rect: Rect,
        resp: &egui::Response,
        row_idx: usize,
    ) -> Color32 {
        let win_bg = ui.visuals().window_fill;
        let row_bg = if row_idx == self.state.selected_row {
            let sel_bg = ui.visuals().selection.bg_fill;
            ui.painter().rect_filled(rect, 2.0_f32, sel_bg);
            let stroke = Stroke::new(1.0_f32, ui.visuals().selection.stroke.color);
            ui.painter()
                .rect_stroke(rect, 2.0_f32, stroke, egui::StrokeKind::Inside);
            if self.state.scroll_to_selected {
                resp.scroll_to_me(Some(Align::Center));
                self.state.scroll_to_selected = false;
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

    fn paint_row(
        &mut self,
        ui: &mut Ui,
        table_id: Id,
        row_idx: usize,
        rows: usize,
        active: &TableSeriesRef3D<'_>,
        width: f32,
    ) {
        let _ = scope_table_row(
            ui,
            table_id,
            ("series_3d_row", row_idx),
            vec2(width, ROW_H),
            Sense::click(),
            row_idx + 1,
            |row_ui, rect, pre_resp| {
                if pre_resp.clicked() {
                    self.state.selected_row = row_idx;
                    if let Some(msg) = self.format_row_summary(row_idx, rows) {
                        emit_plot_live_message(row_ui.ctx(), msg, false);
                    }
                }
                let sel = row_idx == self.state.selected_row;
                let row_label = self
                    .format_row_summary(row_idx, rows)
                    .unwrap_or_else(|| format!("Row {} of {rows}", row_idx + 1));
                let row_bg = self.paint_row_bg(row_ui, rect, pre_resp, row_idx);
                let pt = active.points[row_idx];
                let coords = [
                    ("X", format_table_value(pt[0])),
                    ("Y", format_table_value(pt[1])),
                    ("Z", format_table_value(pt[2])),
                ];
                let (idx_rect, coord_rects) = {
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
                    let idx_fg = accessible_marker_color(
                        row_ui.visuals(),
                        row_ui.visuals().weak_text_color(),
                        row_bg,
                    );
                    let idx_rect = cells.paint(0, &(row_idx + 1).to_string(), idx_fg, None);
                    let coord_rects = [
                        cells.paint(1, &coords[0].1, fg, None),
                        cells.paint(2, &coords[1].1, fg, None),
                        cells.paint(3, &coords[2].1, fg, None),
                    ];
                    (idx_rect, coord_rects)
                };
                let row_id = row_ui.unique_id();
                let _ = register_cell_node(
                    row_ui,
                    row_id.with("row_hdr"),
                    idx_rect,
                    egui::accesskit::Role::RowHeader,
                    0,
                    format!("Row {}", row_idx + 1),
                    false,
                    row_bg,
                );
                for (i, ((axis, val), cell_rect)) in coords.iter().zip(coord_rects).enumerate() {
                    let _ = register_cell_node(
                        row_ui,
                        row_id.with(("cell_3d", i)),
                        cell_rect,
                        egui::accesskit::Role::Cell,
                        i + 1,
                        format!("{axis}: {val}"),
                        false,
                        row_bg,
                    );
                }
                (row_label, Some(sel))
            },
        );
    }

    fn render_grid(&mut self, ui: &mut Ui, active: &TableSeriesRef3D<'_>, series_color: Color32) {
        let rows = active.points.len();
        let width = IDX_W + COL_W * 3.0;
        ui.scope_builder(
            UiBuilder::new().id_salt(self.plot_id.with("series_table_3d_a11y")),
            |ui| {
                let table_id = ui.unique_id();
                register_table_node(ui.ctx(), table_id, "3D series data table", rows + 1, 4);
                ScrollArea::horizontal()
                    .id_salt(self.plot_id.with("data_table_3d_hscroll"))
                    .max_width(ui.available_width().max(320.0))
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.set_min_width(width);
                        render_header_row_3d(ui, table_id, series_color, width);
                        ui.separator();
                        ui.style_mut().always_scroll_the_only_direction = false;
                        let mut vs = ScrollArea::vertical()
                            .id_salt(self.plot_id.with("data_table_3d_vscroll"))
                            .max_height(MAX_H)
                            .auto_shrink([false, false]);
                        if let Some(off) = self.state.pending_scroll_offset.take() {
                            vs = vs.vertical_scroll_offset(off);
                        }
                        vs.show_rows(ui, ROW_H, rows, |ui, range| {
                            ui.set_min_width(width);
                            if self.state.scroll_to_selected
                                && !range.contains(&self.state.selected_row)
                            {
                                let pitch = ROW_H + ui.spacing().item_spacing.y;
                                let target =
                                    (self.state.selected_row as f32 * pitch - MAX_H * 0.4).max(0.0);
                                self.state.pending_scroll_offset = Some(target);
                                self.state.scroll_to_selected = false;
                                ui.ctx().request_repaint();
                            }
                            for r in range {
                                self.paint_row(ui, table_id, r, rows, active, width);
                            }
                        });
                    });
                set_accesskit_bounds(ui.ctx(), table_id, ui.min_rect());
            },
        );
    }

    fn render_footer(&mut self, ui: &mut Ui, rows: usize) -> bool {
        ui.separator();
        let (mut close_clicked, has_data) = (false, !self.series.is_empty());
        let disabled_reason = "No series data points available to copy";
        ui.horizontal(|ui| {
            let copy_tsv = modal_action_button(ui, has_data, "Copy TSV", disabled_reason);
            if has_data && copy_tsv.clicked() {
                let tsv = format_tsv_3d(self.series);
                copy_to_clipboard(ui.ctx(), self.plot_id, self.state, "TSV table", tsv);
            }
            let copy_csv = modal_action_button(ui, has_data, "Copy CSV", disabled_reason);
            if has_data && copy_csv.clicked() {
                let ds: Vec<_> = self
                    .series
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.display_name(i), s.points.to_vec()))
                    .collect();
                copy_to_clipboard(
                    ui.ctx(),
                    self.plot_id,
                    self.state,
                    "CSV dataset",
                    format_csv_3d(&ds),
                );
            }
            close_clicked = ui.button("Close").clicked();
            if has_data {
                let hint = format!(
                    "Row {}/{rows} (↑/↓ to navigate)",
                    self.state.selected_row + 1
                );
                ui.label(RichText::new(hint).small().weak());
            }
        });
        if let Some((ref msg, is_err)) = self.state.status_message {
            let _ = status_message_label(ui, msg, is_err);
        }
        close_clicked
    }

    fn select_series_tab(&mut self, ctx: &Context, idx: usize, label: &str) {
        if let Some(item) = self.series.get(idx) {
            self.state.selected_series = idx;
            self.state.selected_row = self.state.selected_row.min(item.points.len() - 1);
            self.state.scroll_to_selected = true;
            emit_plot_live_message(ctx, format!("Selected series tab: {label}"), false);
        }
    }

    fn render_series_tabs(&mut self, ui: &mut Ui) {
        if self.series.len() <= 1 {
            return;
        }
        ScrollArea::horizontal()
            .id_salt(self.plot_id.with("data_table_3d_series_tabs"))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Series:").small().weak());
                    ui.scope_builder(
                        UiBuilder::new()
                            .id_salt(self.plot_id.with("series_3d_tablist_a11y"))
                            .layout(egui::Layout::left_to_right(Align::Center)),
                        |ui| {
                            let tablist_id = ui.unique_id();
                            ui.ctx().accesskit_node_builder(tablist_id, |builder| {
                                builder.set_role(egui::accesskit::Role::TabList);
                                builder.set_label("3D plot series");
                                builder.set_orientation(egui::accesskit::Orientation::Horizontal);
                            });
                            let mut labels = Vec::with_capacity(self.series.len());
                            let mut responses = Vec::with_capacity(self.series.len());
                            let mut clicked_idx = None;
                            for (idx, item) in self.series.iter().enumerate() {
                                let lbl =
                                    format!("{} ({})", item.display_name(idx), item.points.len());
                                let resp = focusable_tab_button(
                                    ui,
                                    self.state.selected_series == idx,
                                    &lbl,
                                );
                                if resp.clicked() {
                                    clicked_idx = Some(idx);
                                }
                                labels.push(lbl);
                                responses.push(resp);
                            }
                            if let Some(idx) = clicked_idx {
                                self.select_series_tab(ui.ctx(), idx, &labels[idx]);
                            } else if let Some(idx) = step_tablist_on_arrow_keys(ui, &responses) {
                                self.select_series_tab(ui.ctx(), idx, &labels[idx]);
                            }
                            set_accesskit_bounds(ui.ctx(), tablist_id, ui.min_rect());
                        },
                    );
                });
            });
        ui.separator();
    }

    fn render_body(&mut self, ui: &mut Ui) -> bool {
        if self.series.is_empty() {
            self.clamp(0);
            ui.label(
                RichText::new("No data points are currently available for this plot.").strong(),
            );
            let sub = "Run or reset the simulation, or enable a visible series to inspect values.";
            ui.label(RichText::new(sub).small().weak());
            return self.render_footer(ui, 0);
        }
        self.render_series_tabs(ui);
        let sel = self.state.selected_series.min(self.series.len() - 1);
        let active = self.series[sel];
        let rows = active.points.len();
        self.clamp(rows);
        self.render_grid(ui, &active, active.color);
        self.render_footer(ui, rows)
    }
}

fn render_header_row_3d(ui: &mut Ui, table_id: Id, color: Color32, width: f32) {
    let _ = scope_table_row(
        ui,
        table_id,
        "series_3d_hdr_row",
        vec2(width, HDR_H),
        Sense::hover(),
        0,
        |hdr_ui, rect, _| {
            let faint = hdr_ui.visuals().faint_bg_color;
            hdr_ui.painter().rect_filled(rect, 3.0_f32, faint);
            let hdr_bg = composite_over(faint, hdr_ui.visuals().window_fill);
            let dot = accessible_marker_color(hdr_ui.visuals(), color, hdr_bg);
            let specs = [
                (0, "#", None),
                (1, "X", Some(dot)),
                (2, "Y", None),
                (3, "Z", None),
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
                    .map(|&(col, txt, marker)| cells.paint(col, txt, fg, marker))
                    .collect()
            };
            let row_id = hdr_ui.unique_id();
            for ((col_idx, name, _), cell_rect) in specs.iter().zip(rects) {
                let _ = register_cell_node(
                    hdr_ui,
                    row_id.with(("hdr_3d_col", *col_idx)),
                    cell_rect,
                    egui::accesskit::Role::ColumnHeader,
                    *col_idx,
                    (*name).to_owned(),
                    false,
                    hdr_bg,
                );
            }
            ("3D column headers".to_owned(), None)
        },
    );
}

/// Render the closeable, virtualized 3D `(X, Y, Z)` Data Table modal when `*open` is `true`.
pub fn show_data_table_modal_3d(
    ctx: &Context,
    plot_id: Id,
    window_title: &str,
    open: &mut bool,
    state: &mut DataTableState,
    series_3d: &[Plot3DSeries],
) {
    poll_data_table_clipboard_status(ctx, plot_id, state);
    if !*open {
        let _ = ModalLifecycle::new(plot_id.with("data_table_modal")).show(ctx, open, |_, _| {});
        return;
    }
    let series: Vec<_> = series_3d
        .iter()
        .filter(|s| !s.world.is_empty())
        .map(|s| TableSeriesRef3D {
            name: &s.name,
            points: &s.world,
            color: s.color,
        })
        .collect();
    let mut view = TableView3D {
        plot_id,
        series: &series,
        state,
    };
    let max_w = (ctx.content_rect().width() - 48.0).clamp(360.0, 740.0);
    ModalLifecycle::new(plot_id.with("data_table_modal"))
        .title(window_title)
        .default_width(520.0_f32.min(max_w))
        .max_width(max_w)
        .max_height(460.0)
        .show(ctx, open, |ui, frame_ctx| {
            if !series.is_empty() {
                let idx = view.state.selected_series.min(series.len() - 1);
                let rows = series[idx].points.len();
                view.clamp(rows);
                if frame_ctx.is_top_modal {
                    view.handle_keys(ui.ctx(), rows);
                }
            }
            if view.render_body(ui) {
                frame_ctx.close();
            }
        });
}
