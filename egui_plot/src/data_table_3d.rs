//! Virtualized 3D `(X, Y, Z)` series-tabbed Data Table modal for [`crate::Plot`].

use egui::{
    Align, Color32, Context, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense, Stroke,
    TextStyle, Ui, vec2,
};

use super::{
    COL_W, DataTableState, HDR_H, IDX_W, MAX_H, ROW_H, RowCells, TableSeriesRef3D,
    copy_to_clipboard, format, format_table_value, format_tsv_3d, poll_data_table_clipboard_status,
};
use crate::{ModalLifecycle, Plot3DSeries, emit_plot_live_message, export::format_csv_3d};

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
            && let Some(pt) = self
                .series
                .get(self.state.selected_series)
                .and_then(|s| s.points.get(self.state.selected_row))
        {
            let x = format_table_value(pt[0]);
            let y = format_table_value(pt[1]);
            let z = format_table_value(pt[2]);
            let msg = format!(
                "Row {} of {rows}: X = {x}, Y = {y}, Z = {z}",
                self.state.selected_row + 1
            );
            emit_plot_live_message(ctx, msg, false);
        }
        if ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)))
            && let Some(tsv) = self.selected_row_tsv()
        {
            copy_to_clipboard(ctx, self.plot_id, self.state, "selected row (TSV)", tsv);
        }
    }

    fn paint_row_bg(&mut self, ui: &Ui, rect: Rect, resp: &egui::Response, row_idx: usize) {
        if row_idx == self.state.selected_row {
            ui.painter()
                .rect_filled(rect, 2.0_f32, ui.visuals().selection.bg_fill);
            let stroke = Stroke::new(1.0_f32, ui.visuals().selection.stroke.color);
            ui.painter()
                .rect_stroke(rect, 2.0_f32, stroke, egui::StrokeKind::Inside);
            if self.state.scroll_to_selected {
                resp.scroll_to_me(Some(Align::Center));
                self.state.scroll_to_selected = false;
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

    fn paint_row(
        &mut self,
        ui: &mut Ui,
        row_idx: usize,
        active: &TableSeriesRef3D<'_>,
        width: f32,
    ) {
        let (rect, resp) = ui.allocate_exact_size(vec2(width, ROW_H), Sense::click());
        if resp.clicked() {
            self.state.selected_row = row_idx;
        }
        self.paint_row_bg(ui, rect, &resp, row_idx);
        let cells = RowCells {
            ui,
            rect,
            height: ROW_H,
            font: TextStyle::Monospace.resolve(ui.style()),
        };
        let fg = if row_idx == self.state.selected_row {
            ui.visuals().selection.stroke.color
        } else {
            ui.visuals().text_color()
        };
        let pt = active.points[row_idx];
        let _ = cells.paint(
            0,
            &(row_idx + 1).to_string(),
            ui.visuals().weak_text_color(),
            None,
        );
        let _ = cells.paint(1, &format_table_value(pt[0]), fg, None);
        let _ = cells.paint(2, &format_table_value(pt[1]), fg, None);
        let _ = cells.paint(3, &format_table_value(pt[2]), fg, None);
    }

    fn render_grid(&mut self, ui: &mut Ui, active: &TableSeriesRef3D<'_>, series_color: Color32) {
        let rows = active.points.len();
        let width = IDX_W + COL_W * 3.0;
        ScrollArea::horizontal()
            .id_salt(self.plot_id.with("data_table_3d_hscroll"))
            .max_width(ui.available_width().max(320.0))
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.set_min_width(width);
                render_header_row_3d(ui, series_color, width);
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
                    if self.state.scroll_to_selected && !range.contains(&self.state.selected_row) {
                        let pitch = ROW_H + ui.spacing().item_spacing.y;
                        let target =
                            (self.state.selected_row as f32 * pitch - MAX_H * 0.4).max(0.0);
                        self.state.pending_scroll_offset = Some(target);
                        self.state.scroll_to_selected = false;
                        ui.ctx().request_repaint();
                    }
                    for r in range {
                        self.paint_row(ui, r, active, width);
                    }
                });
            });
    }

    fn render_footer(&mut self, ui: &mut Ui, rows: usize) -> bool {
        if let Some((ref msg, is_err)) = self.state.status_message {
            let c = if is_err {
                ui.visuals().error_fg_color
            } else {
                ui.visuals().selection.stroke.color
            };
            ui.colored_label(c, msg);
        }
        ui.separator();
        let (mut close_clicked, has_data) = (false, !self.series.is_empty());
        ui.horizontal(|ui| {
            if ui
                .add_enabled(has_data, egui::Button::new("Copy TSV"))
                .clicked()
            {
                let tsv = format_tsv_3d(self.series);
                copy_to_clipboard(ui.ctx(), self.plot_id, self.state, "TSV table", tsv);
            }
            if ui
                .add_enabled(has_data, egui::Button::new("Copy CSV"))
                .clicked()
            {
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
        close_clicked
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
                    for (idx, item) in self.series.iter().enumerate() {
                        let lbl = format!("{} ({})", item.display_name(idx), item.points.len());
                        if ui
                            .selectable_label(self.state.selected_series == idx, lbl)
                            .clicked()
                        {
                            self.state.selected_series = idx;
                            self.state.selected_row =
                                self.state.selected_row.min(item.points.len() - 1);
                            self.state.scroll_to_selected = true;
                        }
                    }
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

fn render_header_row_3d(ui: &mut Ui, color: Color32, width: f32) {
    let (rect, _resp) = ui.allocate_exact_size(vec2(width, HDR_H), Sense::hover());
    ui.painter()
        .rect_filled(rect, 3.0_f32, ui.visuals().faint_bg_color);
    let cells = RowCells {
        ui,
        rect,
        height: HDR_H,
        font: TextStyle::Body.resolve(ui.style()),
    };
    let fg = ui.visuals().strong_text_color();
    let _ = cells.paint(0, "#", fg, None);
    let _ = cells.paint(1, "X", fg, Some(color));
    let _ = cells.paint(2, "Y", fg, None);
    let _ = cells.paint(3, "Z", fg, None);
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
    if !series.is_empty() {
        let idx = view.state.selected_series.min(series.len() - 1);
        let rows = series[idx].points.len();
        view.clamp(rows);
        view.handle_keys(ctx, rows);
    }
    let max_w = (ctx.content_rect().width() - 48.0).clamp(360.0, 740.0);
    ModalLifecycle::new(plot_id.with("data_table_modal"))
        .title(window_title)
        .default_width(520.0_f32.min(max_w))
        .max_width(max_w)
        .max_height(460.0)
        .show(ctx, open, |ui, frame_ctx| {
            if view.render_body(ui) {
                frame_ctx.close();
            }
        });
}
