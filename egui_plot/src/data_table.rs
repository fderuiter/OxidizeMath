//! Virtualized multi-column and series-tabbed data table modal for [`crate::Plot`].

#[path = "data_table_format.rs"]
pub(crate) mod format;
#[path = "data_table_poi.rs"]
mod poi_view;
#[path = "data_table_3d.rs"]
mod table_3d;

pub(crate) use format::{
    TableSeriesRef, TableSeriesRef3D, format_table_value, format_tsv, format_tsv_3d, series_share_x,
};
pub(crate) use table_3d::show_data_table_modal_3d;

use egui::{
    Align, Align2, Color32, Context, FontId, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense,
    Stroke, TextStyle, Ui, UiBuilder, pos2, vec2,
};

use crate::{
    ModalLifecycle,
    a11y::{
        AccessibleTooltipExt as _, accessible_marker_color, composite_over, focus_ring_stroke,
        focusable_tab_button, modal_action_button, register_cell_node, register_table_node,
        scope_table_row, set_accesskit_bounds, status_message_label, step_tablist_on_arrow_keys,
    },
    emit_plot_live_message,
    export::{
        copy_text_to_clipboard_with_callback, format_csv, read_text_from_clipboard_with_callback,
    },
    file_io::enqueue_csv_text_import,
    poi::CriticalPoint,
};

pub(crate) const ROW_H: f32 = 22.0;
pub(crate) const HDR_H: f32 = 26.0;
pub(crate) const MAX_H: f32 = 280.0;
pub(crate) const IDX_W: f32 = 56.0;
pub(crate) const COL_W: f32 = 136.0;

/// Active tab inside [`show_data_table_modal_with_shared_pois`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DataTableViewMode {
    /// Sampled series coordinates (`X`, `Y`).
    #[default]
    SeriesData,
    /// Detected critical points (roots, y-intercepts, extrema, intersections).
    CriticalPoints,
}

/// Persistent UI state for a plot's interactive Data Table modal.
#[derive(Clone, Default)]
pub struct DataTableState {
    pub view_mode: DataTableViewMode,
    pub selected_series: usize,
    pub selected_row: usize,
    pub selected_poi_row: usize,
    pub scroll_to_selected: bool,
    pub pending_scroll_offset: Option<f32>,
    pub status_message: Option<(String, bool)>,
}

/// Temp storage key used to deliver asynchronous clipboard results to a plot's Data Table.
#[must_use]
pub fn data_table_clipboard_status_id(plot_id: Id) -> Id {
    plot_id.with("data_table_clipboard_status")
}

pub(crate) fn set_data_table_allow_csv_import(ctx: &Context, plot_id: Id, allow: bool) {
    ctx.data_mut(|d| d.insert_temp(plot_id.with("allow_csv_import"), allow));
}

fn data_table_allows_csv_import(ctx: &Context, plot_id: Id) -> bool {
    ctx.data(|d| {
        d.get_temp::<bool>(plot_id.with("allow_csv_import"))
            .unwrap_or(true)
    })
}

fn handle_data_table_paste_events(ctx: &Context, plot_id: Id, state: &mut DataTableState) {
    if !data_table_allows_csv_import(ctx, plot_id) {
        return;
    }
    let pasted: Vec<String> = ctx.input_mut(|i| {
        let mut collected = Vec::new();
        i.events.retain(|evt| {
            if let egui::Event::Paste(text) = evt {
                collected.push(text.clone());
                false
            } else {
                true
            }
        });
        collected
    });
    for text in pasted {
        let status = enqueue_csv_text_import(ctx, plot_id, "clipboard", &text);
        if !status.1 {
            state.view_mode = DataTableViewMode::SeriesData;
        }
        state.status_message = Some(status);
    }
}

fn paste_from_clipboard_into_table(ctx: &Context, plot_id: Id, st: &mut DataTableState) {
    let status_id = data_table_clipboard_status_id(plot_id);
    let cb_ctx = ctx.clone();
    match read_text_from_clipboard_with_callback(move |res| {
        let status = match res {
            Ok(text) => enqueue_csv_text_import(&cb_ctx, plot_id, "clipboard", &text),
            Err(err) => (err, true),
        };
        cb_ctx.data_mut(|d| d.insert_temp(status_id, status));
        cb_ctx.request_repaint();
    }) {
        Ok(()) => {
            let fallback = ("Reading CSV/TSV from clipboard...".to_owned(), false);
            if let Some(resolved) = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(status_id)) {
                if !resolved.1 {
                    st.view_mode = DataTableViewMode::SeriesData;
                }
                st.status_message = Some(resolved);
            } else {
                emit_plot_live_message(ctx, fallback.0.clone(), false);
                st.status_message = Some(fallback);
            }
        }
        Err(err) => {
            let _ = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(status_id));
            emit_plot_live_message(ctx, err.clone(), true);
            st.status_message = Some((err, true));
        }
    }
}

/// Drain any completed asynchronous clipboard status for `plot_id` into `state`.
pub fn poll_data_table_clipboard_status(ctx: &Context, plot_id: Id, state: &mut DataTableState) {
    let id = data_table_clipboard_status_id(plot_id);
    if let Some(status) = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(id)) {
        emit_plot_live_message(ctx, status.0.clone(), status.1);
        state.status_message = Some(status);
    }
}

pub(crate) fn copy_to_clipboard(
    ctx: &Context,
    plot_id: Id,
    st: &mut DataTableState,
    label: &str,
    text: String,
) {
    ctx.copy_text(text.clone());
    let id = data_table_clipboard_status_id(plot_id);
    let (cb_ctx, lbl) = (ctx.clone(), label.to_owned());
    match copy_text_to_clipboard_with_callback(&text, move |res| {
        let msg = match res {
            Ok(()) => (format!("Copied {lbl} to clipboard."), false),
            Err(e) => (format!("Clipboard copy failed: {e}"), true),
        };
        cb_ctx.data_mut(|d| d.insert_temp(id, msg));
        cb_ctx.request_repaint();
    }) {
        Ok(()) => {
            let fallback = (format!("Copying {label} to clipboard..."), false);
            let status = ctx.data_mut(|d| d.remove_temp(id)).unwrap_or(fallback);
            emit_plot_live_message(ctx, status.0.clone(), status.1);
            st.status_message = Some(status);
        }
        Err(e) => {
            let _ = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(id));
            let status = (format!("Clipboard copy failed: {e}"), true);
            emit_plot_live_message(ctx, status.0.clone(), status.1);
            st.status_message = Some(status);
        }
    }
}

pub(crate) struct RowCells<'a> {
    pub(crate) ui: &'a Ui,
    pub(crate) rect: Rect,
    pub(crate) height: f32,
    pub(crate) font: FontId,
}

impl RowCells<'_> {
    pub(crate) fn paint(
        &self,
        col_idx: usize,
        text: &str,
        fg: Color32,
        dot: Option<Color32>,
    ) -> Rect {
        let (x_off, w) = if col_idx == 0 {
            (0.0, IDX_W)
        } else {
            (IDX_W + COL_W * (col_idx - 1) as f32, COL_W)
        };
        self.paint_span(x_off, w, text, fg, dot)
    }

    pub(crate) fn paint_span(
        &self,
        x_offset: f32,
        w: f32,
        text: &str,
        fg: Color32,
        dot: Option<Color32>,
    ) -> Rect {
        let cell = Rect::from_min_size(
            pos2(self.rect.min.x + x_offset, self.rect.min.y),
            vec2(w, self.height),
        );
        let clip = cell.intersect(self.ui.clip_rect());
        if clip.is_positive() {
            let painter = self.ui.painter().with_clip_rect(clip);
            let mut x = cell.min.x + 6.0;
            if let Some(c) = dot.filter(|c| *c != Color32::TRANSPARENT) {
                painter.circle_filled(pos2(x + 4.0, cell.center().y), 4.0, c);
                x += 12.0;
            }
            painter.text(
                pos2(x, cell.center().y),
                Align2::LEFT_CENTER,
                text,
                self.font.clone(),
                fg,
            );
        }
        cell
    }
}

struct TableView<'a> {
    plot_id: Id,
    series: &'a [TableSeriesRef<'a>],
    shared_x: bool,
    allow_csv_import: bool,
    state: &'a mut DataTableState,
}

impl TableView<'_> {
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
        let r = self.state.selected_row;
        if self.shared_x || self.series.len() == 1 {
            let _ = self.series.first()?.points.get(r)?;
            Some(format::write_aligned_tsv(self.series, r..r + 1))
        } else {
            let s = self.series.get(self.state.selected_series)?;
            let pt = s.points.get(r)?;
            let hdr = format::sanitize_tsv_label(&s.display_name(self.state.selected_series));
            Some(format!("X\t{hdr}\n{}\t{}\n", pt[0], pt[1]))
        }
    }

    fn format_row_summary(&self, row_idx: usize, rows: usize) -> Option<String> {
        if self.shared_x {
            let first = self.series.first()?;
            let x_pt = first.points.get(row_idx)?;
            let x = format_table_value(x_pt[0]);
            let mut parts = Vec::with_capacity(self.series.len());
            for (i, s) in self.series.iter().enumerate() {
                let pt = s.points.get(row_idx)?;
                parts.push(format!(
                    "{} = {}",
                    s.display_name(i),
                    format_table_value(pt[1])
                ));
            }
            Some(format!(
                "Row {} of {rows}: X = {x}, {}",
                row_idx + 1,
                parts.join(", ")
            ))
        } else {
            let idx = if self.series.len() == 1 {
                0
            } else {
                self.state.selected_series
            };
            let s = self.series.get(idx)?;
            let pt = s.points.get(row_idx)?;
            let (x, y) = (format_table_value(pt[0]), format_table_value(pt[1]));
            let name = s.display_name(idx);
            Some(format!(
                "Row {} of {rows}: X = {x}, {name} = {y}",
                row_idx + 1
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
            && let Some(msg) = self.format_row_summary(self.state.selected_row, rows)
        {
            emit_plot_live_message(ctx, msg, false);
        }
        if ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)))
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
        active: &[TableSeriesRef<'_>],
        cols: &[(String, Color32)],
        width: f32,
    ) {
        let _ = scope_table_row(
            ui,
            table_id,
            ("series_row", row_idx),
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
                let x_txt = format_table_value(active[0].points[row_idx][0]);
                let y_txts: Vec<String> = active
                    .iter()
                    .map(|s| format_table_value(s.points[row_idx][1]))
                    .collect();
                let (idx_rect, x_rect, y_rects) = {
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
                    let idx_rect = cells.paint(
                        0,
                        &(row_idx + 1).to_string(),
                        row_ui.visuals().weak_text_color(),
                        None,
                    );
                    let x_rect = cells.paint(1, &x_txt, fg, None);
                    let y_rects: Vec<Rect> = y_txts
                        .iter()
                        .enumerate()
                        .map(|(i, y_txt)| cells.paint(i + 2, y_txt, fg, None))
                        .collect();
                    (idx_rect, x_rect, y_rects)
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
                let _ = register_cell_node(
                    row_ui,
                    row_id.with("cell_x"),
                    x_rect,
                    egui::accesskit::Role::Cell,
                    1,
                    format!("X: {x_txt}"),
                    false,
                    row_bg,
                );
                for (i, y_rect) in y_rects.into_iter().enumerate() {
                    let col_name = cols.get(i).map_or("Y", |(name, _)| name.as_str());
                    let _ = register_cell_node(
                        row_ui,
                        row_id.with(("cell_y", i)),
                        y_rect,
                        egui::accesskit::Role::Cell,
                        i + 2,
                        format!("{col_name}: {}", y_txts[i]),
                        false,
                        row_bg,
                    );
                }
                (row_label, Some(sel))
            },
        );
    }

    fn render_grid(
        &mut self,
        ui: &mut Ui,
        active: &[TableSeriesRef<'_>],
        cols: &[(String, Color32)],
    ) {
        let rows = active[0].points.len();
        let width = IDX_W + COL_W * (cols.len() as f32 + 1.0);
        ui.scope_builder(
            UiBuilder::new().id_salt(self.plot_id.with("series_table_a11y")),
            |ui| {
                let table_id = ui.unique_id();
                register_table_node(
                    ui.ctx(),
                    table_id,
                    "Series data table",
                    rows + 1,
                    cols.len() + 2,
                );
                ScrollArea::horizontal()
                    .id_salt(self.plot_id.with("data_table_hscroll"))
                    .max_width(ui.available_width().max(320.0))
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.set_min_width(width);
                        render_header_row(ui, table_id, cols, width);
                        ui.separator();
                        ui.style_mut().always_scroll_the_only_direction = false;
                        let mut vs = ScrollArea::vertical()
                            .id_salt(self.plot_id.with("data_table_vscroll"))
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
                                self.paint_row(ui, table_id, r, rows, active, cols, width);
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
                let tsv = format_tsv(self.series);
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
                    format_csv(&ds),
                );
            }
            if self.allow_csv_import {
                let paste_btn = modal_action_button(ui, true, "Paste Clipboard", "");
                if paste_btn.clicked() {
                    paste_from_clipboard_into_table(ui.ctx(), self.plot_id, self.state);
                }
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
        ScrollArea::horizontal()
            .id_salt(self.plot_id.with("data_table_series_tabs"))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Series:").small().weak());
                    ui.scope_builder(
                        UiBuilder::new()
                            .id_salt(self.plot_id.with("series_tablist_a11y"))
                            .layout(egui::Layout::left_to_right(Align::Center)),
                        |ui| {
                            let tablist_id = ui.unique_id();
                            ui.ctx().accesskit_node_builder(tablist_id, |builder| {
                                builder.set_role(egui::accesskit::Role::TabList);
                                builder.set_label("Plot series");
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
    }

    fn render_body(&mut self, ui: &mut Ui) -> bool {
        if self.series.is_empty() {
            self.clamp(0);
            ui.label(
                RichText::new("No data points are currently available for this plot.").strong(),
            );
            let sub = if self.allow_csv_import {
                "Press Ctrl+V, click Paste Clipboard, or use Import CSV to plot tabular data."
            } else {
                "Run or reset the simulation, or enable a visible series to inspect values."
            };
            ui.label(RichText::new(sub).small().weak());
            return self.render_footer(ui, 0);
        }
        if self.series.len() > 1 && !self.shared_x {
            self.render_series_tabs(ui);
            ui.separator();
        }
        let multi = self.shared_x || self.series.len() == 1;
        let active = if multi {
            self.series
        } else {
            std::slice::from_ref(&self.series[self.state.selected_series])
        };
        let rows = active[0].points.len();
        self.clamp(rows);
        let cols: Vec<_> = if multi {
            self.series
                .iter()
                .enumerate()
                .map(|(i, s)| (s.display_name(i), s.color))
                .collect()
        } else {
            let s = &self.series[self.state.selected_series];
            vec![(s.display_name(self.state.selected_series), s.color)]
        };
        self.render_grid(ui, active, &cols);
        self.render_footer(ui, rows)
    }
}

fn render_header_row(ui: &mut Ui, table_id: Id, cols: &[(String, Color32)], width: f32) {
    let _ = scope_table_row(
        ui,
        table_id,
        "series_hdr_row",
        vec2(width, HDR_H),
        Sense::hover(),
        0,
        |hdr_ui, rect, _| {
            let faint = hdr_ui.visuals().faint_bg_color;
            hdr_ui.painter().rect_filled(rect, 3.0_f32, faint);
            let hdr_bg = composite_over(faint, hdr_ui.visuals().window_fill);
            let (idx_rect, x_rect, col_rects) = {
                let cells = RowCells {
                    ui: hdr_ui,
                    rect,
                    height: HDR_H,
                    font: TextStyle::Body.resolve(hdr_ui.style()),
                };
                let fg = hdr_ui.visuals().strong_text_color();
                let idx_rect = cells.paint(0, "#", fg, None);
                let x_rect = cells.paint(1, "X", fg, None);
                let col_rects: Vec<Rect> = cols
                    .iter()
                    .enumerate()
                    .map(|(i, (name, c))| {
                        let dot = accessible_marker_color(hdr_ui.visuals(), *c, hdr_bg);
                        cells.paint(i + 2, name, fg, Some(dot))
                    })
                    .collect();
                (idx_rect, x_rect, col_rects)
            };
            let row_id = hdr_ui.unique_id();
            let _ = register_cell_node(
                hdr_ui,
                row_id.with("hdr_idx"),
                idx_rect,
                egui::accesskit::Role::ColumnHeader,
                0,
                "#".to_owned(),
                false,
                hdr_bg,
            );
            let _ = register_cell_node(
                hdr_ui,
                row_id.with("hdr_x"),
                x_rect,
                egui::accesskit::Role::ColumnHeader,
                1,
                "X".to_owned(),
                false,
                hdr_bg,
            );
            for (i, ((name, _), cell_rect)) in cols.iter().zip(col_rects).enumerate() {
                let _ = register_cell_node(
                    hdr_ui,
                    row_id.with(("hdr_col", i)),
                    cell_rect,
                    egui::accesskit::Role::ColumnHeader,
                    i + 2,
                    name.clone(),
                    true,
                    hdr_bg,
                )
                .accessible_tooltip(name.as_str());
            }
            ("Column headers".to_owned(), None)
        },
    );
}

#[cfg(test)]
fn collect_nonempty_series(
    datasets: &[(String, Vec<[f64; 2]>, Color32)],
) -> Vec<TableSeriesRef<'_>> {
    datasets
        .iter()
        .filter(|(_, p, _)| !p.is_empty())
        .map(|(name, points, color)| TableSeriesRef {
            name,
            points,
            color: *color,
        })
        .collect()
}

/// Render the Data Table modal with both `Series Data` and `Critical Points` views.
#[cfg(test)]
pub(crate) fn show_data_table_modal_with_pois(
    ctx: &Context,
    plot_id: Id,
    window_title: &str,
    open: &mut bool,
    state: &mut DataTableState,
    datasets: &[(String, Vec<[f64; 2]>, Color32)],
    critical_points: &[CriticalPoint],
) {
    poll_data_table_clipboard_status(ctx, plot_id, state);
    if !*open {
        let _ = ModalLifecycle::new(plot_id.with("data_table_modal")).show(ctx, open, |_, _| {});
        return;
    }
    let series = collect_nonempty_series(datasets);
    render_data_table_modal_series(
        ctx,
        plot_id,
        window_title,
        open,
        state,
        &series,
        critical_points,
    );
}

/// Render the Data Table modal directly over shared coordinate buffers without cloning series vectors.
pub(crate) fn show_data_table_modal_with_shared_pois(
    ctx: &Context,
    plot_id: Id,
    window_title: &str,
    open: &mut bool,
    state: &mut DataTableState,
    datasets: &[(String, crate::SharedPoints, Color32)],
    critical_points: &[CriticalPoint],
) {
    poll_data_table_clipboard_status(ctx, plot_id, state);
    if !*open {
        let _ = ModalLifecycle::new(plot_id.with("data_table_modal")).show(ctx, open, |_, _| {});
        return;
    }
    let series: Vec<_> = datasets
        .iter()
        .filter(|(_, p, _)| !p.is_empty())
        .map(|(name, points, color)| TableSeriesRef {
            name,
            points: points.as_slice(),
            color: *color,
        })
        .collect();
    render_data_table_modal_series(
        ctx,
        plot_id,
        window_title,
        open,
        state,
        &series,
        critical_points,
    );
}

fn render_data_table_modal_series(
    ctx: &Context,
    plot_id: Id,
    window_title: &str,
    open: &mut bool,
    state: &mut DataTableState,
    series: &[TableSeriesRef<'_>],
    critical_points: &[CriticalPoint],
) {
    let shared_x = series.len() > 1 && series_share_x(series);
    let allow_csv_import = data_table_allows_csv_import(ctx, plot_id);
    preprocess_data_table_frame(
        ctx,
        plot_id,
        state,
        series,
        shared_x,
        allow_csv_import,
        critical_points,
    );
    let max_w = (ctx.content_rect().width() - 48.0).clamp(360.0, 740.0);
    ModalLifecycle::new(plot_id.with("data_table_modal"))
        .title(window_title)
        .default_width(520.0_f32.min(max_w))
        .max_width(max_w)
        .max_height(480.0)
        .show(ctx, open, |ui, frame_ctx| {
            render_mode_selector(ui, state, series.len(), critical_points.len());
            ui.separator();
            let should_close = match state.view_mode {
                DataTableViewMode::SeriesData => {
                    let mut view = TableView {
                        plot_id,
                        series,
                        shared_x,
                        allow_csv_import,
                        state,
                    };
                    view.render_body(ui)
                }
                DataTableViewMode::CriticalPoints => {
                    poi_view::render_poi_body(ui, plot_id, state, critical_points)
                }
            };
            if should_close {
                frame_ctx.close();
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn preprocess_data_table_frame(
    ctx: &Context,
    plot_id: Id,
    state: &mut DataTableState,
    series: &[TableSeriesRef<'_>],
    shared_x: bool,
    allow_csv_import: bool,
    critical_points: &[CriticalPoint],
) {
    handle_data_table_paste_events(ctx, plot_id, state);
    if series.is_empty() && !critical_points.is_empty() && state.status_message.is_none() {
        state.view_mode = DataTableViewMode::CriticalPoints;
    }
    if state.view_mode == DataTableViewMode::CriticalPoints {
        poi_view::clamp_poi_state(state, critical_points.len());
        poi_view::handle_poi_keys(ctx, plot_id, state, critical_points);
    } else if !series.is_empty() {
        let mut view = TableView {
            plot_id,
            series,
            shared_x,
            allow_csv_import,
            state,
        };
        let idx = if shared_x || series.len() == 1 {
            0
        } else {
            view.state.selected_series.min(series.len() - 1)
        };
        let rows = series[idx].points.len();
        view.clamp(rows);
        view.handle_keys(ctx, rows);
    }
}

fn render_mode_selector(
    ui: &mut Ui,
    state: &mut DataTableState,
    series_count: usize,
    poi_count: usize,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("View:").small().weak());
        ui.scope_builder(
            UiBuilder::new()
                .id_salt("data_table_mode_tablist")
                .layout(egui::Layout::left_to_right(Align::Center)),
            |ui| {
                let tablist_id = ui.unique_id();
                ui.ctx().accesskit_node_builder(tablist_id, |builder| {
                    builder.set_role(egui::accesskit::Role::TabList);
                    builder.set_label("Data table views");
                    builder.set_orientation(egui::accesskit::Orientation::Horizontal);
                });
                let modes = [
                    (
                        DataTableViewMode::SeriesData,
                        format!("Series Data ({series_count})"),
                    ),
                    (
                        DataTableViewMode::CriticalPoints,
                        format!("Critical Points ({poi_count})"),
                    ),
                ];
                let mut responses = Vec::with_capacity(modes.len());
                let mut target_idx = None;
                for (idx, (mode, lbl)) in modes.iter().enumerate() {
                    let resp = focusable_tab_button(ui, state.view_mode == *mode, lbl);
                    if resp.clicked() && state.view_mode != *mode {
                        target_idx = Some(idx);
                    }
                    responses.push(resp);
                }
                if target_idx.is_none() {
                    target_idx = step_tablist_on_arrow_keys(ui, &responses);
                }
                if let Some(idx) = target_idx {
                    let (next_mode, ref lbl) = modes[idx];
                    if state.view_mode != next_mode {
                        state.view_mode = next_mode;
                        state.scroll_to_selected = true;
                    }
                    emit_plot_live_message(ui.ctx(), format!("Selected {lbl} tab"), false);
                }
                set_accesskit_bounds(ui.ctx(), tablist_id, ui.min_rect());
            },
        );
    });
}

#[cfg(test)]
#[path = "data_table_tests.rs"]
mod tests;
