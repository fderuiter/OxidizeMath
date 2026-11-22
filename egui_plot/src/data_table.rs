//! Virtualized multi-column and series-tabbed data table modal for [`crate::Plot`].

#[path = "data_table_format.rs"]
pub(crate) mod format;
#[path = "data_table_poi.rs"]
mod poi_view;
#[path = "data_table_3d.rs"]
mod table_3d;

pub use format::{
    TableSeriesRef, TableSeriesRef3D, format_pois_csv, format_pois_tsv, format_table_value,
    format_tsv, format_tsv_3d, series_share_x,
};
pub use table_3d::show_data_table_modal_3d;

use egui::{
    Align, Align2, Color32, Context, FontId, Id, Key, Modifiers, Rect, RichText, ScrollArea, Sense,
    Stroke, TextStyle, Ui, pos2, vec2,
};

use crate::{
    ModalLifecycle, emit_plot_live_message,
    export::{copy_text_to_clipboard_with_callback, format_csv},
    poi::CriticalPoint,
};

pub(crate) const ROW_H: f32 = 22.0;
pub(crate) const HDR_H: f32 = 26.0;
pub(crate) const MAX_H: f32 = 280.0;
pub(crate) const IDX_W: f32 = 56.0;
pub(crate) const COL_W: f32 = 136.0;

/// Active tab inside [`show_data_table_modal_with_pois`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DataTableViewMode {
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

/// Drain any completed asynchronous clipboard status for `plot_id` into `state`.
pub fn poll_data_table_clipboard_status(ctx: &Context, plot_id: Id, state: &mut DataTableState) {
    let id = data_table_clipboard_status_id(plot_id);
    if let Some(status) = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(id)) {
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
            st.status_message = ctx.data_mut(|d| d.remove_temp(id)).or(Some(fallback));
        }
        Err(e) => {
            let _ = ctx.data_mut(|d| d.remove_temp::<(String, bool)>(id));
            st.status_message = Some((format!("Clipboard copy failed: {e}"), true));
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
        if self.state.selected_row != prev {
            let idx = if self.shared_x {
                0
            } else {
                self.state.selected_series
            };
            if let Some(pt) = self
                .series
                .get(idx)
                .and_then(|s| s.points.get(self.state.selected_row))
            {
                let (x, y) = (format_table_value(pt[0]), format_table_value(pt[1]));
                let msg = format!(
                    "Row {} of {rows}: X = {x}, Y = {y}",
                    self.state.selected_row + 1
                );
                emit_plot_live_message(ctx, msg, false);
            }
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
        active: &[TableSeriesRef<'_>],
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
        let sel = row_idx == self.state.selected_row;
        let fg = if sel {
            ui.visuals().selection.stroke.color
        } else {
            ui.visuals().text_color()
        };
        let _ = cells.paint(
            0,
            &(row_idx + 1).to_string(),
            ui.visuals().weak_text_color(),
            None,
        );
        let _ = cells.paint(
            1,
            &format_table_value(active[0].points[row_idx][0]),
            fg,
            None,
        );
        for (i, s) in active.iter().enumerate() {
            let _ = cells.paint(i + 2, &format_table_value(s.points[row_idx][1]), fg, None);
        }
    }

    fn render_grid(
        &mut self,
        ui: &mut Ui,
        active: &[TableSeriesRef<'_>],
        cols: &[(String, Color32)],
    ) {
        let rows = active[0].points.len();
        let width = IDX_W + COL_W * (cols.len() as f32 + 1.0);
        ScrollArea::horizontal()
            .id_salt(self.plot_id.with("data_table_hscroll"))
            .max_width(ui.available_width().max(320.0))
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.set_min_width(width);
                render_header_row(ui, cols, width);
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
                let tsv = format_tsv(self.series);
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
                    format_csv(&ds),
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
        if self.series.len() > 1 && !self.shared_x {
            ScrollArea::horizontal()
                .id_salt(self.plot_id.with("data_table_series_tabs"))
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

fn render_header_row(ui: &mut Ui, cols: &[(String, Color32)], width: f32) {
    let (rect, resp) = ui.allocate_exact_size(vec2(width, HDR_H), Sense::hover());
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
    let _ = cells.paint(1, "X", fg, None);
    for (i, (name, c)) in cols.iter().enumerate() {
        let cell = cells.paint(i + 2, name, fg, Some(*c));
        if resp.hover_pos().is_some_and(|p| cell.contains(p)) {
            resp.clone().on_hover_text(name);
        }
    }
}

/// Render the closeable, virtualized Data Table modal for a plot when `*open` is `true`.
pub fn show_data_table_modal(
    ctx: &Context,
    plot_id: Id,
    window_title: &str,
    open: &mut bool,
    state: &mut DataTableState,
    datasets: &[(String, Vec<[f64; 2]>, Color32)],
) {
    show_data_table_modal_with_pois(ctx, plot_id, window_title, open, state, datasets, &[]);
}

/// Render the Data Table modal with both `Series Data` and `Critical Points` views.
pub fn show_data_table_modal_with_pois(
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
    let series: Vec<_> = datasets
        .iter()
        .filter(|(_, p, _)| !p.is_empty())
        .map(|(name, points, color)| TableSeriesRef {
            name,
            points,
            color: *color,
        })
        .collect();
    let shared_x = series.len() > 1 && series_share_x(&series);
    preprocess_data_table_frame(ctx, plot_id, state, &series, shared_x, critical_points);
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
                        series: &series,
                        shared_x,
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

fn preprocess_data_table_frame(
    ctx: &Context,
    plot_id: Id,
    state: &mut DataTableState,
    series: &[TableSeriesRef<'_>],
    shared_x: bool,
    critical_points: &[CriticalPoint],
) {
    if series.is_empty() && !critical_points.is_empty() {
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
        let series_lbl = format!("Series Data ({series_count})");
        if ui
            .selectable_label(state.view_mode == DataTableViewMode::SeriesData, series_lbl)
            .clicked()
            && state.view_mode != DataTableViewMode::SeriesData
        {
            state.view_mode = DataTableViewMode::SeriesData;
            state.scroll_to_selected = true;
        }
        let poi_lbl = format!("Critical Points ({poi_count})");
        if ui
            .selectable_label(
                state.view_mode == DataTableViewMode::CriticalPoints,
                poi_lbl,
            )
            .clicked()
            && state.view_mode != DataTableViewMode::CriticalPoints
        {
            state.view_mode = DataTableViewMode::CriticalPoints;
            state.scroll_to_selected = true;
        }
    });
}

#[cfg(test)]
#[path = "data_table_tests.rs"]
mod tests;
