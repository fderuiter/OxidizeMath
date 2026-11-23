//! CSV overlay rendering, `Overlays` toolbar/menu controls, `Zoom Fit` framing,
//! narrow-pane toolbar overflow compaction, clipboard paste, and drag-and-drop handling.

use std::sync::Arc;

use egui::{Color32, Id, Rect, Response, Ui, Vec2b};

use crate::{
    AccessState, Line, MarkerShape, PlotBounds, PlotItem, PlotMemory, PlotUi, Points,
    a11y::{self, AccessibleTooltipExt as _},
    apply_aspect_and_sync_linked,
    commands::CommandRegistryData,
    csv_import::{MAX_CSV_IMPORT_BYTES, MAX_CSV_IMPORT_SERIES},
    emit_plot_live_message,
    export::{self, ImportedDataset},
    file_io::parse_immediate_csv_overlay,
    fit_overlay,
};

/// Plot width below which secondary toolbar controls collapse into a `⋯ More` overflow menu.
pub(crate) const NARROW_PLOT_TOOLBAR_WIDTH: f32 = 520.0;

pub(crate) struct OverlayViewportSync<'a> {
    pub data_aspect: Option<f32>,
    pub linked_axes: &'a Option<(Id, Vec2b)>,
    pub default_auto_bounds: Vec2b,
}

pub(crate) struct PlotToolbarConfig<'a, 'b> {
    pub plot_id: Id,
    pub header_rect: Rect,
    pub allow_csv_import: bool,
    pub allow_2d_analysis: bool,
    pub scene_3d_mode: bool,
    pub has_table_content: bool,
    pub has_fit_candidates: bool,
    pub base_items_hash: u64,
    pub items: &'a [Box<dyn PlotItem + 'b>],
    pub export_datasets_3d: &'a [(String, Vec<[f64; 3]>)],
    pub fit_status: &'a fit_overlay::FitOverlayStatus,
    pub viewport_sync: OverlayViewportSync<'a>,
}

/// Merges newly imported CSV/TSV datasets into `existing`, preserving per-series style toggles.
pub(crate) fn merge_imported_datasets(
    existing: &mut Vec<ImportedDataset>,
    incoming: Vec<ImportedDataset>,
) {
    for new_ds in incoming {
        if let Some(slot) = existing.iter_mut().find(|d| d.name == new_ds.name) {
            slot.points = new_ds.points;
            slot.visible = true;
            if !slot.show_points && !slot.show_lines {
                slot.show_points = true;
            }
        } else if existing.len() < MAX_CSV_IMPORT_SERIES {
            existing.push(new_ds);
        }
    }
}

/// Renders active [`ImportedDataset`] overlays into `plot_ui` according to `show_points`/`show_lines`.
pub(crate) fn render_imported_overlays(plot_ui: &mut PlotUi<'_>, datasets: &[ImportedDataset]) {
    for dataset in datasets {
        if !dataset.visible
            || dataset.points.is_empty()
            || (!dataset.show_points && !dataset.show_lines)
        {
            continue;
        }
        let color = plot_ui.auto_color();
        if dataset.show_lines {
            plot_ui.line(
                Line::new(&dataset.name, dataset.points.clone())
                    .color(color)
                    .companion_visual_only(dataset.show_points),
            );
        }
        if dataset.show_points {
            plot_ui.points(
                Points::new(&dataset.name, dataset.points.clone())
                    .color(color)
                    .radius(3.0_f32)
                    .shape(MarkerShape::Circle),
            );
        }
    }
}

fn pad_axis_range(min_v: f64, max_v: f64, is_log: bool) -> (f64, f64) {
    if (max_v - min_v).abs() < f64::EPSILON {
        if is_log {
            ((min_v * 0.85).max(f64::MIN_POSITIVE), max_v * 1.15)
        } else {
            let span = (min_v.abs() * 0.1).max(1.0);
            (min_v - span, max_v + span)
        }
    } else if is_log {
        ((min_v * 0.92).max(f64::MIN_POSITIVE), max_v * 1.08)
    } else {
        let pad = (max_v - min_v) * 0.05;
        (min_v - pad, max_v + pad)
    }
}

/// Computes padded finite `[x, y]` bounds framing `datasets`, respecting logarithmic axes.
pub(crate) fn compute_overlay_fit_bounds(
    datasets: &[&ImportedDataset],
    log_x: bool,
    log_y: bool,
) -> Option<PlotBounds> {
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut found = false;

    for ds in datasets {
        for &[x, y] in &ds.points {
            let valid_x = x.is_finite() && (!log_x || x > 0.0);
            let valid_y = y.is_finite() && (!log_y || y > 0.0);
            if valid_x && valid_y {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }
    if !found {
        return None;
    }
    let (x0, x1) = pad_axis_range(min_x, max_x, log_x);
    let (y0, y1) = pad_axis_range(min_y, max_y, log_y);
    Some(PlotBounds::from_min_max([x0, y0], [x1, y1]))
}

/// Snaps `mem.transform` bounds to `bounds` even when `mem.auto_bounds` is `false`.
pub(crate) fn apply_computed_fit_bounds(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    bounds: PlotBounds,
    target_label: &str,
    sync: &OverlayViewportSync<'_>,
) {
    mem.auto_bounds = false.into();
    mem.transform.set_bounds(bounds);
    apply_aspect_and_sync_linked(
        ui,
        mem,
        sync.data_aspect,
        sync.linked_axes,
        sync.default_auto_bounds,
    );
    emit_plot_live_message(
        ui.ctx(),
        format!("Zoomed plot bounds to fit {target_label}"),
        false,
    );
}

fn is_supported_csv_extension(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    lower.ends_with(".csv") || lower.ends_with(".tsv") || lower.ends_with(".txt")
}

fn dropped_file_display_name(dropped: &egui::DroppedFile) -> Option<String> {
    if !dropped.name.trim().is_empty() && is_supported_csv_extension(&dropped.name) {
        return Some(export::sanitize_download_filename(
            &dropped.name,
            "dropped.csv",
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ref path) = dropped.path {
        let s = path.to_string_lossy();
        if is_supported_csv_extension(&s) {
            return Some(export::sanitize_download_filename(&s, "dropped.csv"));
        }
    }
    None
}

fn read_dropped_csv_payload(
    dropped: &egui::DroppedFile,
    name: &str,
) -> Result<(String, String), String> {
    if let Some(ref bytes) = dropped.bytes {
        if bytes.len() > MAX_CSV_IMPORT_BYTES {
            return Err(format!(
                "Dropped file {name} ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
                bytes.len() as f64 / (1024.0 * 1024.0)
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| format!("Dropped file {name} is not valid UTF-8 text."))?;
        return Ok((name.to_owned(), text.to_owned()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ref path) = dropped.path {
        return crate::file_io::read_bounded_csv_path(path, name);
    }
    Err(format!("Dropped file {name} contained no readable bytes."))
}

/// Registers CSV shortcuts and handles `Ctrl+Shift+I`, focused `Event::Paste`, and hovered drops.
pub(crate) fn handle_plot_csv_inputs(
    ui: &mut Ui,
    response: &Response,
    plot_rect: Rect,
    plot_id: Id,
    allow_csv_import: bool,
    mem: &mut PlotMemory,
    access_state: &mut AccessState,
) {
    if !allow_csv_import || !ui.is_enabled() {
        return;
    }
    CommandRegistryData::register_csv_import_commands(ui.ctx());

    let modal_open = access_state.show_table
        || access_state.export_modal.open
        || access_state.show_settings_modal
        || access_state.show_add_callout_modal
        || access_state.show_edit_callout_modal;
    if modal_open {
        return;
    }

    let can_trigger_shortcut = response.has_focus()
        || (!ui.ctx().wants_keyboard_input()
            && (response.hovered() || ui.memory(|m| m.focused().is_none())));
    if can_trigger_shortcut && CommandRegistryData::check_import_csv_shortcut(ui.ctx(), ui) {
        export::trigger_import_csv_for_plot(ui.ctx(), plot_id);
    }

    if response.has_focus() {
        handle_focused_plot_paste(ui, mem, access_state);
    }
    let pointer_inside = ui.input(|i| {
        i.pointer
            .latest_pos()
            .is_some_and(|p| plot_rect.contains(p))
    });
    if response.hovered() || response.contains_pointer() || pointer_inside {
        handle_hovered_plot_drops(ui, mem, access_state);
    }
}

fn handle_focused_plot_paste(ui: &mut Ui, mem: &mut PlotMemory, access_state: &mut AccessState) {
    let pasted: Vec<String> = ui.ctx().input_mut(|i| {
        let mut out = Vec::new();
        i.events.retain(|evt| {
            if let egui::Event::Paste(text) = evt {
                out.push(text.clone());
                false
            } else {
                true
            }
        });
        out
    });
    for text in pasted {
        match parse_immediate_csv_overlay(ui.ctx(), "clipboard", &text) {
            Ok((datasets, msg)) => {
                merge_imported_datasets(&mut mem.imported_datasets, datasets);
                access_state.import_toast = Some((msg, false));
            }
            Err(err) => {
                access_state.import_toast = Some((err, true));
            }
        }
    }
}

fn handle_hovered_plot_drops(ui: &mut Ui, mem: &mut PlotMemory, access_state: &mut AccessState) {
    let matched_drops: Vec<(egui::DroppedFile, String)> = ui.ctx().input_mut(|i| {
        let mut matched = Vec::new();
        i.raw.dropped_files.retain(|f| {
            if let Some(name) = dropped_file_display_name(f) {
                matched.push((f.clone(), name));
                false
            } else {
                true
            }
        });
        matched
    });
    for (dropped, name) in matched_drops {
        match read_dropped_csv_payload(&dropped, &name) {
            Ok((src, content)) => match parse_immediate_csv_overlay(ui.ctx(), &src, &content) {
                Ok((datasets, msg)) => {
                    merge_imported_datasets(&mut mem.imported_datasets, datasets);
                    access_state.import_toast = Some((msg, false));
                }
                Err(err) => {
                    access_state.import_toast = Some((err, true));
                }
            },
            Err(err) => {
                emit_plot_live_message(ui.ctx(), err.clone(), true);
                access_state.import_toast = Some((err, true));
                ui.ctx().request_repaint();
            }
        }
    }
}

fn render_overlays_menu_header(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    access_state: &mut AccessState,
    sync: &OverlayViewportSync<'_>,
    close_on_action: bool,
) {
    ui.horizontal(|ui| {
        if ui.button("🔍 Zoom Fit All").clicked() {
            let bounds = {
                let active: Vec<&ImportedDataset> = mem
                    .imported_datasets
                    .iter()
                    .filter(|d| d.visible && (d.show_points || d.show_lines))
                    .collect();
                let refs: Vec<&ImportedDataset> = if active.is_empty() {
                    mem.imported_datasets.iter().collect()
                } else {
                    active
                };
                compute_overlay_fit_bounds(&refs, mem.log_x, mem.log_y)
            };
            if let Some(bounds) = bounds {
                apply_computed_fit_bounds(ui, mem, bounds, "all overlays", sync);
            }
            if close_on_action {
                ui.close();
            }
        }
        if ui.button("🗑 Delete All").clicked() {
            mem.imported_datasets.clear();
            access_state.fit_model = None;
            emit_plot_live_message(ui.ctx(), "Deleted all imported overlays".to_owned(), false);
            ui.ctx().request_repaint();
            ui.close();
        }
    });
}

fn render_overlay_series_row(ui: &mut Ui, ds: &mut ImportedDataset) -> (bool, bool) {
    let mut zoom_clicked = false;
    let mut delete_clicked = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{} ({})", ds.name, ds.points.len())).strong());
        if ui.checkbox(&mut ds.show_points, "Points").changed() {
            let state_word = if ds.show_points {
                "Enabled"
            } else {
                "Disabled"
            };
            emit_plot_live_message(
                ui.ctx(),
                format!("{state_word} points for {}", ds.name),
                false,
            );
        }
        if ui.checkbox(&mut ds.show_lines, "Lines").changed() {
            let state_word = if ds.show_lines { "Enabled" } else { "Disabled" };
            emit_plot_live_message(
                ui.ctx(),
                format!("{state_word} lines for {}", ds.name),
                false,
            );
        }
        zoom_clicked = ui.small_button("Zoom Fit").clicked();
        delete_clicked = ui.small_button("Delete").clicked();
    });
    (zoom_clicked, delete_clicked)
}

/// Renders the `Overlays` menu contents (`Points`, `Lines`, `Zoom Fit`, single/all delete).
pub(crate) fn render_overlays_menu_contents(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    access_state: &mut AccessState,
    sync: &OverlayViewportSync<'_>,
    close_on_action: bool,
) {
    render_overlays_menu_header(ui, mem, access_state, sync, close_on_action);
    ui.separator();

    let mut delete_idx = None;
    let mut zoom_fit_idx = None;
    for (idx, ds) in mem.imported_datasets.iter_mut().enumerate() {
        let (zoom_clicked, del_clicked) = render_overlay_series_row(ui, ds);
        if zoom_clicked {
            zoom_fit_idx = Some(idx);
        }
        if del_clicked {
            delete_idx = Some(idx);
        }
    }
    if let Some(idx) = zoom_fit_idx
        && let Some((bounds, name)) = mem.imported_datasets.get(idx).and_then(|ds| {
            compute_overlay_fit_bounds(&[ds], mem.log_x, mem.log_y).map(|b| (b, ds.name.clone()))
        })
    {
        apply_computed_fit_bounds(ui, mem, bounds, &name, sync);
        if close_on_action {
            ui.close();
        }
    }
    if let Some(idx) = delete_idx {
        let removed = mem.imported_datasets.remove(idx);
        access_state.fit_model = None;
        emit_plot_live_message(
            ui.ctx(),
            format!("Deleted imported dataset {}", removed.name),
            false,
        );
        ui.ctx().request_repaint();
    }
}

fn render_secondary_toolbar_actions(
    ui: &mut Ui,
    cfg: &PlotToolbarConfig<'_, '_>,
    access_state: &mut AccessState,
    in_overflow: bool,
) {
    if ui.button("Save PNG").clicked() {
        access_state.export_modal.open = true;
        if in_overflow {
            ui.close();
        }
    }
    if ui.button("Export CSV").clicked() {
        if !cfg.export_datasets_3d.is_empty() {
            export::export_csv_3d(cfg.export_datasets_3d);
        } else {
            export::export_csv(&crate::collect_export_datasets(cfg.items));
        }
        if in_overflow {
            ui.close();
        }
    }
    if ui.button("Export JSON").clicked() {
        if !cfg.export_datasets_3d.is_empty() {
            export::export_json_3d(cfg.export_datasets_3d);
        } else {
            export::export_json(&crate::collect_export_datasets(cfg.items));
        }
        if in_overflow {
            ui.close();
        }
    }
    if !cfg.scene_3d_mode && ui.button("🔧 Settings").clicked() {
        access_state.show_settings_modal = !access_state.show_settings_modal;
        if in_overflow {
            ui.close();
        }
    }
    render_toolbar_fit_controls(ui, cfg, access_state, in_overflow);
}

fn render_toolbar_fit_controls(
    ui: &mut Ui,
    cfg: &PlotToolbarConfig<'_, '_>,
    access_state: &mut AccessState,
    in_overflow: bool,
) {
    if !cfg.allow_2d_analysis {
        return;
    }
    ui.separator();
    if !cfg.has_fit_candidates {
        let _ = a11y::focusable_action_button(
            ui,
            false,
            egui::Button::new("Fit Curve"),
            Some("Fit Curve"),
        )
        .accessible_tooltip("Requires a dataset with at least 2 points for curve fitting");
        if access_state.fit_model.is_some() && ui.button("Clear Fit").clicked() {
            clear_active_fit_state(access_state);
        }
        return;
    }
    ui.menu_button("Fit Curve", |ui| {
        let base_ds = Arc::clone(crate::get_or_populate_base_datasets(
            cfg.items,
            cfg.base_items_hash,
            &mut access_state.cached_base_datasets,
        ));
        fit_overlay::render_fit_curve_menu(ui, &base_ds, access_state, in_overflow, cfg.fit_status);
    });
    if cfg.fit_status.can_toggle_visibility() {
        let toggle_text = if access_state.show_fit_overlay {
            "Hide Fit"
        } else {
            "Show Fit"
        };
        if ui.button(toggle_text).clicked() {
            access_state.show_fit_overlay = !access_state.show_fit_overlay;
        }
    }
    if access_state.fit_model.is_some() && ui.button("Clear Fit").clicked() {
        clear_active_fit_state(access_state);
    }
}

fn clear_active_fit_state(access_state: &mut AccessState) {
    access_state.fit_model = None;
    access_state.cached_fit = None;
    access_state.custom_fit_error = None;
    access_state.last_fit_error_announced = None;
}

fn render_import_toast(ui: &mut Ui, access_state: &mut AccessState) {
    let Some((ref msg, is_error)) = access_state.import_toast else {
        return;
    };
    ui.separator();
    let dark_mode = ui.visuals().dark_mode;
    let (color, icon) = match (is_error, dark_mode) {
        (true, true) => (Color32::from_rgb(255, 118, 118), "⚠️"),
        (true, false) => (Color32::from_rgb(180, 28, 28), "⚠️"),
        (false, true) => (Color32::from_rgb(74, 214, 122), "ℹ️"),
        (false, false) => (Color32::from_rgb(15, 115, 54), "ℹ️"),
    };
    ui.colored_label(color, format!("{icon} {msg}"));
    if a11y::focusable_action_button(
        ui,
        true,
        egui::Button::new("✖").min_size(egui::vec2(24.0, 24.0)),
        Some("Dismiss import notification"),
    )
    .accessible_tooltip("Dismiss import notification")
    .clicked()
    {
        access_state.import_toast = None;
    }
}

fn render_csv_toolbar_controls(
    ui: &mut Ui,
    cfg: &PlotToolbarConfig<'_, '_>,
    mem: &mut PlotMemory,
    access_state: &mut AccessState,
) {
    if !cfg.allow_csv_import {
        return;
    }
    if a11y::focusable_action_button(
        ui,
        true,
        egui::Button::new("Import CSV"),
        Some("Import CSV"),
    )
    .accessible_tooltip("Import CSV or TSV dataset (Ctrl+Shift+I)")
    .clicked()
    {
        export::trigger_import_csv_for_plot(ui.ctx(), cfg.plot_id);
    }
    if !mem.imported_datasets.is_empty() {
        let lbl = format!("Overlays ({})", mem.imported_datasets.len());
        let menu_cfg = egui::containers::menu::MenuConfig::new()
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
        egui::containers::menu::MenuButton::new(lbl)
            .config(menu_cfg)
            .ui(ui, |ui| {
                render_overlays_menu_contents(ui, mem, access_state, &cfg.viewport_sync, false);
            });
    }
}

fn render_table_view_button(ui: &mut Ui, has_table_content: bool) -> bool {
    let mut table_btn = a11y::focusable_action_button(
        ui,
        has_table_content,
        egui::Button::new("Table View"),
        Some("Table View"),
    );
    if !has_table_content {
        table_btn = table_btn.accessible_tooltip("No tabular data points available for this plot");
    }
    has_table_content && table_btn.clicked()
}

fn render_collapse_toolbar_button(ui: &mut Ui, mem: &mut PlotMemory) {
    let collapse_btn = egui::Button::new("▲").min_size(egui::vec2(24.0, 24.0));
    if a11y::focusable_action_button(
        ui,
        ui.is_enabled(),
        collapse_btn,
        Some("Collapse plot toolbar"),
    )
    .accessible_tooltip("Collapse toolbar (restore via right-click menu or Plot Settings)")
    .clicked()
    {
        mem.toolbar_collapsed = true;
        ui.ctx().request_repaint();
    }
}

/// Renders the docked horizontal plot header, compacting secondary actions on narrow split panes.
pub(crate) fn render_plot_top_toolbar(
    ui: &mut Ui,
    cfg: &PlotToolbarConfig<'_, '_>,
    mem: &mut PlotMemory,
    access_state: &mut AccessState,
) {
    let inner_rect = cfg.header_rect.shrink2(egui::vec2(2.0, 2.0));
    let mut header_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("plot_docked_header", cfg.plot_id))
            .max_rect(inner_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    header_ui.set_clip_rect(cfg.header_rect.intersect(ui.clip_rect()));
    header_ui.spacing_mut().interact_size.x = header_ui.spacing().interact_size.x.max(24.0);
    header_ui.spacing_mut().interact_size.y = header_ui.spacing().interact_size.y.max(24.0);
    header_ui.spacing_mut().button_padding = egui::vec2(5.0, 3.0);
    header_ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);

    render_csv_toolbar_controls(&mut header_ui, cfg, mem, access_state);

    let show_inline_table = cfg.header_rect.width() >= 340.0;
    if show_inline_table && render_table_view_button(&mut header_ui, cfg.has_table_content) {
        access_state.show_table = !access_state.show_table;
    }

    if cfg.header_rect.width() >= NARROW_PLOT_TOOLBAR_WIDTH {
        render_secondary_toolbar_actions(&mut header_ui, cfg, access_state, false);
        render_import_toast(&mut header_ui, access_state);
    } else {
        header_ui.menu_button("⋯ More", |ui| {
            if !show_inline_table && render_table_view_button(ui, cfg.has_table_content) {
                access_state.show_table = !access_state.show_table;
                ui.close();
            }
            render_secondary_toolbar_actions(ui, cfg, access_state, true);
        });
    }

    render_collapse_toolbar_button(&mut header_ui, mem);
}

#[cfg(test)]
#[path = "plot_overlays_tests.rs"]
mod tests;
