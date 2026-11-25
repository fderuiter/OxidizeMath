//! Numeric formatting, shared-`X` detection, and TSV serialization for [`crate::data_table`].

use std::fmt::Write as _;

use egui::Color32;

/// Borrowed view of a non-empty plottable series.
#[derive(Clone, Copy)]
pub struct TableSeriesRef<'a> {
    pub name: &'a str,
    pub points: &'a [[f64; 2]],
    pub color: Color32,
}

impl TableSeriesRef<'_> {
    pub(crate) fn display_name(&self, idx: usize) -> String {
        let trimmed = self.name.trim();
        if trimmed.is_empty() {
            format!("Series {}", idx + 1)
        } else {
            trimmed.to_owned()
        }
    }
}

/// Format a numeric coordinate for tabular display with scientific precision.
#[must_use]
pub fn format_table_value(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    if value == 0.0 {
        return "0.000000".to_owned();
    }
    if !(1e-3..1e6).contains(&value.abs()) {
        format!("{value:.6e}")
    } else {
        format!("{value:.6}")
    }
}

fn same_x_coord(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    let tol = 1e-12 * a.abs().max(b.abs()).max(1.0);
    a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= tol)
}

/// Returns `true` when all provided series are non-empty, equal-length, and share the same `X` grid.
#[must_use]
pub fn series_share_x(series: &[TableSeriesRef<'_>]) -> bool {
    let Some(first) = series.first().filter(|s| !s.points.is_empty()) else {
        return false;
    };
    series.iter().skip(1).all(|other| {
        other.points.len() == first.points.len()
            && first
                .points
                .iter()
                .zip(other.points)
                .all(|(a, b)| same_x_coord(a[0], b[0]))
    })
}

pub(crate) fn sanitize_tsv_label(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|ch| {
            if matches!(ch, '\t' | '\n' | '\r') {
                ' '
            } else {
                ch
            }
        })
        .collect();
    if cleaned.trim_start().starts_with(['=', '+', '-', '@']) {
        format!("'{cleaned}")
    } else {
        cleaned
    }
}

pub(crate) fn write_aligned_tsv(
    series: &[TableSeriesRef<'_>],
    rows: std::ops::Range<usize>,
) -> String {
    let mut out = String::from("X");
    for (idx, s) in series.iter().enumerate() {
        let _ = write!(out, "\t{}", sanitize_tsv_label(&s.display_name(idx)));
    }
    out.push('\n');
    for r in rows {
        let _ = write!(out, "{}", series[0].points[r][0]);
        for s in series {
            let _ = write!(out, "\t{}", s.points[r][1]);
        }
        out.push('\n');
    }
    out
}

/// Serialize non-empty series to full-precision tab-separated values (TSV).
#[must_use]
pub fn format_tsv(series: &[TableSeriesRef<'_>]) -> String {
    if series.is_empty() {
        return String::from("X\tY\n");
    }
    if series.len() == 1 || series_share_x(series) {
        write_aligned_tsv(series, 0..series[0].points.len())
    } else {
        let mut out = String::from("Dataset\tX\tY\n");
        for (idx, s) in series.iter().enumerate() {
            let name = sanitize_tsv_label(&s.display_name(idx));
            for pt in s.points {
                let _ = writeln!(out, "{name}\t{}\t{}", pt[0], pt[1]);
            }
        }
        out
    }
}

/// Borrowed view of a non-empty plottable 3D world-coordinate series.
#[derive(Clone, Copy)]
pub struct TableSeriesRef3D<'a> {
    pub name: &'a str,
    pub points: &'a [[f64; 3]],
    pub color: Color32,
}

impl TableSeriesRef3D<'_> {
    pub(crate) fn display_name(&self, idx: usize) -> String {
        let trimmed = self.name.trim();
        if trimmed.is_empty() {
            format!("Series {}", idx + 1)
        } else {
            trimmed.to_owned()
        }
    }
}

/// Serialize non-empty 3D world-coordinate series to full-precision tab-separated values (TSV).
#[must_use]
pub fn format_tsv_3d(series: &[TableSeriesRef3D<'_>]) -> String {
    if series.is_empty() {
        return String::from("X\tY\tZ\n");
    }
    if series.len() == 1 {
        let mut out = String::from("X\tY\tZ\n");
        for pt in series[0].points {
            let _ = writeln!(out, "{}\t{}\t{}", pt[0], pt[1], pt[2]);
        }
        out
    } else {
        let mut out = String::from("Dataset\tX\tY\tZ\n");
        for (idx, s) in series.iter().enumerate() {
            let name = sanitize_tsv_label(&s.display_name(idx));
            for pt in s.points {
                let _ = writeln!(out, "{name}\t{}\t{}\t{}", pt[0], pt[1], pt[2]);
            }
        }
        out
    }
}

/// Serialize critical points to full-precision tab-separated values (TSV).
#[must_use]
pub fn format_pois_tsv(pois: &[crate::poi::CriticalPoint]) -> String {
    let mut out = String::from("Category\tSeries\tX\tY\n");
    for poi in pois {
        let cat = sanitize_tsv_label(poi.kind.label());
        let ser = sanitize_tsv_label(&poi.series_summary());
        let _ = writeln!(out, "{cat}\t{ser}\t{}\t{}", poi.point[0], poi.point[1]);
    }
    out
}

fn quote_csv_cell(raw: &str) -> String {
    let guarded = if raw.trim_start().starts_with(['=', '+', '-', '@']) {
        format!("'{raw}")
    } else {
        raw.to_owned()
    };
    if guarded.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", guarded.replace('"', "\"\""))
    } else {
        guarded
    }
}

/// Serialize critical points to RFC-4180 comma-separated values (CSV).
#[must_use]
pub fn format_pois_csv(pois: &[crate::poi::CriticalPoint]) -> String {
    let mut out = String::from("Category,Series,X,Y\n");
    for poi in pois {
        let cat = quote_csv_cell(poi.kind.label());
        let ser = quote_csv_cell(&poi.series_summary());
        let _ = writeln!(out, "{cat},{ser},{},{}", poi.point[0], poi.point[1]);
    }
    out
}

fn step_series_index(active_idx: usize, count: usize, forward: bool) -> usize {
    if forward {
        (active_idx + 1) % count
    } else if active_idx == 0 {
        count.saturating_sub(1)
    } else {
        active_idx - 1
    }
}

fn render_virtualized_series_combo(
    ui: &mut egui::Ui,
    combo_id: egui::Id,
    count: usize,
    active_idx: usize,
    active_label: &str,
    format_item: &mut impl FnMut(usize) -> String,
) -> Option<usize> {
    let mut chosen = None;
    let combo_resp = egui::ComboBox::from_id_salt(combo_id)
        .selected_text(active_label)
        .width(240.0)
        .show_ui(ui, |ui| {
            let row_h = ui.spacing().interact_size.y.max(18.0);
            egui::ScrollArea::vertical().max_height(240.0).show_rows(
                ui,
                row_h,
                count,
                |ui, range| {
                    for idx in range {
                        if ui
                            .selectable_label(active_idx == idx, format_item(idx))
                            .clicked()
                        {
                            chosen = Some(idx);
                        }
                    }
                },
            );
        });
    let btn_id = combo_resp.response.id;
    ui.ctx().accesskit_node_builder(btn_id, |builder| {
        builder.set_label(format!("Select plot series, current {active_label}"));
    });
    if combo_resp.response.has_focus() && combo_resp.inner.is_none() {
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                chosen = Some(step_series_index(active_idx, count, false));
            } else if i.key_pressed(egui::Key::ArrowRight) {
                chosen = Some(step_series_index(active_idx, count, true));
            }
        });
    }
    chosen
}

/// Renders a compact, virtualized, keyboard-accessible [`egui::ComboBox`] series selector bar.
pub(crate) fn render_series_selector_bar(
    ui: &mut egui::Ui,
    combo_id: egui::Id,
    count: usize,
    active_idx: usize,
    mut format_item: impl FnMut(usize) -> String,
) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let active_idx = active_idx.min(count - 1);
    let active_label = format!("{}/{count} · {}", active_idx + 1, format_item(active_idx));
    let mut chosen = None;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Series ({count}):"))
                .small()
                .weak(),
        );
        if crate::a11y::focusable_action_button(
            ui,
            count > 1,
            egui::Button::new("◀").small(),
            Some("Previous series"),
        )
        .clicked()
        {
            chosen = Some(step_series_index(active_idx, count, false));
        }
        if let Some(idx) = render_virtualized_series_combo(
            ui,
            combo_id,
            count,
            active_idx,
            &active_label,
            &mut format_item,
        ) {
            chosen = Some(idx);
        }
        if crate::a11y::focusable_action_button(
            ui,
            count > 1,
            egui::Button::new("▶").small(),
            Some("Next series"),
        )
        .clicked()
        {
            chosen = Some(step_series_index(active_idx, count, true));
        }
    });
    chosen.filter(|&idx| idx != active_idx && idx < count)
}
