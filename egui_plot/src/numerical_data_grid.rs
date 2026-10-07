use egui::{Align, Color32, Layout, RichText, Sense, Ui, Vec2};
use std::cmp::Ordering;

/// Represents a single cell edit event.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEdit {
    /// The row index of the edited cell.
    pub row: usize,
    /// The column index of the edited cell.
    pub col: usize,
    /// The new floating-point value entered by the user.
    pub new_value: f64,
}

/// Abstract data source trait for tabular numerical data grid rendering.
pub trait GridDataSource {
    /// Total number of rows in the grid.
    fn num_rows(&self) -> usize;
    /// Total number of data columns in the grid.
    fn num_cols(&self) -> usize;
    /// Header title for column `col`.
    fn header(&self, col: usize) -> String;
    /// Optional row label title for row `row`.
    fn row_label(&self, _row: usize) -> Option<String> {
        None
    }
    /// Retrieve floating point numerical value at `(row, col)`.
    fn cell_value(&self, row: usize, col: usize) -> f64;
    /// Update cell value at `(row, col)`.
    fn set_cell_value(&mut self, _row: usize, _col: usize, _val: f64) {}
    /// Whether cell at `(row, col)` is user-editable.
    fn is_editable(&self, _row: usize, _col: usize) -> bool {
        true
    }
}

/// A simple matrix dataset implementing `GridDataSource`.
#[derive(Clone, Debug, Default)]
pub struct SimpleMatrixGrid {
    pub headers: Vec<String>,
    pub row_labels: Option<Vec<String>>,
    pub data: Vec<Vec<f64>>,
}

impl SimpleMatrixGrid {
    pub fn new(headers: Vec<String>, data: Vec<Vec<f64>>) -> Self {
        Self {
            headers,
            row_labels: None,
            data,
        }
    }

    pub fn with_row_labels(mut self, row_labels: Vec<String>) -> Self {
        self.row_labels = Some(row_labels);
        self
    }
}

impl GridDataSource for SimpleMatrixGrid {
    fn num_rows(&self) -> usize {
        self.data.len()
    }

    fn num_cols(&self) -> usize {
        self.headers.len()
    }

    fn header(&self, col: usize) -> String {
        self.headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("Col {}", col))
    }

    fn row_label(&self, row: usize) -> Option<String> {
        self.row_labels
            .as_ref()
            .and_then(|lbls| lbls.get(row).cloned())
    }

    fn cell_value(&self, row: usize, col: usize) -> f64 {
        self.data
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(0.0)
    }

    fn set_cell_value(&mut self, row: usize, col: usize, val: f64) {
        if row < self.data.len() {
            if col < self.data[row].len() {
                self.data[row][col] = val;
            } else if col == self.data[row].len() {
                self.data[row].push(val);
            }
        }
    }
}

/// A synchronized multi-series time-series dataset alignment structure.
#[derive(Clone, Debug, Default)]
pub struct SynchronizedTimeSeriesGrid {
    pub headers: Vec<String>,
    pub data: Vec<Vec<f64>>,
}

impl SynchronizedTimeSeriesGrid {
    /// Constructs a synchronized time-series matrix by unifying time steps across datasets.
    pub fn from_datasets(datasets: &[(&str, &[[f64; 2]])]) -> Self {
        if datasets.is_empty() {
            return Self::default();
        }

        // Gather all unique X coordinates sorted
        let mut all_x: Vec<f64> = Vec::new();
        for (_, pts) in datasets {
            for p in *pts {
                all_x.push(p[0]);
            }
        }
        all_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        all_x.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let mut headers = vec!["Step".to_string(), "X / Time".to_string()];
        for (name, _) in datasets {
            headers.push((*name).to_string());
        }

        let mut data = Vec::with_capacity(all_x.len());

        // For fast lookup build map or index per dataset
        for (step, &x_val) in all_x.iter().enumerate() {
            let mut row = Vec::with_capacity(headers.len());
            row.push(step as f64);
            row.push(x_val);

            for (_, pts) in datasets {
                // Find nearest or exact point
                let y_val = pts
                    .iter()
                    .find(|p| (p[0] - x_val).abs() < 1e-7)
                    .map(|p| p[1])
                    .unwrap_or(f64::NAN);
                row.push(y_val);
            }
            data.push(row);
        }

        Self { headers, data }
    }
}

impl GridDataSource for SynchronizedTimeSeriesGrid {
    fn num_rows(&self) -> usize {
        self.data.len()
    }

    fn num_cols(&self) -> usize {
        self.headers.len()
    }

    fn header(&self, col: usize) -> String {
        self.headers.get(col).cloned().unwrap_or_default()
    }

    fn cell_value(&self, row: usize, col: usize) -> f64 {
        self.data
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(f64::NAN)
    }

    fn set_cell_value(&mut self, row: usize, col: usize, val: f64) {
        if row < self.data.len() && col < self.data[row].len() {
            self.data[row][col] = val;
        }
    }

    fn is_editable(&self, _row: usize, col: usize) -> bool {
        // Step and Time columns are read-only
        col >= 2
    }
}

/// Interactive high-performance numerical data grid widget.
#[derive(Clone, Debug)]
pub struct NumericalDataGrid {
    /// Display precision (number of decimal places).
    pub precision: usize,
    /// Row height in points for virtualized scrolling.
    pub row_height: f32,
    /// Global editable flag.
    pub editable: bool,
    /// Show copy buttons for CSV and TSV export.
    pub show_copy_buttons: bool,
    /// Show precision control widget.
    pub show_precision_control: bool,
    /// Currently edited cell: (row_index, col_index, edit_buffer_string).
    pub editing_cell: Option<(usize, usize, String)>,
    /// Active column sorting state: (col_index, ascending_bool).
    pub sort_column: Option<(usize, bool)>,
}

impl Default for NumericalDataGrid {
    fn default() -> Self {
        Self {
            precision: 4,
            row_height: 22.0,
            editable: true,
            show_copy_buttons: true,
            show_precision_control: true,
            editing_cell: None,
            sort_column: None,
        }
    }
}

impl NumericalDataGrid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    pub fn row_height(mut self, row_height: f32) -> Self {
        self.row_height = row_height;
        self
    }

    pub fn editable(mut self, editable: bool) -> Self {
        self.editable = editable;
        self
    }

    pub fn show_copy_buttons(mut self, show: bool) -> Self {
        self.show_copy_buttons = show;
        self
    }

    pub fn show_precision_control(mut self, show: bool) -> Self {
        self.show_precision_control = show;
        self
    }

    /// Formats grid content as CSV string.
    pub fn export_csv<S: GridDataSource>(&self, source: &S) -> String {
        let mut out = String::new();
        let num_cols = source.num_cols();
        let num_rows = source.num_rows();

        let mut headers = Vec::new();
        if source.row_label(0).is_some() {
            headers.push("Label".to_string());
        }
        for c in 0..num_cols {
            headers.push(source.header(c));
        }
        out.push_str(&headers.join(","));
        out.push('\n');

        for r in 0..num_rows {
            let mut row_vals = Vec::new();
            if let Some(lbl) = source.row_label(r) {
                row_vals.push(lbl);
            }
            for c in 0..num_cols {
                let v = source.cell_value(r, c);
                if v.is_nan() {
                    row_vals.push("NaN".to_string());
                } else {
                    row_vals.push(format!("{:.*}", self.precision, v));
                }
            }
            out.push_str(&row_vals.join(","));
            out.push('\n');
        }

        out
    }

    /// Formats grid content as TSV string.
    pub fn export_tsv<S: GridDataSource>(&self, source: &S) -> String {
        let mut out = String::new();
        let num_cols = source.num_cols();
        let num_rows = source.num_rows();

        let mut headers = Vec::new();
        if source.row_label(0).is_some() {
            headers.push("Label".to_string());
        }
        for c in 0..num_cols {
            headers.push(source.header(c));
        }
        out.push_str(&headers.join("\t"));
        out.push('\n');

        for r in 0..num_rows {
            let mut row_vals = Vec::new();
            if let Some(lbl) = source.row_label(r) {
                row_vals.push(lbl);
            }
            for c in 0..num_cols {
                let v = source.cell_value(r, c);
                if v.is_nan() {
                    row_vals.push("NaN".to_string());
                } else {
                    row_vals.push(format!("{:.*}", self.precision, v));
                }
            }
            out.push_str(&row_vals.join("\t"));
            out.push('\n');
        }

        out
    }

    /// Render numerical data grid into the given egui UI.
    /// Returns `Some(CellEdit)` if a cell value was updated during this frame.
    pub fn show<S: GridDataSource>(&mut self, ui: &mut Ui, source: &mut S) -> Option<CellEdit> {
        let mut cell_edit_result: Option<CellEdit> = None;

        // Header controls (Precision, Copy CSV/TSV)
        ui.horizontal(|ui| {
            if self.show_precision_control {
                ui.label("Precision:");
                if ui.button(" - ").clicked() && self.precision > 0 {
                    self.precision -= 1;
                }
                ui.label(RichText::new(format!("{}", self.precision)).strong());
                if ui.button(" + ").clicked() && self.precision < 12 {
                    self.precision += 1;
                }
                ui.separator();
            }

            if self.show_copy_buttons {
                if ui.button("📋 Copy CSV").clicked() {
                    let csv = self.export_csv(source);
                    ui.ctx().copy_text(csv);
                }
                if ui.button("📋 Copy TSV").clicked() {
                    let tsv = self.export_tsv(source);
                    ui.ctx().copy_text(tsv);
                }
                ui.separator();
            }

            ui.label(format!(
                "Rows: {} | Cols: {}",
                source.num_rows(),
                source.num_cols()
            ));
        });

        ui.add_space(4.0);

        let total_rows = source.num_rows();
        let num_cols = source.num_cols();
        let has_row_labels = total_rows > 0 && source.row_label(0).is_some();

        // Calculate sorted row indices if column sorting is active
        let mut row_indices: Vec<usize> = (0..total_rows).collect();
        if let Some((sort_col, ascending)) = self.sort_column {
            row_indices.sort_by(|&a, &b| {
                let val_a = source.cell_value(a, sort_col);
                let val_b = source.cell_value(b, sort_col);
                let cmp = val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal);
                if ascending { cmp } else { cmp.reverse() }
            });
        }

        // Determine column widths
        let min_col_width = 80.0_f32;
        let label_col_width = 110.0_f32;

        // Render Column Headers
        ui.horizontal(|ui| {
            if has_row_labels {
                ui.allocate_ui_with_layout(
                    Vec2::new(label_col_width, self.row_height),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.label(RichText::new("Row / Label").strong());
                    },
                );
            }

            for c in 0..num_cols {
                let header_text = source.header(c);
                let sort_indicator = match self.sort_column {
                    Some((sc, true)) if sc == c => " ▲",
                    Some((sc, false)) if sc == c => " ▼",
                    _ => "",
                };
                let full_title = format!("{}{}", header_text, sort_indicator);

                ui.allocate_ui_with_layout(
                    Vec2::new(min_col_width, self.row_height),
                    Layout::right_to_left(Align::Center),
                    |ui| {
                        let btn = ui.button(RichText::new(full_title).strong());
                        if btn.clicked() {
                            self.sort_column = match self.sort_column {
                                Some((sc, true)) if sc == c => Some((c, false)),
                                Some((sc, false)) if sc == c => None,
                                _ => Some((c, true)),
                            };
                        }
                    },
                );
            }
        });

        ui.separator();

        // Render Virtualized Scroll Area
        egui::ScrollArea::both().max_height(450.0).show_rows(
            ui,
            self.row_height,
            total_rows,
            |ui, row_range| {
                for vis_idx in row_range {
                    let actual_row = row_indices[vis_idx];

                    ui.horizontal(|ui| {
                        if has_row_labels {
                            ui.allocate_ui_with_layout(
                                Vec2::new(label_col_width, self.row_height),
                                Layout::left_to_right(Align::Center),
                                |ui| {
                                    let label_str =
                                        source.row_label(actual_row).unwrap_or_default();
                                    ui.label(RichText::new(label_str).color(Color32::LIGHT_GRAY));
                                },
                            );
                        }

                        for col_idx in 0..num_cols {
                            ui.allocate_ui_with_layout(
                                Vec2::new(min_col_width, self.row_height),
                                Layout::right_to_left(Align::Center),
                                |ui| {
                                    let is_editing =
                                        self.editing_cell.as_ref().is_some_and(|(r, c, _)| {
                                            *r == actual_row && *c == col_idx
                                        });

                                    if is_editing {
                                        if let Some((_, _, ref mut buf)) = self.editing_cell {
                                            let text_edit = ui.add(
                                                egui::TextEdit::singleline(buf).desired_width(70.0),
                                            );
                                            let enter_pressed =
                                                ui.input(|i| i.key_pressed(egui::Key::Enter));
                                            let escape_pressed =
                                                ui.input(|i| i.key_pressed(egui::Key::Escape));

                                            if escape_pressed {
                                                self.editing_cell = None;
                                            } else if enter_pressed || text_edit.lost_focus() {
                                                if let Ok(new_val) = buf.trim().parse::<f64>() {
                                                    source.set_cell_value(
                                                        actual_row, col_idx, new_val,
                                                    );
                                                    cell_edit_result = Some(CellEdit {
                                                        row: actual_row,
                                                        col: col_idx,
                                                        new_value: new_val,
                                                    });
                                                }
                                                self.editing_cell = None;
                                            }
                                        }
                                    } else {
                                        let val = source.cell_value(actual_row, col_idx);
                                        let display_str = if val.is_nan() {
                                            "NaN".to_string()
                                        } else {
                                            format!("{:.*}", self.precision, val)
                                        };

                                        let can_edit = self.editable
                                            && source.is_editable(actual_row, col_idx);
                                        let label = ui.add(
                                            egui::Button::selectable(false, display_str)
                                                .sense(Sense::click()),
                                        );

                                        if can_edit && (label.double_clicked() || label.clicked()) {
                                            self.editing_cell = Some((
                                                actual_row,
                                                col_idx,
                                                if val.is_nan() {
                                                    "".to_string()
                                                } else {
                                                    val.to_string()
                                                },
                                            ));
                                        }
                                    }
                                },
                            );
                        }
                    });
                }
            },
        );

        cell_edit_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_matrix_grid() {
        let headers = vec!["ColA".to_string(), "ColB".to_string()];
        let data = vec![vec![1.234, 5.678], vec![9.101, 11.1213]];
        let mut grid = SimpleMatrixGrid::new(headers, data)
            .with_row_labels(vec!["Row 0".to_string(), "Row 1".to_string()]);

        assert_eq!(grid.num_rows(), 2);
        assert_eq!(grid.num_cols(), 2);
        assert_eq!(grid.header(0), "ColA");
        assert_eq!(grid.row_label(1), Some("Row 1".to_string()));
        assert_eq!(grid.cell_value(0, 1), 5.678);

        grid.set_cell_value(0, 1, 42.0);
        assert_eq!(grid.cell_value(0, 1), 42.0);
    }

    #[test]
    fn test_synchronized_time_series_grid() {
        let s1: [[f64; 2]; 3] = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]];
        let s2: [[f64; 2]; 3] = [[0.0, 10.0], [1.5, 20.0], [2.0, 30.0]];
        let datasets = [("Series 1", &s1[..]), ("Series 2", &s2[..])];

        let mut grid = SynchronizedTimeSeriesGrid::from_datasets(&datasets);

        assert_eq!(grid.num_cols(), 4); // Step, X/Time, Series 1, Series 2
        assert_eq!(grid.header(2), "Series 1");
        assert_eq!(grid.header(3), "Series 2");
        assert_eq!(grid.num_rows(), 4); // x = 0.0, 1.0, 1.5, 2.0

        // At x = 1.0, Series 2 is missing (NaN)
        assert_eq!(grid.cell_value(1, 1), 1.0);
        assert_eq!(grid.cell_value(1, 2), 2.0);
        assert!(grid.cell_value(1, 3).is_nan());

        // Cell edit
        grid.set_cell_value(0, 2, 99.0);
        assert_eq!(grid.cell_value(0, 2), 99.0);
        assert!(!grid.is_editable(0, 0)); // Step is read-only
        assert!(grid.is_editable(0, 2)); // Series columns editable
    }

    #[test]
    fn test_numerical_data_grid_export() {
        let headers = vec!["X".to_string(), "Y".to_string()];
        let data = vec![vec![1.0, 2.34567], vec![2.0, 4.56789]];
        let grid_source = SimpleMatrixGrid::new(headers, data);

        let widget = NumericalDataGrid::new().precision(2);
        let csv = widget.export_csv(&grid_source);
        assert!(csv.contains("X,Y"));
        assert!(csv.contains("1.00,2.35"));
        assert!(csv.contains("2.00,4.57"));

        let tsv = widget.export_tsv(&grid_source);
        assert!(tsv.contains("X\tY"));
        assert!(tsv.contains("1.00\t2.35"));
    }
}
