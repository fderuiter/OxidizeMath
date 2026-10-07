use std::fmt::Write as _;

use crate::framework::InteractiveTool;

/// Representation of exportable simulation dataset (2D point series or grid matrix).
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ExportDataKind {
    Series {
        headers: Vec<String>,
        datasets: Vec<(String, Vec<[f64; 2]>)>,
    },
    Grid {
        row_headers: Vec<String>,
        col_headers: Vec<String>,
        matrix: Vec<Vec<f64>>,
    },
}

/// Container struct for exportable simulation data.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct ExportableData {
    pub title: String,
    pub kind: ExportDataKind,
}

impl ExportableData {
    pub fn series(
        title: impl Into<String>,
        headers: Vec<String>,
        datasets: Vec<(String, Vec<[f64; 2]>)>,
    ) -> Self {
        Self {
            title: title.into(),
            kind: ExportDataKind::Series { headers, datasets },
        }
    }

    pub fn single_series(
        title: impl Into<String>,
        headers: Vec<String>,
        dataset_name: impl Into<String>,
        points: Vec<[f64; 2]>,
    ) -> Self {
        Self::series(title, headers, vec![(dataset_name.into(), points)])
    }

    pub fn grid(
        title: impl Into<String>,
        row_headers: Vec<String>,
        col_headers: Vec<String>,
        matrix: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            title: title.into(),
            kind: ExportDataKind::Grid {
                row_headers,
                col_headers,
                matrix,
            },
        }
    }

    pub fn to_csv(&self) -> String {
        self.format_delimited(',')
    }

    pub fn to_tsv(&self) -> String {
        self.format_delimited('\t')
    }

    fn format_delimited(&self, sep: char) -> String {
        match &self.kind {
            ExportDataKind::Series { headers, datasets } => {
                let total_pts: usize = datasets.iter().map(|(_, pts)| pts.len()).sum();
                let mut out = String::with_capacity(32 + total_pts * 28);
                let h_x = headers.first().map(|s| s.as_str()).unwrap_or("X");
                let h_y = headers.get(1).map(|s| s.as_str()).unwrap_or("Y");
                let _ = writeln!(out, "Dataset{sep}{h_x}{sep}{h_y}");
                for (name, pts) in datasets {
                    for p in pts {
                        let _ = writeln!(out, "{name}{sep}{}{sep}{}", p[0], p[1]);
                    }
                }
                out
            }
            ExportDataKind::Grid {
                row_headers,
                col_headers,
                matrix,
            } => {
                let rows = matrix.len();
                let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
                let mut out = String::with_capacity(32 + rows * (cols + 1) * 12);
                out.push_str("Row/Col");
                for col in col_headers {
                    let _ = write!(out, "{sep}{col}");
                }
                out.push('\n');
                for (r_idx, row) in matrix.iter().enumerate() {
                    let r_head = row_headers
                        .get(r_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("Row {r_idx}"));
                    out.push_str(&r_head);
                    for val in row {
                        let _ = write!(out, "{sep}{val}");
                    }
                    out.push('\n');
                }
                out
            }
        }
    }
}

/// Trait for simulation tools that expose structured data exports.
#[allow(missing_docs)]
pub trait DataExport {
    fn export_data(&self) -> Option<ExportableData> {
        None
    }
    fn to_csv(&self) -> Option<String> {
        self.export_data().map(|data| data.to_csv())
    }
    fn to_tsv(&self) -> Option<String> {
        self.export_data().map(|data| data.to_tsv())
    }
}

impl<T: InteractiveTool + ?Sized> DataExport for T {
    fn export_data(&self) -> Option<ExportableData> {
        InteractiveTool::export_data(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exportable_data_formatting() {
        let data_series = ExportableData::series(
            "ODE Simulation",
            vec!["Time".into(), "State".into()],
            vec![("y(t)".into(), vec![[0.0, 1.0], [1.0, 2.5]])],
        );
        let csv = data_series.to_csv();
        assert!(csv.contains("Dataset,Time,State"));
        assert!(csv.contains("y(t),0,1"));
        let tsv = data_series.to_tsv();
        assert!(tsv.contains("Dataset\tTime\tState"));

        let grid_data = ExportableData::grid(
            "Q-Table",
            vec!["State 0".into(), "State 1".into()],
            vec!["Up".into(), "Down".into()],
            vec![vec![0.5, 0.8], vec![0.1, -0.2]],
        );
        let grid_csv = grid_data.to_csv();
        assert!(grid_csv.contains("Row/Col,Up,Down"));
        assert!(grid_csv.contains("State 0,0.5,0.8"));
        assert!(grid_csv.contains("State 1,0.1,-0.2"));
    }

    #[test]
    fn test_csv_performance_100k_points() {
        let pts = vec![[123.456, 789.012]; 100_000];
        let data = ExportableData::series(
            "Perf",
            vec!["X".into(), "Y".into()],
            vec![("S1".into(), pts)],
        );
        let start = std::time::Instant::now();
        let csv = data.to_csv();
        let elapsed = start.elapsed();
        assert_eq!(csv.lines().count(), 100_001);
        #[cfg(not(debug_assertions))]
        assert!(
            elapsed.as_millis() < 50,
            "Release CSV generation took {} ms",
            elapsed.as_millis()
        );
        #[cfg(debug_assertions)]
        assert!(
            elapsed.as_millis() < 200,
            "Debug CSV generation took {} ms",
            elapsed.as_millis()
        );
    }
}
