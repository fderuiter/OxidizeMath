#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use egui_plot::numerical_data_grid::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexported_data_grid_structures() {
        let headers = vec!["Col 1".to_string(), "Col 2".to_string()];
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut grid = SimpleMatrixGrid::new(headers, data);

        assert_eq!(grid.num_rows(), 2);
        assert_eq!(grid.num_cols(), 2);
        assert_eq!(grid.cell_value(0, 0), 1.0);

        grid.set_cell_value(0, 0, 42.0);
        assert_eq!(grid.cell_value(0, 0), 42.0);

        let widget = NumericalDataGrid::new().precision(3);
        let csv = widget.export_csv(&grid);
        assert!(csv.contains("42.000"));
    }

    #[test]
    fn test_time_series_sync_grid() {
        let s1: [[f64; 2]; 2] = [[0.0, 10.0], [1.0, 20.0]];
        let s2: [[f64; 2]; 2] = [[0.0, 5.0], [2.0, 15.0]];
        let datasets = [("Series 1", &s1[..]), ("Series 2", &s2[..])];

        let sync_grid = SynchronizedTimeSeriesGrid::from_datasets(&datasets);
        assert_eq!(sync_grid.num_cols(), 4); // Step, Time, Series 1, Series 2
        assert_eq!(sync_grid.num_rows(), 3); // 0.0, 1.0, 2.0
    }
}
