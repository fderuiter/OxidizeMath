//! Custom GUI widgets for Math Explorer.

/// Re-exports for the numerical data grid widget.
pub mod numerical_data_grid;

pub use numerical_data_grid::{
    CellEdit, GridDataSource, NumericalDataGrid, SimpleMatrixGrid, SynchronizedTimeSeriesGrid,
};
