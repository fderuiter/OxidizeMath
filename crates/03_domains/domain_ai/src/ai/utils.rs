// Utility functions and traits for AI modules.
use nalgebra::{DMatrix, RowDVector};

/// An extension trait for `DMatrix` to allow broadcasting addition of a row vector.
pub trait AddRowVector {
    /// Adds a row vector to every row of the matrix.
    #[verified_engine::verified]
    fn add_row_vector_to_all_rows(&mut self, row_vector: &RowDVector<f64>);
}

impl AddRowVector for DMatrix<f64> {
    #[verified_engine::verified]
    fn add_row_vector_to_all_rows(&mut self, row_vector: &RowDVector<f64>) {
        assert_eq!(
            self.ncols(),
            row_vector.len(),
            "Matrix columns must match row vector length for broadcasting."
        );
        for mut row in self.row_iter_mut() {
            row += row_vector;
        }
    }
}
