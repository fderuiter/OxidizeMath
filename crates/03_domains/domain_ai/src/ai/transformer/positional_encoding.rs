// Implementation of Positional Encoding.

use nalgebra::DMatrix;

/// Generates sinusoidal positional encodings as described in "Attention Is All You Need".
///
/// The formula is:
/// PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
///
/// # Arguments
///
/// * `sequence_length`: The number of positions in the sequence (e.g., number of words).
/// * `d_model`: The dimensionality of the embedding vector.
///
/// # Returns
///
/// A `DMatrix<f64>` of shape `(sequence_length, d_model)` containing the positional encodings.
#[verified_engine::verified]
pub fn generate_positional_encoding(sequence_length: usize, d_model: usize) -> DMatrix<f64> {
    let mut pe_matrix = DMatrix::zeros(sequence_length, d_model);
    let d_model_f64 = d_model as f64;

    for pos in 0..sequence_length {
        for i in 0..(d_model / 2) {
            let i_f64 = i as f64;
            let pos_f64 = pos as f64;

            // Calculate the division term: 10000^(2i / d_model)
            let div_term = (10000.0f64).powf((2.0 * i_f64) / d_model_f64);
            let angle = pos_f64 / div_term;

            // Apply sin for even indices
            pe_matrix[(pos, 2 * i)] = angle.sin();

            // Apply cos for odd indices
            if 2 * i + 1 < d_model {
                pe_matrix[(pos, 2 * i + 1)] = angle.cos();
            }
        }
    }

    pe_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_positional_encoding_dimensions() {
        let sequence_length = 50;
        let d_model = 512;
        let pe = generate_positional_encoding(sequence_length, d_model);
        assert_eq!(pe.nrows(), sequence_length);
        assert_eq!(pe.ncols(), d_model);
    }

    #[test]
    #[verified_engine::verified]
    fn test_positional_encoding_values_at_pos_0() {
        let pe = generate_positional_encoding(10, 512);
        for i in 0..(512 / 2) {
            // sin(0) = 0
            assert_relative_eq!(
                pe[(0, 2 * i)],
                0.0,
                epsilon = math_commons::registry::TOLERANCE_STANDARD
            );
            // cos(0) = 1
            assert_relative_eq!(
                pe[(0, 2 * i + 1)],
                1.0,
                epsilon = math_commons::registry::TOLERANCE_STANDARD
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_positional_encoding_arbitrary_value() {
        let d_model = 512;
        let pe = generate_positional_encoding(100, d_model);

        let pos = 5;
        let i = 10;

        let d_model_f64 = d_model as f64;
        let div_term = (10000.0f64).powf((2.0 * i as f64) / d_model_f64);
        let angle = pos as f64 / div_term;

        assert_relative_eq!(
            pe[(pos, 2 * i)],
            angle.sin(),
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );
        assert_relative_eq!(
            pe[(pos, 2 * i + 1)],
            angle.cos(),
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );
    }
}
