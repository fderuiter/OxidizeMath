#[allow(missing_docs)]
pub trait BoundaryCondition {
    /// Given a coordinate (x, y) and grid dimensions (width, height),
    /// resolve the target index based on boundary logic.
    fn resolve(&self, x: isize, y: isize, width: usize, height: usize) -> Option<usize>;
}

#[allow(missing_docs)]
pub struct NeumannBoundary;
impl BoundaryCondition for NeumannBoundary {
    fn resolve(&self, x: isize, y: isize, width: usize, height: usize) -> Option<usize> {
        let mut x_clamped = x;
        let mut y_clamped = y;

        if x_clamped < 0 {
            x_clamped = 0;
        }
        if x_clamped >= width as isize {
            x_clamped = width as isize - 1;
        }
        if y_clamped < 0 {
            y_clamped = 0;
        }
        if y_clamped >= height as isize {
            y_clamped = height as isize - 1;
        }

        Some((y_clamped as usize) * width + (x_clamped as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neumann_boundary_interior() {
        let bc = NeumannBoundary;
        assert_eq!(bc.resolve(0, 0, 4, 3), Some(0));
        assert_eq!(bc.resolve(1, 1, 4, 3), Some(5));
        assert_eq!(bc.resolve(3, 2, 4, 3), Some(11));
    }

    #[test]
    fn test_neumann_boundary_negative_coordinates() {
        let bc = NeumannBoundary;
        // Clamping negative x: (-1, 2) -> (0, 2) => 2 * 4 + 0 = 8
        assert_eq!(bc.resolve(-1, 2, 4, 3), Some(8));
        // Clamping negative y: (2, -3) -> (2, 0) => 0 * 4 + 2 = 2
        assert_eq!(bc.resolve(2, -3, 4, 3), Some(2));
        // Clamping both negative: (-5, -10) -> (0, 0) => 0
        assert_eq!(bc.resolve(-5, -10, 4, 3), Some(0));
    }

    #[test]
    fn test_neumann_boundary_exceeding_coordinates() {
        let bc = NeumannBoundary;
        // Clamping exceeding x: (4, 1) -> (3, 1) => 1 * 4 + 3 = 7
        assert_eq!(bc.resolve(4, 1, 4, 3), Some(7));
        assert_eq!(bc.resolve(10, 1, 4, 3), Some(7));
        // Clamping exceeding y: (2, 3) -> (2, 2) => 2 * 4 + 2 = 10
        assert_eq!(bc.resolve(2, 3, 4, 3), Some(10));
        assert_eq!(bc.resolve(2, 20, 4, 3), Some(10));
        // Clamping both exceeding: (10, 15) -> (3, 2) => 2 * 4 + 3 = 11
        assert_eq!(bc.resolve(10, 15, 4, 3), Some(11));
    }

    #[test]
    fn test_neumann_boundary_mixed_clamping() {
        let bc = NeumannBoundary;
        // Negative x, exceeding y: (-2, 10) -> (0, 2) => 2 * 4 + 0 = 8
        assert_eq!(bc.resolve(-2, 10, 4, 3), Some(8));
        // Exceeding x, negative y: (10, -5) -> (3, 0) => 0 * 4 + 3 = 3
        assert_eq!(bc.resolve(10, -5, 4, 3), Some(3));
    }
}
