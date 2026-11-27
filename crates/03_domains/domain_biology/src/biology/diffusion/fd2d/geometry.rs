use math_commons::math_kernel::types::{Dimension, StepSize};

/// Strategy describing 2D grid dimensions and uniform spatial step sizes.
pub trait GeometryStrategy {
    /// Returns the number of grid columns $N_x$.
    #[verified_engine::verified]
    fn width(&self) -> Dimension;
    /// Returns the number of grid rows $N_y$.
    #[verified_engine::verified]
    fn height(&self) -> Dimension;
    /// Returns the horizontal grid spacing $\Delta x > 0$.
    #[verified_engine::verified]
    fn dx(&self) -> StepSize;
    /// Returns the vertical grid spacing $\Delta y > 0$.
    #[verified_engine::verified]
    fn dy(&self) -> StepSize;
    /// Returns the total flattened cell count $N_x \cdot N_y$.
    #[verified_engine::verified]
    fn size(&self) -> usize {
        *self.width() * *self.height()
    }
}

/// Uniform 2D Cartesian grid geometry $(N_x, N_y, \Delta x, \Delta y)$.
#[derive(Debug, Clone, Copy)]
pub struct Cartesian2D {
    /// Number of grid columns $N_x$.
    pub width: Dimension,
    /// Number of grid rows $N_y$.
    pub height: Dimension,
    /// Horizontal grid spacing $\Delta x$.
    pub dx: StepSize,
    /// Vertical grid spacing $\Delta y$.
    pub dy: StepSize,
}

impl GeometryStrategy for Cartesian2D {
    #[verified_engine::verified]
    fn width(&self) -> Dimension {
        self.width
    }

    #[verified_engine::verified]
    fn height(&self) -> Dimension {
        self.height
    }

    #[verified_engine::verified]
    fn dx(&self) -> StepSize {
        self.dx
    }

    #[verified_engine::verified]
    fn dy(&self) -> StepSize {
        self.dy
    }
}
