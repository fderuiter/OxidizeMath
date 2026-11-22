use nalgebra::DMatrix;
use pure_math::pure_math::analysis::evolution::DoubleBufferedState;

/// State container for Lattice Boltzmann Simulation.
///
/// Holds the distribution functions and macroscopic variables.
/// This struct is purely a data holder and does not implement the solver logic.
#[derive(Debug, Clone)]
pub struct LatticeState<const Q: usize> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Distribution functions. Each cell holds [f64; Q].
    pub(crate) f: DMatrix<[f64; Q]>,
    /// Buffer for streaming step.
    pub(crate) f_new: DMatrix<[f64; Q]>,
    /// Macroscopic density.
    pub(crate) rho: DMatrix<f64>,
    /// Macroscopic velocity X.
    pub(crate) ux: DMatrix<f64>,
    /// Macroscopic velocity Y.
    pub(crate) uy: DMatrix<f64>,
    /// Boolean grid for obstacles (true = solid).
    pub(crate) obstacles: DMatrix<bool>,
}

impl<const Q: usize> DoubleBufferedState for LatticeState<Q> {
    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.f, &mut self.f_new);
    }
}

impl<const Q: usize> LatticeState<Q> {
    /// Creates a new state with the given dimensions.
    /// Initialized with:
    /// - Density = 1.0
    /// - Velocity = 0.0
    /// - f = 0.0 (Caller must initialize equilibrium)
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize) -> Self {
        let _ = width
            .checked_mul(height)
            .expect("Grid dimensions too large");
        Self {
            width,
            height,
            f: DMatrix::from_element(height, width, [0.0; Q]),
            f_new: DMatrix::from_element(height, width, [0.0; Q]),
            rho: DMatrix::from_element(height, width, 1.0),
            ux: DMatrix::from_element(height, width, 0.0),
            uy: DMatrix::from_element(height, width, 0.0),
            obstacles: DMatrix::from_element(height, width, false),
        }
    }

    /// Checks if the coordinates are within the grid.
    #[inline]
    #[verified_engine::verified]
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    // --- Public Accessors ---

    /// Returns the width of the lattice grid.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the lattice grid.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns a slice of the macroscopic density field.
    #[verified_engine::verified]
    pub fn density(&self) -> &[f64] {
        self.rho.as_slice()
    }

    /// Returns a slice of the macroscopic X-velocity field.
    #[verified_engine::verified]
    pub fn velocity_x(&self) -> &[f64] {
        self.ux.as_slice()
    }

    /// Returns a slice of the macroscopic Y-velocity field.
    #[verified_engine::verified]
    pub fn velocity_y(&self) -> &[f64] {
        self.uy.as_slice()
    }

    /// Returns a slice of the obstacle mask (true = obstacle).
    #[verified_engine::verified]
    pub fn obstacles(&self) -> &[bool] {
        self.obstacles.as_slice()
    }

    /// Copies macroscopic velocity components `(ux, uy)` and `obstacles` from column-major
    /// `DMatrix(height, width)` storage into row-major `(y * width + x)` slices with zero heap allocation.
    #[verified_engine::verified]
    pub fn copy_macroscopic_row_major(
        &self,
        ux_out: &mut [f64],
        uy_out: &mut [f64],
        obstacles_out: &mut [bool],
    ) {
        let total = self.width * self.height;
        assert!(
            ux_out.len() >= total && uy_out.len() >= total && obstacles_out.len() >= total,
            "Output buffers must hold at least width * height elements"
        );
        for y in 0..self.height {
            for x in 0..self.width {
                let dst = math_commons::math_kernel::types::flatten_2d_index(x, y, self.width);
                ux_out[dst] = self.ux[(y, x)];
                uy_out[dst] = self.uy[(y, x)];
                obstacles_out[dst] = self.obstacles[(y, x)];
            }
        }
    }
}
