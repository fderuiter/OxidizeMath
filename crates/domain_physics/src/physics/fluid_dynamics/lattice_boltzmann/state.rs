use crate::error::FluidError;
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
    /// Drag force accumulated during the most recent streaming bounce-back pass.
    pub(crate) last_drag_force: f64,
    /// Lift force accumulated during the most recent streaming bounce-back pass.
    pub(crate) last_lift_force: f64,
    /// Whether `last_drag_force` and `last_lift_force` come from a completed streaming pass.
    pub(crate) has_stepped_forces: bool,
}

impl<const Q: usize> DoubleBufferedState for LatticeState<Q> {
    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.f, &mut self.f_new);
    }
}

impl<const Q: usize> LatticeState<Q> {
    /// Fallibly constructs a new Lattice Boltzmann state grid with checked allocation bounds.
    ///
    /// Returns [`FluidError::GridDimensionsOverflow`] if `width * height` or the byte footprint of
    /// a distribution buffer overflows addressable memory (`isize::MAX`).
    #[verified_engine::verified]
    pub fn try_new(width: usize, height: usize) -> Result<Self, FluidError> {
        let cell_bytes = std::mem::size_of::<[f64; Q]>().max(1);
        let valid_capacity = width
            .checked_mul(height)
            .and_then(|cells| cells.checked_mul(cell_bytes))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
        if !valid_capacity {
            return Err(FluidError::GridDimensionsOverflow { width, height });
        }
        Ok(Self {
            width,
            height,
            f: DMatrix::from_element(height, width, [0.0; Q]),
            f_new: DMatrix::from_element(height, width, [0.0; Q]),
            rho: DMatrix::from_element(height, width, 1.0),
            ux: DMatrix::from_element(height, width, 0.0),
            uy: DMatrix::from_element(height, width, 0.0),
            obstacles: DMatrix::from_element(height, width, false),
            last_drag_force: 0.0,
            last_lift_force: 0.0,
            has_stepped_forces: false,
        })
    }

    /// Creates a new state with the given dimensions.
    /// Initialized with:
    /// - Density = 1.0
    /// - Velocity = 0.0
    /// - f = 0.0 (Caller must initialize equilibrium)
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize) -> Self {
        Self::try_new(width, height).expect("Grid dimensions too large")
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
}
