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

    /// Required `f64` buffer length (`(Q + 3) * width * height`) for [`Self::export_checkpoint`].
    #[verified_engine::verified]
    pub fn checkpoint_field_len(&self) -> usize {
        (Q + 3) * self.width * self.height
    }

    /// Exports populations `f`, macroscopic fields `(rho, ux, uy)`, force scalars, and obstacles
    /// into preallocated slices without heap allocation.
    #[verified_engine::verified]
    pub fn export_checkpoint(
        &self,
        scalars_out: &mut [f64; 4],
        field_out: &mut [f64],
        obstacles_out: &mut [u8],
    ) {
        let cells = self.width * self.height;
        let f_len = Q * cells;
        assert!(
            field_out.len() >= f_len + 3 * cells,
            "Checkpoint field buffer too small"
        );
        assert!(
            obstacles_out.len() >= cells,
            "Checkpoint obstacle buffer too small"
        );
        scalars_out[0] = self.last_drag_force;
        scalars_out[1] = self.last_lift_force;
        scalars_out[2] = if self.has_stepped_forces { 1.0 } else { 0.0 };
        scalars_out[3] = 0.0;
        let mut cursor = 0;
        for pop in self.f.as_slice() {
            field_out[cursor..cursor + Q].copy_from_slice(pop);
            cursor += Q;
        }
        field_out[f_len..f_len + cells].copy_from_slice(self.rho.as_slice());
        field_out[f_len + cells..f_len + 2 * cells].copy_from_slice(self.ux.as_slice());
        field_out[f_len + 2 * cells..f_len + 3 * cells].copy_from_slice(self.uy.as_slice());
        for (dst, &obs) in obstacles_out
            .iter_mut()
            .zip(self.obstacles.as_slice().iter())
        {
            *dst = u8::from(obs);
        }
    }

    /// Restores populations `f`, macroscopic fields `(rho, ux, uy)`, force scalars, and obstacles
    /// from preallocated checkpoint slices without heap allocation.
    #[verified_engine::verified]
    pub fn restore_checkpoint(&mut self, scalars: &[f64; 4], field: &[f64], obstacles: &[u8]) {
        let cells = self.width * self.height;
        let f_len = Q * cells;
        assert!(
            field.len() >= f_len + 3 * cells,
            "Checkpoint field buffer too small"
        );
        assert!(
            obstacles.len() >= cells,
            "Checkpoint obstacle buffer too small"
        );
        self.last_drag_force = scalars[0];
        self.last_lift_force = scalars[1];
        self.has_stepped_forces = scalars[2] >= 0.5;
        let mut cursor = 0;
        for pop in self.f.as_mut_slice() {
            pop.copy_from_slice(&field[cursor..cursor + Q]);
            cursor += Q;
        }
        self.rho
            .as_mut_slice()
            .copy_from_slice(&field[f_len..f_len + cells]);
        self.ux
            .as_mut_slice()
            .copy_from_slice(&field[f_len + cells..f_len + 2 * cells]);
        self.uy
            .as_mut_slice()
            .copy_from_slice(&field[f_len + 2 * cells..f_len + 3 * cells]);
        for (dst, &b) in self
            .obstacles
            .as_mut_slice()
            .iter_mut()
            .zip(obstacles.iter())
        {
            *dst = b != 0;
        }
    }
}
