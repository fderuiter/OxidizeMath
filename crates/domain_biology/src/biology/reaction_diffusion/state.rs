use nalgebra::DMatrix;
use pure_math::pure_math::analysis::ode::traits::VectorOperations;
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// Represents the state of a multi-species chemical system.
///
/// Stores concentrations in a flattened "Structure of Arrays" format using a DMatrix
/// to ensure contiguous memory allocation in column-major orientation.
/// The layout is such that each column represents a species, where nrows = grid_size and ncols = num_species.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalState {
    #[allow(missing_docs)]
    pub grid: DMatrix<f64>,
}

impl ChemicalState {
    /// Creates a new zero-initialized chemical state.
    #[verified_engine::verified]
    pub fn new(num_species: usize, grid_size: usize) -> Self {
        Self {
            grid: DMatrix::from_element(grid_size, num_species, 0.0),
        }
    }

    /// Returns the number of chemical species.
    #[inline]
    #[verified_engine::verified]
    pub fn num_species(&self) -> usize {
        self.grid.ncols()
    }

    /// Returns the size of the spatial grid.
    #[inline]
    #[verified_engine::verified]
    pub fn grid_size(&self) -> usize {
        self.grid.nrows()
    }

    /// Returns a reference to the concentration slice for a specific species.
    #[inline]
    #[verified_engine::verified]
    pub fn species(&self, index: usize) -> &[f64] {
        let start = index * self.grid_size();
        &self.grid.as_slice()[start..start + self.grid_size()]
    }

    /// Returns a mutable reference to the concentration slice for a specific species.
    #[inline]
    #[verified_engine::verified]
    pub fn species_mut(&mut self, index: usize) -> &mut [f64] {
        let start = index * self.grid_size();
        let size = self.grid_size();
        &mut self.grid.as_mut_slice()[start..start + size]
    }
}

// Implement standard ops for ODE integration compatibility
impl Add for ChemicalState {
    type Output = Self;

    #[verified_engine::verified]
    fn add(mut self, rhs: Self) -> Self {
        for (val, r_val) in self
            .grid
            .as_mut_slice()
            .iter_mut()
            .zip(rhs.grid.as_slice().iter())
        {
            *val += r_val;
        }
        self
    }
}

impl AddAssign for ChemicalState {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        for (val, r_val) in self
            .grid
            .as_mut_slice()
            .iter_mut()
            .zip(rhs.grid.as_slice().iter())
        {
            *val += r_val;
        }
    }
}

impl Mul<f64> for ChemicalState {
    type Output = Self;

    #[verified_engine::verified]
    fn mul(mut self, scalar: f64) -> Self {
        for val in self.grid.as_mut_slice().iter_mut() {
            *val *= scalar;
        }
        self
    }
}

impl MulAssign<f64> for ChemicalState {
    #[verified_engine::verified]
    fn mul_assign(&mut self, scalar: f64) {
        for val in self.grid.as_mut_slice().iter_mut() {
            *val *= scalar;
        }
    }
}

impl VectorOperations for ChemicalState {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (val, r_val) in self
            .grid
            .as_mut_slice()
            .iter_mut()
            .zip(other.grid.as_slice().iter())
        {
            *val += r_val * scale;
        }
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        if self.grid.nrows() != other.grid.nrows() || self.grid.ncols() != other.grid.ncols() {
            self.grid = other.grid.clone();
            return;
        }
        self.grid
            .as_mut_slice()
            .copy_from_slice(other.grid.as_slice());
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        if self.grid.nrows() != source.grid.nrows() || self.grid.ncols() != source.grid.ncols() {
            self.grid = DMatrix::from_element(source.grid.nrows(), source.grid.ncols(), 0.0);
        }

        for ((dst, src), oth) in self
            .grid
            .as_mut_slice()
            .iter_mut()
            .zip(source.grid.as_slice().iter())
            .zip(other.grid.as_slice().iter())
        {
            *dst = *src + *oth * scale;
        }
    }

    #[verified_engine::verified]
    fn take_for_step(&mut self) -> Self {
        std::mem::replace(
            self,
            ChemicalState {
                grid: DMatrix::zeros(0, 0),
            },
        )
    }
}
