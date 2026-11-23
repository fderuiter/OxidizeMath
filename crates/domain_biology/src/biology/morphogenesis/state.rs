use pure_math::pure_math::analysis::ode::VectorOperations;
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// Represents the state of a Turing system at a point in time.
///
/// This struct encapsulates the concentration vectors for N species,
/// protecting them from invalid resizing while providing safe access.
#[derive(Debug, Clone, PartialEq)]
pub struct TuringState<const N: usize = 2> {
    pub(crate) concentrations: [Vec<f64>; N],
}

impl<const N: usize> TuringState<N> {
    /// Creates a new zero-initialized state of a given size.
    #[verified_engine::verified]
    pub fn new(size: usize) -> Self {
        // specific initialization for N=0 case handled by array::from_fn gracefully
        // (produces empty array)
        let concentrations = std::array::from_fn(|_| vec![0.0; size]);
        Self { concentrations }
    }

    /// Returns the length of the grid (number of spatial points).
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        if N > 0 {
            self.concentrations[0].len()
        } else {
            0
        }
    }

    /// Returns true if the grid is empty.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        if N > 0 {
            self.concentrations[0].is_empty()
        } else {
            true
        }
    }
}

// Backward compatibility for N=2
impl TuringState<2> {
    /// Returns a slice of the activator concentrations.
    #[verified_engine::verified]
    pub fn u(&self) -> &[f64] {
        &self.concentrations[0]
    }

    /// Returns a slice of the inhibitor concentrations.
    #[verified_engine::verified]
    pub fn v(&self) -> &[f64] {
        &self.concentrations[1]
    }

    /// Returns a mutable slice of the activator concentrations.
    #[verified_engine::verified]
    pub fn u_mut(&mut self) -> &mut [f64] {
        &mut self.concentrations[0]
    }

    /// Returns a mutable slice of the inhibitor concentrations.
    #[verified_engine::verified]
    pub fn v_mut(&mut self) -> &mut [f64] {
        &mut self.concentrations[1]
    }
}

impl<const N: usize> Add for TuringState<N> {
    type Output = Self;

    #[verified_engine::verified]
    fn add(mut self, rhs: Self) -> Self {
        for (s, r_vec) in self
            .concentrations
            .iter_mut()
            .zip(rhs.concentrations.iter())
        {
            for (val, r_val) in s.iter_mut().zip(r_vec.iter()) {
                *val += r_val;
            }
        }
        self
    }
}

impl<const N: usize> AddAssign for TuringState<N> {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        for (s, r_vec) in self
            .concentrations
            .iter_mut()
            .zip(rhs.concentrations.iter())
        {
            for (val, r_val) in s.iter_mut().zip(r_vec.iter()) {
                *val += r_val;
            }
        }
    }
}

impl<const N: usize> Mul<f64> for TuringState<N> {
    type Output = Self;

    #[verified_engine::verified]
    fn mul(mut self, scalar: f64) -> Self {
        for s in self.concentrations.iter_mut() {
            for val in s.iter_mut() {
                *val *= scalar;
            }
        }
        self
    }
}

impl<const N: usize> MulAssign<f64> for TuringState<N> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, scalar: f64) {
        for s in self.concentrations.iter_mut() {
            for val in s.iter_mut() {
                *val *= scalar;
            }
        }
    }
}

impl<const N: usize> VectorOperations for TuringState<N> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (s, r_vec) in self
            .concentrations
            .iter_mut()
            .zip(other.concentrations.iter())
        {
            for (val, r_val) in s.iter_mut().zip(r_vec.iter()) {
                *val += r_val * scale;
            }
        }
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        for (s, r_vec) in self
            .concentrations
            .iter_mut()
            .zip(other.concentrations.iter())
        {
            if s.len() != r_vec.len() {
                s.resize(r_vec.len(), 0.0);
            }
            s.copy_from_slice(r_vec);
        }
    }

    #[verified_engine::verified]
    fn take_for_step(&mut self) -> Self {
        std::mem::replace(
            self,
            TuringState {
                concentrations: std::array::from_fn(|_| Vec::new()),
            },
        )
    }
}
