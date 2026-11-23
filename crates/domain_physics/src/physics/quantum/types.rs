// [cite:quantum_mechanics]
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Represents the state of a quantum system as a vector in a Hilbert space.
///
/// In quantum mechanics, the state of a system is described by a state vector |\psi\rangle
/// belonging to a complex Hilbert space. The state vector contains all the information
/// about the system.
///
/// The probability of finding the system in a particular state is related to the
/// square of the absolute value of the vector's components (Born Rule).
#[derive(Clone, Debug)]
pub struct QuantumState {
    /// The complex vector representing the state |\psi\rangle.
    pub vector: DVector<Complex<f64>>,
}

impl TheoryDescribable for QuantumState {
    #[verified_engine::verified]
    fn theory_description(&self) -> String {
        // Use a heuristic to detect the |0> spin state
        if self.vector.len() == 2
            && self.vector[0].re == 1.0
            && self.vector[0].im == 0.0
            && self.vector[1].re == 0.0
            && self.vector[1].im == 0.0
        {
            "|0> spin state".to_string()
        } else {
            "Quantum state vector".to_string()
        }
    }

    #[verified_engine::verified]
    fn phonetic_description(&self) -> String {
        if self.vector.len() == 2
            && self.vector[0].re == 1.0
            && self.vector[0].im == 0.0
            && self.vector[1].re == 0.0
            && self.vector[1].im == 0.0
        {
            "Ket zero spin state".to_string()
        } else {
            "Quantum state vector".to_string()
        }
    }

    #[verified_engine::verified]
    fn theory_citation(&self) -> String {
        "Griffiths, D. J., & Schroeter, D. F. (2018). Introduction to Quantum Mechanics (3rd ed.). Cambridge University Press.".to_string()
    }

    #[verified_engine::verified]
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("|0>".to_string(), "|0> spin state".to_string());
        map.insert("default".to_string(), "Quantum state vector".to_string());
        map
    }
}

impl QuantumState {
    /// Returns the |0> spin state (spin up along Z)
    #[verified_engine::verified]
    pub fn spin_zero() -> Self {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        Self::new(DVector::from_vec(vec![one, zero]))
    }

    /// Creates a new QuantumState from a vector of complex numbers.
    ///
    /// # Arguments
    /// * `vector` - A `DVector<Complex<f64>>` representing the state.
    #[verified_engine::verified]
    pub fn new(vector: DVector<Complex<f64>>) -> Self {
        Self { vector }
    }

    /// Calculates the norm (magnitude) of the state vector.
    ///
    /// \sqrt{\langle\psi|\psi\rangle}
    #[verified_engine::verified]
    pub fn norm(&self) -> f64 {
        self.vector.norm()
    }

    /// Normalizes the state vector so that its norm is 1.
    ///
    /// |\psi'\rangle = \frac{|\psi\rangle}{\sqrt{\langle\psi|\psi\rangle}}
    ///
    /// Returns a new QuantumState.
    #[verified_engine::verified]
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            // Can't normalize zero vector, return as is or handle error.
            // Returning as is for now, but in QM this is invalid state.
            return self.clone();
        }
        Self {
            // Need to divide by complex or scale by 1/norm
            // Using a reference avoids an unnecessary heap allocation of the intermediate vector
            vector: &self.vector / Complex::new(norm, 0.0),
        }
    }

    /// Normalizes the state vector in place so that its norm is 1, without heap allocations.
    ///
    /// |\psi'\rangle = \frac{|\psi\rangle}{\sqrt{\langle\psi|\psi\rangle}}
    ///
    /// Leaves zero-norm vectors unchanged.
    #[verified_engine::verified]
    pub fn normalize_in_place(&mut self) {
        let norm = self.norm();
        if norm == 0.0 {
            return;
        }
        self.vector /= Complex::new(norm, 0.0);
    }

    /// Calculates the inner product with another state (Bra-Ket notation: \langle\phi|\psi\rangle).
    ///
    /// This is equivalent to the dot product of the conjugate transpose of `other` (Bra)
    /// with `self` (Ket). Note: In physics notation <phi|psi>, phi is the bra.
    /// Here `self.inner_product(&other)` computes <self|other>.
    ///
    /// # Arguments
    /// * `other` - The Ket vector |\phi\rangle (argument).
    ///
    /// # Returns
    /// The complex scalar result of \langle\psi|\phi\rangle.
    #[verified_engine::verified]
    pub fn inner_product(&self, other: &QuantumState) -> Complex<f64> {
        self.vector.dotc(&other.vector)
    }

    /// Returns the probability density of the state.
    ///
    /// For a discrete state vector, this returns a vector of real probabilities
    /// corresponding to each basis state: P_i = |\psi_i|^2.
    #[verified_engine::verified]
    pub fn probability_density(&self) -> DVector<f64> {
        self.vector.map(|c| c.norm_sqr())
    }

    /// Projects the complex state vector onto the first `M` discrete orthonormal
    /// standing-wave spatial eigenmodes `<phi_m | psi>` without heap allocation,
    /// preserving complex amplitudes and phases prior to Born-rule norm collapse.
    #[verified_engine::verified]
    #[must_use]
    pub fn project_spatial_modes<const M: usize>(&self) -> [Complex<f64>; M] {
        let mut modes = [Complex::new(0.0, 0.0); M];
        let n = self.vector.len();
        if n == 0 || M == 0 {
            return modes;
        }
        let denom = (n + 1) as f64;
        let norm_factor = (2.0 / denom).sqrt();
        for (m, slot) in modes.iter_mut().enumerate() {
            if m >= n {
                break;
            }
            let k_pi = ((m + 1) as f64) * std::f64::consts::PI / denom;
            let mut acc = Complex::new(0.0, 0.0);
            for (j, amp) in self.vector.iter().enumerate() {
                let basis = norm_factor * (((j + 1) as f64) * k_pi).sin();
                acc += (*amp) * basis;
            }
            *slot = acc;
        }
        modes
    }
}

/// Represents a linear operator acting on the Hilbert space.
///
/// In quantum mechanics, physical observables are represented by linear operators.
/// Hermitian operators correspond to measurable quantities.
#[derive(Clone, Debug)]
pub struct QuantumOperator {
    /// The complex matrix representing the operator \hat{A}.
    pub matrix: DMatrix<Complex<f64>>,
}

impl QuantumOperator {
    /// Creates a new QuantumOperator from a complex matrix.
    #[verified_engine::verified]
    pub fn new(matrix: DMatrix<Complex<f64>>) -> Self {
        Self { matrix }
    }

    /// Checks if the operator is Hermitian (Self-adjoint).
    ///
    /// \hat{A} = \hat{A}^\dagger
    #[verified_engine::verified]
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        let adjoint = self.matrix.adjoint();
        // Using a reference avoids an unnecessary heap allocation of the intermediate matrix
        (&self.matrix - adjoint).norm() < tolerance
    }

    /// Calculates the expectation value of the operator for a given state.
    ///
    /// \langle \hat{A} \rangle = \langle \psi | \hat{A} | \psi \rangle
    #[verified_engine::verified]
    pub fn expectation_value(&self, state: &QuantumState) -> Complex<f64> {
        // <psi| (A |psi>)
        let bra = state.vector.adjoint();
        let ket_transformed = &self.matrix * &state.vector;
        (bra * ket_transformed)[(0, 0)]
    }

    /// Calculates the commutator with another operator.
    ///
    /// [\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}
    #[verified_engine::verified]
    pub fn commutator(&self, other: &QuantumOperator) -> QuantumOperator {
        let ab = &self.matrix * &other.matrix;
        let ba = &other.matrix * &self.matrix;
        QuantumOperator { matrix: ab - ba }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_spatial_modes_orthonormal_and_phase_preserving() {
        let n = 31usize;
        let denom = (n + 1) as f64;
        let norm = (2.0 / denom).sqrt();
        let phase = Complex::from_polar(1.0, 0.75);
        let vec = DVector::from_fn(n, |j, _| {
            let s1 = norm * (((j + 1) as f64) * std::f64::consts::PI / denom).sin();
            let s3 = norm * (3.0 * ((j + 1) as f64) * std::f64::consts::PI / denom).sin();
            Complex::new(0.6 * s1, 0.0) + phase * (0.8 * s3)
        });
        let state = QuantumState::new(vec);
        let modes = state.project_spatial_modes::<4>();
        assert!((modes[0].re - 0.6).abs() < 1e-12);
        assert!(modes[0].im.abs() < 1e-12);
        assert!(modes[1].norm() < 1e-12);
        assert!((modes[2] - phase * 0.8).norm() < 1e-12);
        assert!(modes[3].norm() < 1e-12);
    }

    #[test]
    fn test_project_spatial_modes_empty_and_oversized_basis() {
        let empty = QuantumState::new(DVector::zeros(0));
        let empty_modes = empty.project_spatial_modes::<3>();
        assert_eq!(empty_modes, [Complex::new(0.0, 0.0); 3]);

        let spin = QuantumState::spin_zero();
        let spin_modes = spin.project_spatial_modes::<4>();
        assert!(spin_modes[0].re > 0.5);
        assert_eq!(spin_modes[2], Complex::new(0.0, 0.0));
        assert_eq!(spin_modes[3], Complex::new(0.0, 0.0));
    }
}
