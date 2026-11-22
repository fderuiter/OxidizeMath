//! Second Quantization (Operators and Fock States)
//!
//! The framework for Many-Body systems where states are defined by occupation numbers
//! rather than individual particle coordinates.

/// Particle statistics type.
pub trait ParticleStatistics {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn validate_set_occupation(current: u8, count: u8) -> Result<(), String>;
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn validate_create_particle(current: u8) -> Result<u8, String>;
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn check_commutation(op1: &Operator, op2: &Operator) -> f64;
}

/// Fermions follow Fermi-Dirac statistics and Pauli Exclusion Principle.
/// {c_i, c_j^\dagger} = \delta_{ij}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fermion;

/// Bosons follow Bose-Einstein statistics.
/// [a_i, a_j^\dagger] = \delta_{ij}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Boson;

/// Type of Quantum Operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantumOperatorType {
    /// Creates a particle in a state (raising operator).
    Creation,
    /// Annihilates a particle from a state (lowering operator).
    Annihilation,
}

/// A quantum operator acting on a specific state index (k-vector or site).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Operator {
    #[allow(missing_docs)]
    pub op_type: QuantumOperatorType,
    #[allow(missing_docs)]
    pub index: usize,
}

impl Operator {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(op_type: QuantumOperatorType, index: usize) -> Self {
        Self { op_type, index }
    }
}

/// Fock State representation using occupation numbers.
/// |n_1, n_2, ..., n_M>
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FockState {
    /// Occupation numbers for each mode/site.
    /// For Fermions, valid values are 0 or 1.
    /// For Bosons, values can be any non-negative integer (limited by u8 here).
    pub occupations: Vec<u8>,
}

impl FockState {
    /// Creates a new vacuum state |0, 0, ...> with given size.
    #[verified_engine::verified]
    pub fn new(size: usize) -> Self {
        Self {
            occupations: vec![0; size],
        }
    }

    /// Sets the occupation number of a specific state directly.
    ///
    /// The `P` type parameter specifies the particle statistics (e.g., `Fermion`, `Boson`)
    /// which will validate the occupation rules.
    #[verified_engine::verified]
    pub fn set_occupation<P: ParticleStatistics>(
        &mut self,
        index: usize,
        count: u8,
    ) -> Result<(), String> {
        if index >= self.occupations.len() {
            return Err(format!("Index {} out of bounds", index));
        }
        P::validate_set_occupation(self.occupations[index], count)?;
        self.occupations[index] = count;
        Ok(())
    }

    /// Tries to add a particle to the state (Apply creation operator).
    ///
    /// The `P` type parameter specifies the particle statistics (e.g., `Fermion`, `Boson`)
    /// which will validate the creation rules.
    #[verified_engine::verified]
    pub fn create_particle<P: ParticleStatistics>(&mut self, index: usize) -> Result<(), String> {
        if index >= self.occupations.len() {
            return Err(format!("Index {} out of bounds", index));
        }
        let current = self.occupations[index];
        let next = P::validate_create_particle(current)?;
        self.occupations[index] = next;
        Ok(())
    }
}

impl ParticleStatistics for Fermion {
    #[verified_engine::verified]
    fn validate_set_occupation(_current: u8, count: u8) -> Result<(), String> {
        if count > 1 {
            return Err(
                "Pauli Exclusion Principle: Fermions cannot occupy same state > 1".to_string(),
            );
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn validate_create_particle(current: u8) -> Result<u8, String> {
        if current >= 1 {
            return Err("Pauli Exclusion: State already occupied".to_string());
        }
        Ok(1)
    }

    #[verified_engine::verified]
    fn check_commutation(op1: &Operator, op2: &Operator) -> f64 {
        // Fermions: Anti-commutator {A, B} = AB + BA
        // {c_i, c_j^\dagger} = delta_{ij}
        match (op1.op_type, op2.op_type) {
            (QuantumOperatorType::Annihilation, QuantumOperatorType::Creation)
            | (QuantumOperatorType::Creation, QuantumOperatorType::Annihilation)
                if op1.index == op2.index =>
            {
                1.0
            }
            _ => 0.0, // {c, c} = 0, {c^\dagger, c^\dagger} = 0, or i != j
        }
    }
}

impl ParticleStatistics for Boson {
    #[verified_engine::verified]
    fn validate_set_occupation(_current: u8, _count: u8) -> Result<(), String> {
        Ok(())
    }

    #[verified_engine::verified]
    fn validate_create_particle(current: u8) -> Result<u8, String> {
        if current == u8::MAX {
            return Err("Boson saturation (u8 max)".to_string());
        }
        Ok(current + 1)
    }

    #[verified_engine::verified]
    fn check_commutation(op1: &Operator, op2: &Operator) -> f64 {
        // Bosons: Commutator [A, B] = AB - BA
        // [a_i, a_j^\dagger] = delta_{ij}
        // [a_i^\dagger, a_j] = -delta_{ij}
        match (op1.op_type, op2.op_type) {
            (QuantumOperatorType::Annihilation, QuantumOperatorType::Creation)
                if op1.index == op2.index =>
            {
                1.0
            }
            (QuantumOperatorType::Creation, QuantumOperatorType::Annihilation)
                if op1.index == op2.index =>
            {
                -1.0
            }
            _ => 0.0, // [a, a] = 0, [a^\dagger, a^\dagger] = 0, or i != j
        }
    }
}

/// Checks the canonical commutation (Boson) or anti-commutation (Fermion) relations.
///
/// The `P` type parameter specifies the particle statistics (e.g., `Fermion`, `Boson`)
/// which will determine the commutation rules.
///
/// Returns the value of:
/// * `{op1, op2}` for Fermions. Expected to be `delta_{ij}` for {c, c^\dagger}.
/// * `[op1, op2]` for Bosons. Expected to be `delta_{ij}` for [a, a^\dagger].
#[verified_engine::verified]
pub fn check_commutation<P: ParticleStatistics>(op1: &Operator, op2: &Operator) -> f64 {
    P::check_commutation(op1, op2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_fermion_exclusion() {
        let mut state = FockState::new(2);
        // First addition should succeed
        assert!(state.create_particle::<Fermion>(0).is_ok());
        // Second addition to same state must fail
        assert!(state.create_particle::<Fermion>(0).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_commutation_logic() {
        let c_k = Operator::new(QuantumOperatorType::Annihilation, 1);
        let c_k_dag = Operator::new(QuantumOperatorType::Creation, 1);
        let c_q = Operator::new(QuantumOperatorType::Annihilation, 2);

        // Fermion: {c_k, c_k^dag} = 1
        let val = check_commutation::<Fermion>(&c_k, &c_k_dag);
        assert!((val - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // Fermion: {c_k, c_q} = 0
        let val2 = check_commutation::<Fermion>(&c_k, &c_q);
        assert!(val2.abs() < math_commons::registry::TOLERANCE_STANDARD);

        // Boson: [a_k, a_k^dag] = 1
        let val3 = check_commutation::<Boson>(&c_k, &c_k_dag);
        assert!((val3 - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }
}
