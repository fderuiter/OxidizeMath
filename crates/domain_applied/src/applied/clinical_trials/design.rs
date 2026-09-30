use super::types::ClinicalTrialError;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Represents the group assignment for a patient in a clinical trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    #[allow(missing_docs)]
    Treatment,
    #[allow(missing_docs)]
    Control,
}

/// Represents a patient with an ID and a stratum identifier (e.g., "Male-Young").
#[derive(Debug, Clone)]
pub struct Patient {
    #[allow(missing_docs)]
    pub id: String,
    #[allow(missing_docs)]
    pub stratum: String,
}

/// A strategy for allocating subjects to groups.
///
/// This trait adheres to the Strategy Pattern, allowing different randomization
/// algorithms to be swapped interchangeably.
pub trait AllocationStrategy {
    /// Assigns `n_subjects` to groups using the provided RNG.
    #[verified_engine::verified]
    fn assign<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n_subjects: usize,
    ) -> Result<Vec<Group>, ClinicalTrialError>;
}

/// Simple Randomization Strategy.
///
/// Each patient is assigned to Treatment or Control with equal probability (0.5).
pub struct SimpleRandomizer;

impl AllocationStrategy for SimpleRandomizer {
    #[verified_engine::verified]
    fn assign<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n_subjects: usize,
    ) -> Result<Vec<Group>, ClinicalTrialError> {
        let mut assignments = Vec::with_capacity(n_subjects);
        for _ in 0..n_subjects {
            if rng.r#gen::<bool>() {
                assignments.push(Group::Treatment);
            } else {
                assignments.push(Group::Control);
            }
        }
        Ok(assignments)
    }
}

/// Block Randomization Strategy.
///
/// Ensures balanced allocation within blocks of a fixed size.
pub struct BlockRandomizer {
    block_size: usize,
}

impl BlockRandomizer {
    /// Creates a new BlockRandomizer.
    ///
    /// # Arguments
    /// * `block_size` - The size of each block. Must be even.
    #[verified_engine::verified]
    pub fn new(block_size: usize) -> Result<Self, ClinicalTrialError> {
        if !block_size.is_multiple_of(2) {
            return Err(ClinicalTrialError::InvalidData(
                "Block size must be even for 1:1 allocation.".to_string(),
            ));
        }
        Ok(Self { block_size })
    }
}

impl AllocationStrategy for BlockRandomizer {
    /// Assigns subjects to groups using block randomization.
    ///
    /// # Behavior
    ///
    /// This method generates enough full blocks to cover the requested `n_subjects` and then
    /// truncates the list to the exact size.
    ///
    /// If `n_subjects` is not a multiple of the block size, the final block will be incomplete.
    /// This means the assignments in the last incomplete block may not be perfectly balanced
    /// (e.g., you might get 2 Treatments and 1 Control if the block size is 4 and you request 3 subjects).
    ///
    /// # Examples
    ///
    /// ```
    /// use domain_applied::applied::clinical_trials::design::{BlockRandomizer, AllocationStrategy, Group};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = oxidize_core::rng::OxidizeRng::default();
    /// let randomizer = BlockRandomizer::new(4).unwrap();
    ///
    /// // Request 5 subjects (1 full block + 1 partial block)
    /// let assignments = randomizer.assign(&mut rng, 5).unwrap();
    ///
    /// assert_eq!(assignments.len(), 5);
    /// // The first 4 are balanced (2 Treatment, 2 Control)
    /// let first_four = &assignments[0..4];
    /// let t_count = first_four.iter().filter(|&&g| g == Group::Treatment).count();
    /// assert_eq!(t_count, 2);
    /// ```
    #[verified_engine::verified]
    fn assign<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        n_subjects: usize,
    ) -> Result<Vec<Group>, ClinicalTrialError> {
        // Note: The original implementation didn't strictly check n_subjects % block_size,
        // it just filled enough blocks and truncated. We preserve that behavior.

        let num_blocks = (n_subjects as f64 / self.block_size as f64).ceil() as usize;
        let mut assignments = Vec::with_capacity(num_blocks * self.block_size);

        for _ in 0..num_blocks {
            let mut block = Vec::with_capacity(self.block_size);
            for _ in 0..(self.block_size / 2) {
                block.push(Group::Treatment);
                block.push(Group::Control);
            }
            block.shuffle(rng);
            assignments.extend(block);
        }

        assignments.truncate(n_subjects);
        Ok(assignments)
    }
}

/// Stratified Randomization Strategy.
///
/// Uses an underlying strategy (usually Block Randomization) within each stratum.
pub struct StratifiedRandomizer<S: AllocationStrategy> {
    strategy: S,
}

impl<S: AllocationStrategy> StratifiedRandomizer<S> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(strategy: S) -> Self {
        Self { strategy }
    }

    /// Assigns patients to groups based on their strata.
    #[verified_engine::verified]
    pub fn assign_stratified<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        patients: &[Patient],
    ) -> Result<HashMap<String, Group>, ClinicalTrialError> {
        let mut strata_map: HashMap<String, Vec<&Patient>> = HashMap::new();
        for p in patients {
            strata_map.entry(p.stratum.clone()).or_default().push(p);
        }

        let mut final_assignments = HashMap::new();

        for (_stratum, patients_in_stratum) in strata_map {
            let n = patients_in_stratum.len();
            let assignments = self.strategy.assign(rng, n)?;

            for (i, p) in patients_in_stratum.iter().enumerate() {
                final_assignments.insert(p.id.clone(), assignments[i]);
            }
        }

        Ok(final_assignments)
    }
}
