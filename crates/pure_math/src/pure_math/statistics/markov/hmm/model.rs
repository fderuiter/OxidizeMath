use crate::error::MarkovError;
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;
use verified_engine::Theory;

/// A Hidden Markov Model.
///
/// # Mathematical Background
///
/// A Hidden Markov Model consists of:
/// - **States**: A set of hidden states S = {s₁, ..., sₙ}
/// - **Observations**: A set of observable symbols O = {o₁, ..., oₘ}
/// - **π**: Initial state probabilities, πᵢ = P(X₁ = sᵢ)
/// - **A**: State transition probabilities, Aᵢⱼ = P(Xₜ₊₁ = sⱼ | Xₜ = sᵢ)
/// - **B**: Emission probabilities, Bᵢⱼ = P(Yₜ = oⱼ | Xₜ = sᵢ)
///
/// # Applications
///
/// - Speech recognition
/// - Part-of-speech tagging
/// - Bioinformatics (gene finding)
/// - Basketball: detecting "hot hand" states from shooting data
/// - Finance: regime detection in market states
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::markov::hmm::HiddenMarkovModel;
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Two hidden states: "Cold" (0) and "Hot" (1)
/// // Two observations: "Miss" (0) and "Make" (1)
///
/// let initial = DVector::from_vec(vec![0.5, 0.5]);
///
/// let transitions = DMatrix::from_row_slice(2, 2, &[
///     0.7, 0.3,  // Cold → Cold/Hot
///     0.4, 0.6,  // Hot → Cold/Hot
/// ]);
///
/// let emissions = DMatrix::from_row_slice(2, 2, &[
///     0.7, 0.3,  // Cold: P(Miss)=0.7, P(Make)=0.3
///     0.2, 0.8,  // Hot: P(Miss)=0.2, P(Make)=0.8
/// ]);
///
/// let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
///
/// // Observed sequence: Make, Make, Miss
/// let observations = vec![1, 0];
///
/// // Find most likely state sequence
/// let states = hmm.viterbi(&observations).unwrap();
/// println!("Most likely states: {:?}", states);
/// ```
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobservable (hidden) states.",
    citation = "A tutorial on hidden Markov models and selected applications in speech recognition (Rabiner, 1989)"
)]
pub struct HiddenMarkovModel<T: RealField + Copy + ToPrimitive> {
    /// Initial state probabilities π.
    pub(crate) initial: DVector<T>,
    /// State transition matrix A (num_states × num_states).
    pub(crate) transitions: DMatrix<T>,
    /// Emission matrix B (num_states × num_observations).
    pub(crate) emissions: DMatrix<T>,
    /// Number of hidden states.
    pub(crate) num_states: usize,
    /// Number of observable symbols.
    pub(crate) num_observations: usize,
}

impl<T: RealField + Copy + ToPrimitive> HiddenMarkovModel<T> {
    /// Creates a new Hidden Markov Model.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial state probabilities π (must sum to 1)
    /// * `transitions` - State transition matrix A (rows must sum to 1)
    /// * `emissions` - Emission matrix B (rows must sum to 1)
    ///
    /// # Returns
    ///
    /// A new `HiddenMarkovModel` or an error if validation fails.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If dimensions are inconsistent
    /// - `InvalidProbability`: If probabilities are invalid
    /// - `NotStochastic`: If matrices are not stochastic
    #[verified_engine::verified]
    pub fn new(
        initial: DVector<T>,
        transitions: DMatrix<T>,
        emissions: DMatrix<T>,
    ) -> Result<Self> {
        let num_states = initial.len();

        // Validate dimensions
        if transitions.nrows() != num_states || transitions.ncols() != num_states {
            return Err(crate::error::MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(num_states),
                    actual: math_commons::math_kernel::types::Dimension(transitions.nrows()),
                },
            ));
        }

        if emissions.nrows() != num_states {
            return Err(crate::error::MarkovError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(num_states),
                    actual: math_commons::math_kernel::types::Dimension(emissions.nrows()),
                },
            ));
        }

        let num_observations = emissions.ncols();

        // Validate initial probabilities
        crate::pure_math::statistics::markov::validation::validate_probability_vector(&initial)?;

        // Validate transition matrix
        crate::pure_math::statistics::markov::validation::validate_stochastic_matrix(&transitions)?;

        // Validate emission matrix
        crate::pure_math::statistics::markov::validation::validate_stochastic_matrix(&emissions)?;

        Ok(HiddenMarkovModel {
            initial,
            transitions,
            emissions,
            num_states,
            num_observations,
        })
    }

    /// Returns the number of hidden states.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Returns the number of observable symbols.
    #[verified_engine::verified]
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Returns the initial state probabilities.
    #[verified_engine::verified]
    pub fn initial(&self) -> &DVector<T> {
        &self.initial
    }

    /// Returns the transition matrix.
    #[verified_engine::verified]
    pub fn transitions(&self) -> &DMatrix<T> {
        &self.transitions
    }

    /// Returns the emission matrix.
    #[verified_engine::verified]
    pub fn emissions(&self) -> &DMatrix<T> {
        &self.emissions
    }
}
