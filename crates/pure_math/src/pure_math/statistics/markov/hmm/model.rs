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
///
/// # Governing Joint Distribution & Verification
///
/// $$P(X_{1:T}, Y_{1:T}) = \pi_{X_1} B_{X_1, Y_1} \prod_{t=2}^{T} A_{X_{t-1}, X_t} B_{X_t, Y_t}$$
///
/// - **Citation**: `@rabiner1989tutorial` (`spec:pure_math_v1`).
/// - **Numerical stability**: Log-space Viterbi and scaled forward-backward recurrences prevent floating-point underflow on long observation horizons $T \gg 1$.
/// - **Domain of validity**: Requires $N \ge 1$ hidden states, $M \ge 1$ observation symbols, $\sum_i \pi_i = 1$, and row-stochastic matrices $A \in [0, 1]^{N \times N}$, $B \in [0, 1]^{N \times M}$.
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobservable (hidden) states.",
    citation = "rabiner1989tutorial"
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
    /// Creates a validated Hidden Markov Model $\lambda = (\pi, A, B)$ (`@rabiner1989tutorial`, `spec:pure_math_v1`).
    ///
    /// - **Numerical stability**: Validates row sums within floating-point tolerance before downstream forward/Viterbi inference.
    /// - **Domain of validity**: Requires `initial.len() == N >= 1`, `transitions` shape $(N, N)$, `emissions` shape $(N, M)$ with $M \ge 1$, and non-negative unit-sum probability rows.
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

    /// Returns hidden state cardinality $N = |S|$ (`@rabiner1989tutorial`, `spec:pure_math_v1`). Numerical stability: exact integer copy. Domain of validity: $N \ge 1$.
    #[verified_engine::verified]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Returns observation alphabet cardinality $M = |O|$ (`@rabiner1989tutorial`, `spec:pure_math_v1`). Numerical stability: exact integer copy. Domain of validity: $M \ge 1$.
    #[verified_engine::verified]
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Borrows initial state distribution $\pi \in \Delta^{N-1}$ (`@rabiner1989tutorial`, `spec:pure_math_v1`). Numerical stability: zero-cost borrow. Domain of validity: $\pi_i \in [0, 1], \sum_i \pi_i = 1$.
    #[verified_engine::verified]
    pub fn initial(&self) -> &DVector<T> {
        &self.initial
    }

    /// Borrows row-stochastic transition matrix $A \in [0, 1]^{N \times N}$ (`@rabiner1989tutorial`, `spec:pure_math_v1`). Numerical stability: zero-cost borrow. Domain of validity: $\sum_j A_{ij} = 1$ for all $1 \le i \le N$.
    #[verified_engine::verified]
    pub fn transitions(&self) -> &DMatrix<T> {
        &self.transitions
    }

    /// Borrows row-stochastic emission matrix $B \in [0, 1]^{N \times M}$ (`@rabiner1989tutorial`, `spec:pure_math_v1`). Numerical stability: zero-cost borrow. Domain of validity: $\sum_k B_{ik} = 1$ for all $1 \le i \le N$.
    #[verified_engine::verified]
    pub fn emissions(&self) -> &DMatrix<T> {
        &self.emissions
    }
}
