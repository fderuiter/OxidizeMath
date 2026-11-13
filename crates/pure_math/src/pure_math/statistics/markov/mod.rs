//! Markov Chains and Hidden Markov Models.
//!
//! This module provides comprehensive implementations of Markov processes:
//!
//! # Overview
//!
//! ## Discrete-Time Markov Chains (DTMC)
//!
//! A discrete-time Markov chain is a stochastic process with the Markov property:
//! the future state depends only on the current state, not on the past.
//!
//! **Key features:**
//! - Transient and absorbing state classification
//! - Canonical form decomposition
//! - Fundamental matrix N = (I - Q)⁻¹
//! - Expected Possession Value (EPV) calculations
//! - Stationary distributions
//! - Absorption probabilities and times
//!
//! ## Non-Stationary Chains (Time-Indexed Tensors)
//!
//! For processes where transition probabilities vary with time (e.g., shot clock urgency
//! in basketball), transition tensors provide time-indexed transition matrices with
//! interpolation support.
//!
//! **Key features:**
//! - Time-indexed transition matrices
//! - Linear interpolation between time slices
//! - Average transitions over time intervals
//!
//! ## Continuous-Time Markov Chains (CTMC)
//!
//! CTMCs model processes where transitions can occur at any continuous time point,
//! characterized by a generator (rate) matrix.
//!
//! **Key features:**
//! - Generator matrix validation
//! - Matrix exponential computation P(t) = exp(Gt)
//! - Steady-state distributions
//! - Trajectory simulation via Gillespie algorithm
//! - Expected absorption times
//!
//! ## Hidden Markov Models (HMM)
//!
//! HMMs model processes with hidden states and observable emissions, fundamental
//! to many applications in speech recognition, bioinformatics, and sports analytics.
//!
//! **Key features:**
//! - Forward algorithm (filtering)
//! - Backward algorithm
//! - Viterbi algorithm (most likely state sequence)
//! - Forward-Backward algorithm (posterior probabilities)
//! - Sequence generation
//!
//! # Mathematical Background
//!
//! ## Discrete-Time Markov Chains
//!
//! A DTMC {Xₙ} satisfies:
//! ```text
//! P(Xₙ₊₁ = j | X₀, X₁, ..., Xₙ) = P(Xₙ₊₁ = j | Xₙ) = Pᵢⱼ
//! ```
//!
//! For chains with absorbing states, the transition matrix in canonical form is:
//! ```text
//! P = [ Q  R ]
//!     [ 0  I ]
//! ```
//!
//! The fundamental matrix N = (I - Q)⁻¹ gives expected visit counts.
//!
//! ## Continuous-Time Markov Chains
//!
//! A CTMC is characterized by a generator matrix G where:
//! - Gᵢⱼ (i ≠ j): transition rate from i to j
//! - Gᵢᵢ = -Σⱼ≠ᵢ Gᵢⱼ
//!
//! The transition probabilities satisfy:
//! ```text
//! P'(t) = P(t)·G,  P(0) = I
//! ```
//!
//! Solution: P(t) = exp(Gt) = Σₖ₌₀^∞ (Gt)^k / k!
//!
//! ## Hidden Markov Models
//!
//! An HMM consists of:
//! - Hidden states with transition matrix A
//! - Observable symbols with emission matrix B
//! - Initial distribution π
//!
//! Key algorithms:
//! - **Forward**: P(observations) = Σᵢ α(T, i)
//! - **Viterbi**: argmax P(states | observations)
//! - **Forward-Backward**: P(Xₜ = i | observations) = α(t,i)·β(t,i) / P(obs)
//!
//! # Applications
//!
//! ## Basketball Analytics
//!
//! ### Expected Possession Value (EPV)
//! ```rust
//! use pure_math::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
//! use nalgebra::{DMatrix, DVector};
//!
//! // States: offense, advantage, score, turnover
//! let transitions = DMatrix::from_row_slice(4, 4, &[
//!     0.5, 0.3, 0.1, 0.1,  // Offense
//!     0.2, 0.4, 0.3, 0.1,  // Advantage
//!     0.0, 0.0, 1.0, 0.0,  // Score (absorbing)
//!     0.0, 0.0, 0.0, 1.0,  // Turnover (absorbing)
//! ]);
//!
//! let states = vec![
//!     StateType::Transient,
//!     StateType::Transient,
//!     StateType::Absorbing,
//!     StateType::Absorbing,
//! ];
//!
//! let chain = MarkovChain::new(transitions, states).unwrap();
//!
//! // Rewards: +2 for score, 0 for turnover
//! let rewards = DVector::from_vec(vec![2.0, 0.0]);
//! let epv = chain.expected_possession_value(&rewards).unwrap();
//!
//! println!("EPV from offense: {:.3}", epv[0]);
//! println!("EPV from advantage: {:.3}", epv[1]);
//! ```
//!
//! ### Shot Clock Urgency
//! ```rust
//! use pure_math::pure_math::statistics::markov::tensor::{TransitionTensor, TimeIndex};
//! use nalgebra::DMatrix;
//!
//! let mut tensor = TransitionTensor::new(
//!     3,
//!     TimeIndex::new(0.0).unwrap(),
//!     TimeIndex::new(24.0).unwrap()
//! );
//!
//! // Patient offense at full shot clock
//! let p_24 = DMatrix::from_row_slice(3, 3, &[
//!     0.85, 0.10, 0.05,
//!     0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0,
//! ]);
//! tensor.add_time_slice(TimeIndex::new(24.0).unwrap(), p_24).unwrap();
//!
//! // Urgent offense near expiration
//! let p_3 = DMatrix::from_row_slice(3, 3, &[
//!     0.30, 0.60, 0.10,
//!     0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0,
//! ]);
//! tensor.add_time_slice(TimeIndex::new(3.0).unwrap(), p_3).unwrap();
//!
//! // Query transition matrix at any time (interpolated)
//! let p_12 = tensor.transition_matrix_at(TimeIndex::new(12.0).unwrap()).unwrap();
//! ```
//!
//! ### Hot Hand Detection
//! ```rust
//! use pure_math::pure_math::statistics::markov::hmm::HiddenMarkovModel;
//! use nalgebra::{DMatrix, DVector};
//!
//! // Hidden states: Cold (0), Hot (1)
//! // Observations: Miss (0), Make (1)
//!
//! let initial = DVector::from_vec(vec![0.5, 0.5]);
//!
//! let transitions = DMatrix::from_row_slice(2, 2, &[
//!     0.7, 0.3,  // Cold → Cold/Hot
//!     0.4, 0.6,  // Hot → Cold/Hot
//! ]);
//!
//! let emissions = DMatrix::from_row_slice(2, 2, &[
//!     0.7, 0.3,  // Cold: 30% shooting
//!     0.2, 0.8,  // Hot: 80% shooting
//! ]);
//!
//! let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
//!
//! // Observed shooting: Make, Make, Make, Miss, Make
//! let shots = vec![1, 1, 1, 0, 1];
//!
//! // Most likely state sequence
//! let states = hmm.viterbi(&shots).unwrap();
//! println!("Inferred states: {:?}", states);
//!
//! // Current belief about being hot
//! let posterior = hmm.filter(&shots).unwrap();
//! println!("P(Hot | shots) = {:.3}", posterior[1]);
//! ```
//!
//! ## Finance
//!
//! ### Market Regime Detection
//! ```rust
//! use pure_math::pure_math::statistics::markov::hmm::HiddenMarkovModel;
//! use nalgebra::{DMatrix, DVector};
//!
//! // Hidden states: Bull (0), Bear (1), Sideways (2)
//! // Observations: Large Up (0), Small Up (1), Flat (2), Small Down (3), Large Down (4)
//!
//! let initial = DVector::from_vec(vec![0.33, 0.33, 0.34]);
//!
//! let transitions = DMatrix::from_row_slice(3, 3, &[
//!     0.7, 0.2, 0.1,  // Bull persistence
//!     0.2, 0.7, 0.1,  // Bear persistence
//!     0.2, 0.2, 0.6,  // Sideways
//! ]);
//!
//! // Bull: mostly up moves
//! // Bear: mostly down moves
//! // Sideways: centered distribution
//! let emissions = DMatrix::from_row_slice(3, 5, &[
//!     0.3, 0.3, 0.2, 0.1, 0.1,  // Bull
//!     0.1, 0.1, 0.2, 0.3, 0.3,  // Bear
//!     0.15, 0.2, 0.3, 0.2, 0.15, // Sideways
//! ]);
//!
//! let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
//! ```
//!
//! ## Queueing Theory
//!
//! ```rust
//! use pure_math::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
//! use nalgebra::DMatrix;
//!
//! // M/M/1 queue: arrival rate λ = 2, service rate μ = 3
//! // States: number of customers in system (0, 1, 2, 3, ...)
//! // Truncated to 5 states for example
//!
//! let lambda = 2.0;
//! let mu = 3.0;
//!
//! let generator = DMatrix::from_row_slice(5, 5, &[
//!     -lambda, lambda, 0.0, 0.0, 0.0,
//!     mu, -(lambda + mu), lambda, 0.0, 0.0,
//!     0.0, mu, -(lambda + mu), lambda, 0.0,
//!     0.0, 0.0, mu, -(lambda + mu), lambda,
//!     0.0, 0.0, 0.0, mu, -mu,
//! ]);
//!
//! let chain = ContinuousMarkovChain::new(generator).unwrap();
//!
//! // Steady-state distribution
//! if let Some(pi) = chain.steady_state() {
//!     println!("Steady-state probabilities: {:?}", pi);
//! }
//! ```
//!
//! # Implementation Notes
//!
//! ## Design Principles
//!
//! 1. **Type Safety**: Strong typing for probabilities, time indices, and states
//! 2. **Validation**: All matrices validated for stochasticity/generator properties
//! 3. **Numerical Stability**: Scaling in forward-backward algorithms prevents underflow
//! 4. **Determinism**: All stochastic operations accept explicit RNG for reproducibility
//!
//! ## Performance Considerations
//!
//! - Matrix operations use `nalgebra` with BLAS backend support
//! - N-step transitions use repeated squaring (O(log n) matrix multiplications)
//! - Matrix exponential uses Padé approximation with scaling and squaring
//! - Forward-Backward uses scaling factors to prevent numerical underflow
//!
//! ## Error Handling
//!
//! All operations return `Result<T, MarkovError>` with detailed error messages:
//! - `InvalidProbability`: Values outside [0, 1]
//! - `NotStochastic`: Rows don't sum to 1
//! - `InvalidGenerator`: Rows don't sum to 0 or invalid rates
//! - `DimensionMismatch`: Incompatible matrix/vector sizes
//! - `NumericalError`: Numerical computation failures
//! - `SingularMatrix`: Non-invertible matrices
//!
//! # References
//!
//! ## Books
//!
//! - Norris, J.R. (1997). *Markov Chains*. Cambridge University Press.
//! - Rabiner, L.R. (1989). "A tutorial on hidden Markov models and selected applications
//!   in speech recognition". *Proceedings of the IEEE*, 77(2), 257-286.
//! - Ross, S.M. (1996). *Stochastic Processes* (2nd ed.). Wiley.
//! - Stewart, W.J. (2009). *Probability, Markov Chains, Queues, and Simulation*.
//!   Princeton University Press.
//!
//! ## Papers
//!
//! - Viterbi, A.J. (1967). "Error bounds for convolutional codes and an asymptotically
//!   optimum decoding algorithm". *IEEE Transactions on Information Theory*, 13(2), 260-269.
//! - Baum, L.E., Petrie, T. (1966). "Statistical inference for probabilistic functions
//!   of finite state Markov chains". *The Annals of Mathematical Statistics*, 37(6), 1554-1563.
//! - Gillespie, D.T. (1977). "Exact stochastic simulation of coupled chemical reactions".
//!   *The Journal of Physical Chemistry*, 81(25), 2340-2361.
//!
//! # See Also
//!
//! - [`dtmc`]: Discrete-time Markov chains
//! - [`ctmc`]: Continuous-time Markov chains
//! - [`tensor`]: Time-indexed transition tensors
//! - [`hmm`]: Hidden Markov models
//! - [`MarkovError`]: Error types

pub mod ctmc;
pub mod dtmc;
#[allow(missing_docs)]
pub mod hmm;
pub mod tensor;
#[allow(missing_docs)]
pub mod time_index;
#[allow(missing_docs)]
pub mod validation;

// Re-export commonly used types
pub use crate::error::MarkovError;
pub use ctmc::ContinuousMarkovChain;
pub use dtmc::{MarkovChain, StateType};
#[allow(missing_docs)]
pub type Result<T> = std::result::Result<T, MarkovError>;
pub use hmm::HiddenMarkovModel;
pub use tensor::TransitionTensor;
pub use time_index::TimeIndex;

// [cite:clinical_trials_statistics]
