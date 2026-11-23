//! # Number Theory
//!
//! This module provides algorithms for investigating the properties of integers.
//! It covers foundational areas such as Primality Testing, Partition Theory, and Q-Series.
//!
//! ## Structure
//!
//! ```mermaid
//! graph TD
//!     NT[Number Theory] --> Primes[Primes]
//!     NT --> Part[Partitions]
//!     NT --> Q[Q-Series]
//!     NT --> Class[Class Number]
//!
//!     Part -->|Uses| Q
//!
//!     style NT fill:#f9f,stroke:#333,stroke-width:2px
//! ```
//!
//! ## Submodules
//!
//! *   **[Primes](primes)**: Algorithms for generating prime numbers and primality testing.
//! *   **[Partitions](partitions)**: Restricted partition functions and their generating functions (q-series).
//! *   **[Q-Series](q_series)**: Algebraic manipulation of power series in $q$.
//! *   **[Class Number](class_number)**: Investigations into binary quadratic forms.
//! *   **[Hurwitz-Kronecker](hurwitz_kronecker)**: Class number relations.
//! *   **[ALCF](alcf)**: Algebraic-Lattice Cyclotomic Framework for quasiperfect numbers.

#[allow(missing_docs)]
pub mod alcf;
#[allow(missing_docs)]
pub mod ambs;
pub mod class_number;
pub mod hurwitz_kronecker;
pub mod partitions;
pub mod primes;
pub mod q_series;
#[allow(missing_docs)]
pub mod ualbf;

// Re-exports for convenience
pub use crate::error::NumberTheoryError;
pub use ambs::AmbsError;
pub use primes::{is_prime, primes_up_to};

// [cite:mmwave_radiotherapy_setup]
// [cite:hurwitz_kronecker]
