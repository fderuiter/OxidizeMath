//! # Algorithmic Information Theory (Kolmogorov Complexity)
//!
//! This module explores **Kolmogorov Complexity** (also known as Algorithmic Entropy), which
//! measures the amount of information in an object by the length of the shortest program
//! that can produce it.
//!
//! Unlike Shannon Entropy, which measures the average information of a random variable,
//! Kolmogorov Complexity looks at the *structure* of individual objects.
//!
//! ## Core Concepts
//!
//! 1. **Complexity $K(x)$**: The length of the shortest binary program $p$ such that $U(p) = x$
//!    (where $U$ is a universal Turing machine).
//! 2. **Incomputability**: $K(x)$ is not computable, but we can approximate it from above
//!    using compression algorithms or mathematical bounds.
//! 3. **Randomness**: A string is "random" if it cannot be compressed (i.e., $K(x) \approx |x|$).
//!
//! ```mermaid
//! graph TD
//!     Object[Data Object x]
//!     Prog[Program p]
//!     U[Universal Machine U]
//!
//!     Prog -->|Run| U
//!     U -->|Output| Object
//!
//!     style Prog fill:#f9f,stroke:#333
//!     style Object fill:#aaf,stroke:#333
//!
//!     subgraph Complexity
//!     Len[Length |p|]
//!     Min[Minimize]
//!     Len --> Min
//!     Min --> K[K(x)]
//!     end
//! ```
//!
//! ## Quick Start: Approximating Complexity
//!
//! We provide an upper bound approximation for the prefix Kolmogorov complexity of an integer $n$:
//! $$ K(n) \le \log_2(n) + 2 \log_2(\log_2(n)) + O(1) $$
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algorithmic_information::kolmogorov::kolmogorov_complexity;
//! use math_explorer::math_types::Integer;
//!
//! // Calculate complexity of a large number
//! let n = Integer::from(1000000);
//! let k = kolmogorov_complexity(&n);
//!
//! println!("Approximate Complexity K({:?}): {} bits", n, k);
//! assert_eq!(k, 30);
//! ```
//!
//! ## Modules
//!
//! * **[Combinatorics](combinatorics)**: Implementation of the Combinatorial Lemma for similarity sets.
//! * **[Geometry](geometry)**: Geometric primitives (Points, Lines, Dyadic Rationals) using exact rational arithmetic.
//! * **[Kolmogorov](kolmogorov)**: Prefix, `LZ76`, `BDM`, and dyadic point complexity upper bounds.
//! * **[FRACTRAN](fractran)**: Zero-allocation prime-exponent Universal Minsky Machines and cycle detection.

pub mod combinatorics;
pub mod fractran;
mod fractran_factor;
pub mod fractran_programs;
mod fractran_vm;
pub mod geometry;
pub mod kolmogorov;

pub use fractran::FractranError;

// [cite:algorithmic_information_rust]
