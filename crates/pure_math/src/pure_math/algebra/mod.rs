//! # Abstract Algebra
//!
//! This module provides a rigorous type hierarchy for algebraic structures,
//! ranging from basic Semigroups to Fields and Polynomial Rings.
//!
//! ##  Type Hierarchy
//!
//! The module is built around a set of traits defining algebraic properties:
//!
//! ```mermaid
//! classDiagram
//!     class Semigroup {
//!         <<Trait>>
//!         +operate(a, b)
//!     }
//!     class Monoid {
//!         <<Trait>>
//!         +identity()
//!     }
//!     class Group {
//!         <<Trait>>
//!         +inverse()
//!     }
//!     class Ring {
//!         <<Trait>>
//!         +zero()
//!         +one()
//!         +add()
//!         +mul()
//!     }
//!     class Field {
//!         <<Trait>>
//!         +multiplicative_inverse()
//!         +div()
//!     }
//!
//!     Semigroup <|-- Monoid
//!     Monoid <|-- Group
//!     Group <|-- Ring : (Additive Group)
//!     Monoid <|-- Ring : (Multiplicative Monoid)
//!     Ring <|-- Field
//! ```
//!
//! ##  Quick Start: Finite Fields & Polynomials
//!
//! Construct the Finite Field $\mathbb{F}_7$ and perform polynomial arithmetic over it.
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algebra::{Fp, Polynomial};
//!
//! // 1. Define elements in F_7 (Integers mod 7)
//! let a = Fp::<7>::new(3);
//! let b = Fp::<7>::new(5);
//!
//! // Arithmetic wraps modulo 7
//! assert_eq!(a + b, Fp::<7>::new(1)); // (3 + 5) % 7 = 1
//! assert_eq!(a * b, Fp::<7>::new(1)); // (3 * 5) % 7 = 15 % 7 = 1
//!
//! // 2. Create Polynomials over F_7
//! // P(x) = 3x^2 + 5. Coefficients are stored [const, x, x^2...]
//! let p1 = Polynomial::new(vec![b, Fp::<7>::new(0), a]);
//!
//! // Q(x) = 2x + 1
//! let q1 = Polynomial::new(vec![Fp::<7>::new(1), Fp::<7>::new(2)]);
//!
//! // 3. Polynomial Arithmetic
//! let sum = p1 + q1;
//! // Expected: 3x^2 + 2x + 6
//! // Result: [6, 2, 3]
//!
//! assert_eq!(sum.coeffs[0], Fp::<7>::new(6));
//! assert_eq!(sum.coeffs[1], Fp::<7>::new(2));
//! assert_eq!(sum.coeffs[2], Fp::<7>::new(3));
//! ```

pub mod fields;
pub mod group;
#[allow(missing_docs)]
pub mod linear_algebra;
pub mod polynomial;
#[allow(missing_docs)]
pub mod traits;

pub use fields::Fp;
pub use group::{CyclicElement, Permutation, Zn};
pub use linear_algebra::LinearAlgebraError;
pub use polynomial::Polynomial;
pub use traits::{EuclideanDomain, Field, Group, Monoid, Ring, Semigroup};

// [cite:advanced_linear_algebra]
