//! # Group Theory
//!
//! This module provides implementations of common groups and tools for group theory analysis.
//!
//! It includes:
//! - **Cyclic Groups ($Z_n$)**: Implemented as both `CyclicElement` (dynamic modulus) and `Zn<N>` (static modulus).
//! - **Symmetric Groups ($S_n$)**: Implemented as `Permutation`.
//! - **Analysis Tools**: Functions to check for subgroups, normal subgroups, and generate cosets.

use crate::error::NumberTheoryError;
use crate::pure_math::algebra::traits::{Group, Monoid, Semigroup};
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use std::collections::HashSet;
use std::hash::Hash;

// ============================================================================
// Cyclic Group (Integers Modulo n)
// ============================================================================

/// An element of the Cyclic Group $Z_n$ under addition.
///
/// This struct allows for runtime definition of the modulus.
///
/// # Warning: Monoid Identity
/// The `Monoid` trait requires a static `identity()` method. Since `CyclicElement`'s modulus is defined at runtime,
/// `Monoid::identity()` cannot be correctly implemented (it panics).
/// Use `Zn<N>` for a type-safe implementation where N is known at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CyclicElement {
    #[allow(missing_docs)]
    pub value: usize,
    #[allow(missing_docs)]
    pub modulo: usize,
}

impl CyclicElement {
    /// Creates a new element in $Z_n$ after validating that `modulo > 0`.
    ///
    /// # Errors
    ///
    /// Returns [`NumberTheoryError::ZeroModulus`] when `modulo == 0`.
    #[verified_engine::verified]
    pub fn try_new(value: usize, modulo: usize) -> Result<Self, NumberTheoryError> {
        if modulo == 0 {
            return Err(NumberTheoryError::ZeroModulus);
        }
        Ok(CyclicElement {
            value: value % modulo,
            modulo,
        })
    }

    /// Combines two cyclic elements if they share the same non-zero modulus.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::DimensionMismatch`] when `a.modulo != b.modulo` or
    /// [`MathError::StrictlyOutOfBounds`] when `a.modulo == 0`.
    #[verified_engine::verified]
    pub fn try_operate(a: &Self, b: &Self) -> Result<Self, MathError> {
        if a.modulo != b.modulo {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(a.modulo),
                actual: Dimension(b.modulo),
            });
        }
        if a.modulo == 0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(CyclicElement {
            value: (a.value + b.value) % a.modulo,
            modulo: a.modulo,
        })
    }

    /// Creates a new element in $Z_n$.
    ///
    /// # Panics
    ///
    /// Panics if `modulo == 0`. Prefer [`Self::try_new`] for untrusted moduli.
    #[verified_engine::verified]
    pub fn new(value: usize, modulo: usize) -> Self {
        Self::try_new(value, modulo).expect("CyclicElement modulo must be non-zero")
    }
}

impl Semigroup for CyclicElement {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        Self::try_operate(a, b).expect("Cannot operate on elements from different groups")
    }
}

impl Monoid for CyclicElement {
    /// **Panics:** Use `Zn<N>` if you need `Monoid::identity`.
    #[verified_engine::verified]
    fn identity() -> Self {
        panic!(
            "Monoid::identity() is not supported for CyclicElement because the modulus is dynamic. \
             Use Zn<N> or construct elements explicitly."
        )
    }
}

/// An element of the Cyclic Group $Z_N$ where N is a compile-time constant.
///
/// This implementation fully satisfies `Semigroup`, `Monoid`, and `Group`.
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::{Zn, Group, Semigroup};
///
/// // Work in Z_5
/// let a = Zn::<5>::new(3);
/// let b = Zn::<5>::new(4);
///
/// // (3 + 4) mod 5 = 7 mod 5 = 2
/// let sum = Zn::<5>::operate(&a, &b);
/// assert_eq!(sum.value, 2);
///
/// // Inverse of 3 in Z_5 is 2 (because 3 + 2 = 5 = 0)
/// let inv = a.inverse();
/// assert_eq!(inv.value, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Zn<const N: usize> {
    #[allow(missing_docs)]
    pub value: usize,
}

impl<const N: usize> Zn<N> {
    /// Creates a new element in $Z_N$ after verifying `N > 0`.
    ///
    /// # Errors
    ///
    /// Returns [`NumberTheoryError::ZeroModulus`] if `N == 0`.
    #[verified_engine::verified]
    pub fn try_new(value: usize) -> Result<Self, NumberTheoryError> {
        if N == 0 {
            return Err(NumberTheoryError::ZeroModulus);
        }
        Ok(Zn { value: value % N })
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: usize) -> Self {
        Self::try_new(value).expect("Zn<N> modulus N must be non-zero")
    }
}

impl<const N: usize> Semigroup for Zn<N> {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        Zn {
            value: (a.value + b.value) % N,
        }
    }
}

impl<const N: usize> Monoid for Zn<N> {
    #[verified_engine::verified]
    fn identity() -> Self {
        Zn { value: 0 }
    }
}

impl<const N: usize> Group for Zn<N> {
    #[verified_engine::verified]
    fn inverse(&self) -> Self {
        if self.value == 0 {
            Zn { value: 0 }
        } else {
            Zn {
                value: N - self.value,
            }
        }
    }
}

// ============================================================================
// Symmetric Group (Permutations)
// ============================================================================

/// An element of the Symmetric Group $S_n$ (Permutations).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation {
    #[allow(missing_docs)]
    pub map: Vec<usize>, // map[i] is the image of i
}

impl Permutation {
    /// Constructs a validated permutation of `0..map.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidPermutation`] if any entry is `>= map.len()` or duplicated.
    #[verified_engine::verified]
    pub fn try_new(map: Vec<usize>) -> Result<Self, MathError> {
        let n = map.len();
        let mut seen = vec![false; n];
        for (idx, &x) in map.iter().enumerate() {
            if x >= n || seen[x] {
                return Err(MathError::InvalidPermutation {
                    index: idx,
                    value: x,
                    len: n,
                });
            }
            seen[x] = true;
        }
        Ok(Permutation { map })
    }

    /// Composes two permutations `(a * b)(i) = a(b(i))` when their domain sizes match.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::DimensionMismatch`] when `a.map.len() != b.map.len()`.
    #[verified_engine::verified]
    pub fn try_operate(a: &Self, b: &Self) -> Result<Self, MathError> {
        if a.map.len() != b.map.len() {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(a.map.len()),
                actual: Dimension(b.map.len()),
            });
        }
        let n = a.map.len();
        let mut new_map = vec![0; n];
        for (i, val) in new_map.iter_mut().enumerate().take(n) {
            *val = a.map[b.map[i]];
        }
        Ok(Permutation { map: new_map })
    }

    /// Constructs a permutation of `0..map.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `map` is not a bijection on `0..map.len()`. Prefer [`Self::try_new`].
    #[verified_engine::verified]
    pub fn new(map: Vec<usize>) -> Self {
        match Self::try_new(map) {
            Ok(perm) => perm,
            Err(_) => panic!("Invalid permutation"),
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn identity(n: usize) -> Self {
        Permutation {
            map: (0..n).collect(),
        }
    }
}

impl Semigroup for Permutation {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        Self::try_operate(a, b).expect("Permutations must be of same size")
    }
}

// ============================================================================
// Group Analysis Tools
// ============================================================================

/// Checks if a subset H is a subgroup of G.
/// Requires G to be finite and represented by a list of all elements, or we just check closure/inverse for H.
/// Here we check if H is closed under operation and inverses.
#[verified_engine::verified]
pub fn is_subgroup<G: Group + Eq + Hash>(subgroup: &[G]) -> bool {
    if subgroup.is_empty() {
        return false;
    }

    // Check identity presence
    // Actually, simply checking closure under ab^-1 (One-Step Subgroup Test) is enough for nonempty subsets.

    // We need to know if elements are in subgroup.
    let set: HashSet<&G> = subgroup.iter().collect();

    for a in subgroup {
        for b in subgroup {
            // Check a * b^-1 in H
            let b_inv = b.inverse();
            let product = G::operate(a, &b_inv);
            if !set.contains(&product) {
                return false;
            }
        }
    }
    true
}

/// Generates a right coset Ha = { h * a : h in H }.
#[verified_engine::verified]
pub fn generate_right_coset<G: Group>(subgroup: &[G], a: &G) -> Vec<G> {
    subgroup.iter().map(|h| G::operate(h, a)).collect()
}

/// Generates a left coset aH = { a * h : h in H }.
#[verified_engine::verified]
pub fn generate_left_coset<G: Group>(subgroup: &[G], a: &G) -> Vec<G> {
    subgroup.iter().map(|h| G::operate(a, h)).collect()
}

/// Checks if a subgroup is normal.
/// A subgroup N is normal if aN = Na for all a in G.
/// We need the full group G for this check.
#[verified_engine::verified]
pub fn is_normal_subgroup<G: Group + Eq + Hash>(group: &[G], subgroup: &[G]) -> bool {
    if !is_subgroup(subgroup) {
        return false;
    }

    // Convert subgroup to Set for fast lookup
    let sub_set: HashSet<&G> = subgroup.iter().collect();

    for a in group {
        // Check aNa^-1 <= N
        let a_inv = a.inverse();
        for n in subgroup {
            let conj = G::operate(&G::operate(a, n), &a_inv);
            if !sub_set.contains(&conj) {
                return false;
            }
        }
    }
    true
}
