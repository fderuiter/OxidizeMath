//! # Group Theory
//!
//! This module provides implementations of common groups and tools for group theory analysis.
//!
//! It includes:
//! - **Cyclic Groups ($Z_n$)**: Implemented as both `CyclicElement` (dynamic modulus) and `Zn<N>` (static modulus).
//! - **Symmetric Groups ($S_n$)**: Implemented as `Permutation`.
//! - **Analysis Tools**: Functions to check for subgroups, normal subgroups, and generate cosets.

use crate::pure_math::algebra::traits::{AbelianGroup, Group, Monoid, Semigroup};
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
    /// Creates a new element in $Z_n$.
    #[verified_engine::verified]
    pub fn new(value: usize, modulo: usize) -> Self {
        CyclicElement {
            value: value % modulo,
            modulo,
        }
    }
}

impl Semigroup for CyclicElement {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        assert_eq!(
            a.modulo, b.modulo,
            "Cannot operate on elements from different groups"
        );
        CyclicElement {
            value: (a.value + b.value) % a.modulo,
            modulo: a.modulo,
        }
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
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: usize) -> Self {
        Zn { value: value % N }
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

impl<const N: usize> AbelianGroup for Zn<N> {}

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
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(map: Vec<usize>) -> Self {
        // Validation: map must contain 0..n
        let n = map.len();
        let mut seen = vec![false; n];
        for &x in &map {
            if x >= n || seen[x] {
                panic!("Invalid permutation");
            }
            seen[x] = true;
        }
        Permutation { map }
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
        // Function composition: (a * b)(x) = a(b(x))
        assert_eq!(
            a.map.len(),
            b.map.len(),
            "Permutations must be of same size"
        );
        let n = a.map.len();
        let mut new_map = vec![0; n];
        for (i, val) in new_map.iter_mut().enumerate().take(n) {
            *val = a.map[b.map[i]];
        }
        Permutation { map: new_map }
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
