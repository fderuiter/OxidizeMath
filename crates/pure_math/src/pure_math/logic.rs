//! # Logic and Proofs
//!
//! This module provides computational tools to illustrate concepts from mathematical logic,
//! including predicate calculus (quantifiers) and induction.
//!
//! ## Predicate Calculus
//!
//! - **Universal Quantifier ($\forall$):** Checks if a predicate holds for all elements in a domain.
//! - **Existential Quantifier ($\exists$):** Checks if there is at least one element in the domain for which the predicate holds.
//!
//! ## Induction
//!
//! While rigorous proofs are symbolic, we can computationally verify inductive steps over a finite range.

/// Universal Quantifier ($\forall$).
///
/// Returns `true` if the predicate `p` returns `true` for every element in the iterator `domain`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::logic::forall;
///
/// let domain = vec![2, 4, 6, 8];
/// assert!(forall(domain.into_iter(), |x| x % 2 == 0));
/// ```
#[verified_engine::verified]
pub fn forall<I, F>(domain: I, p: F) -> bool
where
    I: Iterator,
    F: Fn(I::Item) -> bool,
{
    for item in domain {
        if !p(item) {
            return false;
        }
    }
    true
}

/// Existential Quantifier ($\exists$).
///
/// Returns `true` if the predicate `p` returns `true` for at least one element in the iterator `domain`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::logic::exists;
///
/// let domain = vec![1, 3, 5, 8];
/// assert!(exists(domain.into_iter(), |x| x % 2 == 0)); // 8 is even
/// ```
#[verified_engine::verified]
pub fn exists<I, F>(domain: I, p: F) -> bool
where
    I: Iterator,
    F: Fn(I::Item) -> bool,
{
    for item in domain {
        if p(item) {
            return true;
        }
    }
    false
}

/// Negation of a Universal Quantifier.
///
/// Demonstrates the logical equivalence: $\neg [\forall x, P(x)] \iff \exists x, \neg P(x)$.
#[verified_engine::verified]
pub fn negate_forall<I, F>(domain: I, p: F) -> bool
where
    I: Iterator + Clone,
    F: Fn(I::Item) -> bool,
{
    // The negation of "for all x, P(x)" is "exists x such that not P(x)"
    exists(domain, |x| !p(x))
}

/// Negation of an Existential Quantifier.
///
/// Demonstrates the logical equivalence: $\neg [\exists x, P(x)] \iff \forall x, \neg P(x)$.
#[verified_engine::verified]
pub fn negate_exists<I, F>(domain: I, p: F) -> bool
where
    I: Iterator + Clone,
    F: Fn(I::Item) -> bool,
{
    // The negation of "exists x such that P(x)" is "for all x, not P(x)"
    forall(domain, |x| !p(x))
}

/// A helper to demonstrate mathematical induction computationally.
///
/// Checks if a property `P(n)` holds for a range `1..=limit`.
/// It verifies the base case `P(1)` and then checks the inductive step `P(k) -> P(k+1)` implicitly
/// by verifying `P(n)` for all n.
///
/// Note: This is not a formal proof, but a computational verification.
#[verified_engine::verified]
pub fn verify_induction<F>(limit: usize, p: F) -> bool
where
    F: Fn(usize) -> bool,
{
    // Basis Step
    if !p(1) {
        return false;
    }

    // Inductive Step verification (up to limit)
    for k in 1..limit {
        // We assume P(k) is true because we passed it in the previous iteration.
        // We just need to check if P(k+1) is also true.
        if !p(k + 1) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_quantifiers() {
        let evens = vec![2, 4, 6, 8];
        assert!(forall(evens.clone().into_iter(), |x| x % 2 == 0));
        assert!(!exists(evens.into_iter(), |x| x % 2 != 0));

        let mixed = vec![1, 2, 3];
        assert!(!forall(mixed.clone().into_iter(), |x| x % 2 == 0));
        assert!(exists(mixed.into_iter(), |x| x % 2 == 0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_negations() {
        let evens = vec![2, 4, 6];
        // "Not all are even" should be false
        assert!(!negate_forall(evens.clone().into_iter(), |x| x % 2 == 0));

        let mixed = vec![1, 2, 3];
        // "Not all are even" should be true because there exists an odd number
        assert!(negate_forall(mixed.clone().into_iter(), |x| x % 2 == 0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_induction_sum() {
        // Prove sum(1..n) = n(n+1)/2
        let prop = |n: usize| {
            let sum: usize = (1..=n).sum();
            sum == n * (n + 1) / 2
        };
        assert!(verify_induction(100, prop));
    }
}
