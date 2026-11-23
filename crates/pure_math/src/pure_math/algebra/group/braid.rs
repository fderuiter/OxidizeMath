//! Infinite Artin braid groups $B_n$, word reductions, braid relations, and 3D Alexander closures.

// [cite:knot_braid_topology]

use crate::pure_math::algebra::group::Permutation;
use thiserror::Error;

/// Maximum interactive strand count supported by [`BraidWord`] (`1..=16`).
pub const MAX_BRAID_STRANDS: usize = 16;

/// Maximum generator length permitted in a bounded [`BraidWord`].
pub const MAX_BRAID_WORD_LEN: usize = 128;

/// Validation and relation errors for Artin [`BraidWord`] operations.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum BraidError {
    /// Strand count must lie in `1..=MAX_BRAID_STRANDS`.
    #[error("braid strand count must lie in 1..={MAX_BRAID_STRANDS} (got {strands})")]
    InvalidStrandCount {
        /// Requested strand count.
        strands: usize,
    },
    /// Signed generator index must be non-zero and satisfy `1 <= |g| < strands`.
    #[error("invalid Artin generator {generator} for {strands}-strand braid")]
    InvalidGenerator {
        /// Invalid signed generator (`+i` for `sigma_i`, `-i` for `sigma_i^-1`).
        generator: i32,
        /// Active strand count.
        strands: usize,
    },
    /// Composed braid words must share a common strand count.
    #[error("cannot compose {left}-strand braid with {right}-strand braid")]
    StrandCountMismatch {
        /// Left operand strand count.
        left: usize,
        /// Right operand strand count.
        right: usize,
    },
    /// Word length exceeded [`MAX_BRAID_WORD_LEN`].
    #[error("braid word length {len} exceeds maximum allowed limit {MAX_BRAID_WORD_LEN}")]
    WordTooLong {
        /// Attempted generator count.
        len: usize,
    },
    /// Requested local Artin relation does not apply at `index`.
    #[error("braid relation is not applicable at word index {index}")]
    RelationNotApplicable {
        /// Word position inspected.
        index: usize,
    },
}

/// An element of the infinite Artin braid group $B_n$ represented as a word in
/// signed generators $\sigma_1^{\pm 1}, \dots, \sigma_{n-1}^{\pm 1}$.
///
/// Generator `+i` denotes the positive right-handed crossing $\sigma_i$ where strand `i - 1`
/// passes over strand `i`, and `-i` denotes its inverse $\sigma_i^{-1}$.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::BraidWord;
///
/// let trefoil = BraidWord::new(2, vec![1, 1, 1]).unwrap();
/// assert_eq!(trefoil.writhe(), 3);
/// assert_eq!(trefoil.component_count(), 1);
/// assert!(trefoil.compose(&trefoil.inverse()).unwrap().free_reduce().is_empty());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BraidWord {
    strands: usize,
    generators: Vec<i32>,
}

impl BraidWord {
    /// Constructs and validates an Artin braid word on `strands` strands.
    ///
    /// # Errors
    /// Returns [`BraidError`] if `strands` is outside `1..=MAX_BRAID_STRANDS`, if the word
    /// exceeds [`MAX_BRAID_WORD_LEN`], or if any generator fails `1 <= |g| < strands`.
    #[verified_engine::verified]
    pub fn new(strands: usize, generators: Vec<i32>) -> Result<Self, BraidError> {
        if !(1..=MAX_BRAID_STRANDS).contains(&strands) {
            return Err(BraidError::InvalidStrandCount { strands });
        }
        if generators.len() > MAX_BRAID_WORD_LEN {
            return Err(BraidError::WordTooLong {
                len: generators.len(),
            });
        }
        for &g in &generators {
            let idx = g.unsigned_abs() as usize;
            if idx == 0 || idx >= strands {
                return Err(BraidError::InvalidGenerator {
                    generator: g,
                    strands,
                });
            }
        }
        Ok(Self {
            strands,
            generators,
        })
    }

    /// Constructs the trivial identity braid on `strands` strands.
    ///
    /// # Errors
    /// Returns [`BraidError::InvalidStrandCount`] if `strands` is outside `1..=MAX_BRAID_STRANDS`.
    #[verified_engine::verified]
    pub fn identity(strands: usize) -> Result<Self, BraidError> {
        Self::new(strands, Vec::new())
    }

    /// Number of strands $n$ in $B_n$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn strands(&self) -> usize {
        self.strands
    }

    /// Slice of signed Artin generators (`+i` for $\sigma_i$, `-i` for $\sigma_i^{-1}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn generators(&self) -> &[i32] {
        &self.generators
    }

    /// Word length (number of crossings before closure).
    #[must_use]
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// Returns `true` when the braid word has zero generators.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// Appends a validated signed generator $\sigma_i^{\pm 1}$ to the word.
    ///
    /// # Errors
    /// Returns [`BraidError`] if `generator` is invalid for `self.strands` or exceeds length cap.
    #[verified_engine::verified]
    pub fn push_generator(&mut self, generator: i32) -> Result<(), BraidError> {
        if self.generators.len() >= MAX_BRAID_WORD_LEN {
            return Err(BraidError::WordTooLong {
                len: self.generators.len() + 1,
            });
        }
        let idx = generator.unsigned_abs() as usize;
        if idx == 0 || idx >= self.strands {
            return Err(BraidError::InvalidGenerator {
                generator,
                strands: self.strands,
            });
        }
        self.generators.push(generator);
        Ok(())
    }

    /// Concatenates `self` followed by `other` in $B_n$.
    ///
    /// # Errors
    /// Returns [`BraidError`] if strand counts differ or the combined length exceeds [`MAX_BRAID_WORD_LEN`].
    #[verified_engine::verified]
    pub fn compose(&self, other: &Self) -> Result<Self, BraidError> {
        if self.strands != other.strands {
            return Err(BraidError::StrandCountMismatch {
                left: self.strands,
                right: other.strands,
            });
        }
        let total = self.generators.len() + other.generators.len();
        if total > MAX_BRAID_WORD_LEN {
            return Err(BraidError::WordTooLong { len: total });
        }
        let mut combined = Vec::with_capacity(total);
        combined.extend_from_slice(&self.generators);
        combined.extend_from_slice(&other.generators);
        Ok(Self {
            strands: self.strands,
            generators: combined,
        })
    }

    /// Returns the group inverse $\beta^{-1}$ obtained by reversing and negating all generators.
    #[must_use]
    #[verified_engine::verified]
    pub fn inverse(&self) -> Self {
        Self {
            strands: self.strands,
            generators: self.generators.iter().rev().map(|&g| -g).collect(),
        }
    }

    /// Returns the mirror braid obtained by flipping every crossing sign ($\sigma_i \leftrightarrow \sigma_i^{-1}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn mirror(&self) -> Self {
        Self {
            strands: self.strands,
            generators: self.generators.iter().map(|&g| -g).collect(),
        }
    }

    /// Applies positive or negative Markov stabilization $\beta \mapsto \beta \sigma_n^{\pm 1} \in B_{n+1}$.
    ///
    /// # Errors
    /// Returns [`BraidError`] if `strands + 1` or `len + 1` exceeds bounds.
    #[verified_engine::verified]
    pub fn markov_stabilize(&self, positive: bool) -> Result<Self, BraidError> {
        let next_strands = self.strands + 1;
        let mut gens = self.generators.clone();
        let g = self.strands as i32;
        gens.push(if positive { g } else { -g });
        Self::new(next_strands, gens)
    }

    /// Cancels all adjacent inverse pairs $\sigma_i \sigma_i^{-1}$ and $\sigma_i^{-1} \sigma_i$ in place,
    /// returning the number of cancelled generator pairs.
    #[verified_engine::verified]
    pub fn free_reduce_in_place(&mut self) -> usize {
        let orig_len = self.generators.len();
        let mut stack = Vec::with_capacity(orig_len);
        for &g in &self.generators {
            if stack.last().is_some_and(|&top| top == -g) {
                stack.pop();
            } else {
                stack.push(g);
            }
        }
        self.generators = stack;
        (orig_len - self.generators.len()) / 2
    }

    /// Returns a freely reduced copy of `self`.
    #[must_use]
    #[verified_engine::verified]
    pub fn free_reduce(&self) -> Self {
        let mut copy = self.clone();
        copy.free_reduce_in_place();
        copy
    }

    /// Returns `true` if far commutativity $\sigma_i^{\pm 1} \sigma_j^{\pm 1} = \sigma_j^{\pm 1} \sigma_i^{\pm 1}$
    /// (`|i - j| >= 2`) applies at `(index, index + 1)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn can_apply_far_commutativity(&self, index: usize) -> bool {
        if index + 1 >= self.generators.len() {
            return false;
        }
        let i = self.generators[index].unsigned_abs();
        let j = self.generators[index + 1].unsigned_abs();
        i.abs_diff(j) >= 2
    }

    /// Swaps commuting non-adjacent generators at `(index, index + 1)`.
    ///
    /// # Errors
    /// Returns [`BraidError::RelationNotApplicable`] if `|i - j| < 2` or out of range.
    #[verified_engine::verified]
    pub fn apply_far_commutativity(&mut self, index: usize) -> Result<(), BraidError> {
        if !self.can_apply_far_commutativity(index) {
            return Err(BraidError::RelationNotApplicable { index });
        }
        self.generators.swap(index, index + 1);
        Ok(())
    }

    /// Returns `true` if the Artin Yang-Baxter braid relation
    /// $\sigma_i^{\epsilon} \sigma_j^{\epsilon} \sigma_i^{\epsilon} \leftrightarrow \sigma_j^{\epsilon} \sigma_i^{\epsilon} \sigma_j^{\epsilon}$
    /// (`|i - j| == 1`, $\epsilon \in \{\pm 1\}$) applies at `index..index + 3`.
    #[must_use]
    #[verified_engine::verified]
    pub fn can_apply_artin_relation(&self, index: usize) -> bool {
        if index + 2 >= self.generators.len() {
            return false;
        }
        let (g0, g1, g2) = (
            self.generators[index],
            self.generators[index + 1],
            self.generators[index + 2],
        );
        g0 == g2 && g0.signum() == g1.signum() && g0.unsigned_abs().abs_diff(g1.unsigned_abs()) == 1
    }

    /// Rewrites $\sigma_i^{\epsilon} \sigma_j^{\epsilon} \sigma_i^{\epsilon} \mapsto \sigma_j^{\epsilon} \sigma_i^{\epsilon} \sigma_j^{\epsilon}$
    /// in place at `index..index + 3`.
    ///
    /// # Errors
    /// Returns [`BraidError::RelationNotApplicable`] if the triple does not match the Artin relation.
    #[verified_engine::verified]
    pub fn apply_artin_relation(&mut self, index: usize) -> Result<(), BraidError> {
        if !self.can_apply_artin_relation(index) {
            return Err(BraidError::RelationNotApplicable { index });
        }
        let g0 = self.generators[index];
        let g1 = self.generators[index + 1];
        self.generators[index] = g1;
        self.generators[index + 1] = g0;
        self.generators[index + 2] = g1;
        Ok(())
    }

    /// Algebraic writhe (exponent sum $\sum_k \operatorname{sgn}(g_k)$).
    #[must_use]
    #[verified_engine::verified]
    pub fn writhe(&self) -> i64 {
        self.generators.iter().map(|&g| i64::from(g.signum())).sum()
    }

    /// Induced strand permutation in $S_n$ mapping initial bottom slot `s` to final top slot.
    #[must_use]
    #[verified_engine::verified]
    pub fn strand_permutation(&self) -> Permutation {
        let mut slot_of_strand: Vec<usize> = (0..self.strands).collect();
        let mut strand_at_slot: Vec<usize> = (0..self.strands).collect();
        for &g in &self.generators {
            let left = (g.unsigned_abs() as usize) - 1;
            let right = left + 1;
            let s_left = strand_at_slot[left];
            let s_right = strand_at_slot[right];
            strand_at_slot.swap(left, right);
            slot_of_strand[s_left] = right;
            slot_of_strand[s_right] = left;
        }
        Permutation::new(slot_of_strand)
    }

    /// Number of closed link components in the Alexander closure $\hat{\beta}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn component_count(&self) -> usize {
        let perm = self.strand_permutation();
        let mut visited = vec![false; self.strands];
        let mut cycles = 0;
        for start in 0..self.strands {
            if visited[start] {
                continue;
            }
            cycles += 1;
            let mut cur = start;
            let mut guard = 0;
            while !visited[cur] && guard <= self.strands {
                visited[cur] = true;
                cur = perm.map[cur];
                guard += 1;
            }
        }
        cycles
    }
}

mod braid_closure;
