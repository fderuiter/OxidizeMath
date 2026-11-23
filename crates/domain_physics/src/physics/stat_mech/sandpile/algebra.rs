//! Finite Abelian Sandpile Group (`SandpileElement<W, H>`).
//!
//! Connects the recurrent configurations of the 2D Bak-Tang-Wiesenfeld sandpile on an
//! open-boundary $W \times H$ lattice directly to [`pure_math::pure_math::algebra`]
//! (`Semigroup`, `Monoid`, `Group`, `AbelianGroup`, `Zn<N>`, `is_subgroup`, and
//! `generate_left_coset`).

use pure_math::pure_math::algebra::{AbelianGroup, Group, Monoid, Semigroup, Zn};
use serde::{Deserialize, Serialize};

use super::kirchhoff::{
    SandpileError, creutz_dual_zero_array, creutz_identity_array, dhar_burning_array,
    kirchhoff_determinant_bareiss, stabilize_grid_array,
};

/// An element of the finite Abelian sandpile group $G \cong \mathbb{Z}^{W \cdot H} / \Delta \mathbb{Z}^{W \cdot H}$
/// on an open-boundary $W \times H$ lattice.
///
/// Group addition is defined by cellwise addition followed by Abelian toppling stabilization:
/// $(a \oplus b) = (a + b)^\circ$. The identity element is Creutz's recurrent fractal configuration
/// $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$, satisfying $(e + e)^\circ = e$ and $(r + e)^\circ = r$
/// for every recurrent configuration $r$.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::sandpile::SandpileElement;
/// use math_explorer::pure_math::algebra::{Group, Monoid, Semigroup};
///
/// let e = SandpileElement::<3, 3>::identity();
/// assert!(e.is_recurrent());
/// assert_eq!(SandpileElement::operate(&e, &e), e);
///
/// let r = SandpileElement::<3, 3>::new([[3, 2, 3], [2, 1, 2], [3, 2, 3]]);
/// assert_eq!(SandpileElement::operate(&r, &e), r);
/// assert_eq!(SandpileElement::operate(&r, &r.inverse()), e);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SandpileElement<const W: usize, const H: usize> {
    heights: [[u32; W]; H],
}

impl<const W: usize, const H: usize> Serialize for SandpileElement<W, H> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let flat: Vec<u32> = self
            .heights
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        flat.serialize(serializer)
    }
}

impl<'de, const W: usize, const H: usize> Deserialize<'de> for SandpileElement<W, H> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let flat = Vec::<u32>::deserialize(deserializer)?;
        if flat.len() != W * H {
            return Err(serde::de::Error::invalid_length(
                flat.len(),
                &"W * H sandpile heights",
            ));
        }
        let mut heights = [[0_u32; W]; H];
        for (idx, val) in flat.into_iter().enumerate() {
            heights[idx / W][idx % W] = val;
        }
        Ok(Self::from_stabilized_raw(heights))
    }
}

impl<const W: usize, const H: usize> SandpileElement<W, H> {
    /// Projects an arbitrary non-negative height array onto its unique recurrent group
    /// representative $(z + e)^\circ \in G$.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(mut heights: [[u32; W]; H]) -> Self {
        let _ = stabilize_grid_array(&mut heights);
        let e = creutz_identity_array::<W, H>();
        for (row, e_row) in heights.iter_mut().zip(e.iter()) {
            for (cell, &e_val) in row.iter_mut().zip(e_row.iter()) {
                *cell += e_val;
            }
        }
        let _ = stabilize_grid_array(&mut heights);
        Self { heights }
    }

    /// Stabilizes `heights` directly via toppling without adding the recurrent identity $e$.
    ///
    /// Useful when inspecting transient configurations (such as the all-zero state) alongside
    /// recurrent group elements via [`Self::is_recurrent`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_stabilized_raw(mut heights: [[u32; W]; H]) -> Self {
        let _ = stabilize_grid_array(&mut heights);
        Self { heights }
    }

    /// Returns the Creutz recurrent identity element $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$.
    #[must_use]
    #[verified_engine::verified]
    pub fn identity_element() -> Self {
        Self {
            heights: creutz_identity_array::<W, H>(),
        }
    }

    /// Borrows the underlying stabilized $H \times W$ height array (`heights[y][x]`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn heights(&self) -> &[[u32; W]; H] {
        &self.heights
    }

    /// Tests whether this configuration belongs to the recurrent sandpile group using
    /// Dhar's burning algorithm.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_recurrent(&self) -> bool {
        dhar_burning_array(&self.heights).0
    }

    /// Returns `(is_recurrent, wave_depths, max_wave_depth)` from Dhar's burning algorithm.
    #[must_use]
    #[verified_engine::verified]
    pub fn burning_waves(&self) -> (bool, [[u16; W]; H], u16) {
        dhar_burning_array(&self.heights)
    }

    /// Computes the exact group order $|G| = \det(\Delta)$ via Bareiss elimination when
    /// `W * H <= MAX_BAREISS_CELLS`.
    #[verified_engine::verified]
    pub fn group_order() -> Result<i128, SandpileError> {
        kirchhoff_determinant_bareiss(W, H)
    }

    /// Generates the cyclic subgroup $\langle g \rangle = \{e, g, g^2, \dots\}$ generated by `self`,
    /// bounded by `max_order` steps.
    ///
    /// The returned slice can be verified directly with
    /// [`pure_math::pure_math::algebra::is_subgroup`] and partitioned into cosets via
    /// [`pure_math::pure_math::algebra::generate_left_coset`].
    #[must_use]
    #[verified_engine::verified]
    pub fn cyclic_subgroup(&self, max_order: usize) -> Vec<Self> {
        let e = Self::identity();
        let cap = max_order.max(1);
        let mut orbit = Vec::with_capacity(cap.min(64));
        orbit.push(e);
        let mut acc = Self::operate(&e, self);
        while acc != e && orbit.len() < cap {
            orbit.push(acc);
            acc = Self::operate(&acc, self);
        }
        orbit
    }

    /// Evaluates the canonical cyclic homomorphism $\phi(k) = g^k$ from [`Zn<N>`] into the
    /// sandpile group using binary exponentiation.
    ///
    /// Whenever $g^N = e$, `k -> g.zn_homomorphism(k)` is an exact group homomorphism
    /// $(\mathbb{Z}_N, +) \to (G, \oplus)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn zn_homomorphism<const N: usize>(&self, k: Zn<N>) -> Self {
        let mut exp = k.value;
        let mut result = Self::identity();
        let mut base = Self::operate(&result, self);
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::operate(&result, &base);
            }
            exp >>= 1;
            if exp > 0 {
                base = Self::operate(&base, &base);
            }
        }
        result
    }
}

impl<const W: usize, const H: usize> Semigroup for SandpileElement<W, H> {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        let mut sum = [[0_u32; W]; H];
        for y in 0..H {
            for x in 0..W {
                sum[y][x] = a.heights[y][x] + b.heights[y][x];
            }
        }
        let _ = stabilize_grid_array(&mut sum);
        Self { heights: sum }
    }
}

impl<const W: usize, const H: usize> Monoid for SandpileElement<W, H> {
    #[verified_engine::verified]
    fn identity() -> Self {
        Self::identity_element()
    }
}

impl<const W: usize, const H: usize> Group for SandpileElement<W, H> {
    #[verified_engine::verified]
    fn inverse(&self) -> Self {
        let z0 = creutz_dual_zero_array::<W, H>();
        let mut r_stab = self.heights;
        let _ = stabilize_grid_array(&mut r_stab);
        let mut inv = [[0_u32; W]; H];
        for y in 0..H {
            for x in 0..W {
                inv[y][x] = 2 * z0[y][x] - r_stab[y][x];
            }
        }
        let _ = stabilize_grid_array(&mut inv);
        Self { heights: inv }
    }
}

impl<const W: usize, const H: usize> AbelianGroup for SandpileElement<W, H> {}
