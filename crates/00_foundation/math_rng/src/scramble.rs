//! Stateless 53-bit nested hash Owen scrambling and Cranley-Patterson torus shifts.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use crate::scalar::RngScalar;

const MASK_53: u64 = (1_u64 << 53) - 1;
const INV_TWO_POW_53: f64 = 1.0 / ((1_u64 << 53) as f64);

/// Deterministic 64-bit SplitMix64 avalanche mixer used to derive per-(seed, axis, node) hashes.
#[inline]
#[must_use]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Applies stateless 53-bit nested uniform (Owen) scrambling to a 53-bit dyadic integer.
///
/// Reversing the 53-bit word places the MSB ($2^{-1}$) at bit 0; applying prefix-causal
/// additions and XOR-multiplications by **even** 64-bit constants propagates carries strictly
/// from MSB prefixes to finer dyadic subtrees, guaranteeing a bijection on $[0, 2^{53})$ that
/// preserves every base-2 $(t, m, s)$-net stratification bin (Laine & Karras 2011; Burley 2020).
#[inline]
#[must_use]
pub const fn owen_scramble_53(raw_53: u64, seed: u64, dim: usize) -> u64 {
    debug_assert!(raw_53 <= MASK_53);
    let k0 = splitmix64(seed ^ ((dim as u64).wrapping_add(1)).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let k1 = splitmix64(k0 ^ 0xA076_1D64_78BD_642F);

    let mut y = (raw_53 << 11).reverse_bits();
    y = y.wrapping_add(k0);
    y ^= y.wrapping_mul(0x6C50_B47C_9952_8876);
    y = y.wrapping_add(k1);
    y ^= y.wrapping_mul(0xB82F_1E52_D614_F33A);
    y ^= y.wrapping_mul(0xC7EF_372A_E396_C58E);
    y ^= y.wrapping_mul(0x93E1_675A_2F47_4156);

    let out = (y.reverse_bits() >> 11) & MASK_53;
    debug_assert!(out <= MASK_53);
    out
}

/// Applies a Cranley-Patterson torus shift `(unit + shift) mod 1` and clamps strictly inside `(0, 1)`.
#[inline]
#[must_use]
pub fn torus_shift_open<T: RngScalar>(unit_f64: f64, shift: f64) -> T {
    debug_assert!((0.0..=1.0).contains(&unit_f64));
    let mut sum = unit_f64 + shift;
    if sum >= 1.0 {
        sum -= sum as u64 as f64;
    } else if sum < 0.0 {
        sum -= (sum as i64 as f64) - 1.0;
    }
    T::clamp_open_unit_f64(sum)
}

/// Converts a 53-bit dyadic integer (optionally Owen-scrambled) plus Cranley-Patterson `shift`
/// into a strictly open `(0, 1)` scalar `T`.
#[inline]
#[must_use]
pub fn dyadic_53_to_shifted_open<T: RngScalar>(
    raw_53: u64,
    dim: usize,
    scramble_seed: Option<u64>,
    shift: f64,
) -> T {
    let bits = match scramble_seed {
        Some(seed) => owen_scramble_53(raw_53, seed, dim),
        None => raw_53,
    };
    if shift == 0.0 {
        return T::from_dyadic_53_open(bits);
    }
    let unit = ((bits as f64) + 0.5) * INV_TWO_POW_53;
    torus_shift_open(unit, shift)
}
