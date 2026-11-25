//! Compact 64-dimensional 53-bit Joe-Kuo Sobol direction tables (`< 3 KiB` `.rodata`).
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

/// Maximum dimension supported by [`super::SobolView`] (`64`).
pub const MAX_SOBOL_DIM: usize = 64;
/// Number of IEEE-754 `f64` significand bits generated per Sobol coordinate (`53`).
pub const SOBOL_BITS: usize = 53;
const FAST_COLS: usize = 4;

/// Compact primitive polynomial $x^s + c_1 x^{s-1} + \dots + c_{s-1} x + 1 \in \mathbb{F}_2\[x\]$
/// and encoded initial odd direction numbers $m_k = 2 \cdot \text{m\_half}\[k - 1\] + 1$.
#[derive(Clone, Copy)]
pub(crate) struct JoeKuoPoly {
    pub degree: u8,
    pub coeffs: u8,
    pub m_half: [u8; 9],
}

const fn jk(degree: u8, coeffs: u8, m: &[u16]) -> JoeKuoPoly {
    assert!(degree >= 1 && degree <= 9 && m.len() == degree as usize);
    let mut m_half = [0_u8; 9];
    let mut i = 0;
    while i < m.len() {
        assert!(m[i] % 2 == 1 && m[i] < (1_u16 << (i + 1)));
        m_half[i] = ((m[i] - 1) >> 1) as u8;
        i += 1;
    }
    JoeKuoPoly {
        degree,
        coeffs,
        m_half,
    }
}

/// Static 63-entry Joe-Kuo primitive polynomial table for dimensions `d = 2..=64` (`693` bytes).
pub(crate) static JOE_KUO_D2_TO_D64: [JoeKuoPoly; 63] = [
    jk(1, 0, &[1]),
    jk(2, 1, &[1, 3]),
    jk(3, 1, &[1, 3, 1]),
    jk(3, 2, &[1, 1, 1]),
    jk(4, 1, &[1, 1, 3, 3]),
    jk(4, 4, &[1, 3, 5, 13]),
    jk(5, 2, &[1, 1, 5, 5, 17]),
    jk(5, 4, &[1, 1, 5, 5, 5]),
    jk(5, 7, &[1, 1, 7, 11, 19]),
    jk(5, 11, &[1, 1, 5, 1, 1]),
    jk(5, 13, &[1, 1, 1, 3, 11]),
    jk(5, 14, &[1, 3, 5, 5, 31]),
    jk(6, 1, &[1, 3, 3, 9, 7, 49]),
    jk(6, 13, &[1, 1, 1, 15, 21, 21]),
    jk(6, 16, &[1, 3, 1, 13, 27, 49]),
    jk(6, 19, &[1, 1, 1, 15, 7, 5]),
    jk(6, 22, &[1, 3, 1, 15, 13, 25]),
    jk(6, 25, &[1, 1, 5, 5, 19, 61]),
    jk(7, 1, &[1, 3, 7, 11, 23, 15, 103]),
    jk(7, 4, &[1, 3, 7, 13, 13, 15, 69]),
    jk(7, 7, &[1, 1, 3, 13, 7, 35, 63]),
    jk(7, 8, &[1, 3, 5, 9, 1, 25, 53]),
    jk(7, 14, &[1, 3, 1, 13, 9, 35, 107]),
    jk(7, 19, &[1, 3, 1, 5, 27, 61, 31]),
    jk(7, 21, &[1, 1, 5, 11, 19, 41, 61]),
    jk(7, 28, &[1, 3, 5, 3, 3, 13, 69]),
    jk(7, 31, &[1, 1, 7, 13, 1, 19, 1]),
    jk(7, 32, &[1, 3, 7, 5, 13, 19, 59]),
    jk(7, 37, &[1, 1, 3, 9, 25, 29, 41]),
    jk(7, 41, &[1, 3, 5, 13, 23, 1, 55]),
    jk(7, 42, &[1, 3, 7, 3, 13, 59, 17]),
    jk(7, 50, &[1, 3, 3, 3, 31, 17, 77]),
    jk(7, 55, &[1, 3, 3, 3, 25, 21, 5]),
    jk(7, 56, &[1, 3, 3, 3, 5, 55, 97]),
    jk(7, 59, &[1, 1, 1, 15, 1, 33, 11]),
    jk(7, 62, &[1, 1, 1, 9, 9, 49, 27]),
    jk(8, 14, &[1, 1, 1, 9, 13, 23, 127, 97]),
    jk(8, 21, &[1, 1, 1, 9, 13, 25, 55, 11]),
    jk(8, 22, &[1, 1, 1, 3, 27, 5, 55, 103]),
    jk(8, 38, &[1, 1, 1, 3, 1, 25, 17, 79]),
    jk(8, 47, &[1, 1, 5, 3, 9, 25, 97, 23]),
    jk(8, 49, &[1, 1, 1, 3, 1, 43, 93, 37]),
    jk(8, 50, &[1, 1, 1, 3, 1, 41, 3, 45]),
    jk(8, 52, &[1, 1, 7, 7, 11, 9, 63, 33]),
    jk(8, 56, &[1, 1, 1, 7, 1, 3, 121, 193]),
    jk(8, 67, &[1, 1, 7, 7, 19, 33, 33, 3]),
    jk(8, 70, &[1, 1, 7, 5, 21, 13, 127, 139]),
    jk(8, 84, &[1, 3, 3, 7, 1, 9, 79, 57]),
    jk(8, 97, &[1, 1, 3, 7, 1, 53, 103, 101]),
    jk(8, 103, &[1, 3, 3, 3, 1, 5, 85, 225]),
    jk(8, 115, &[1, 1, 3, 7, 9, 63, 123, 191]),
    jk(8, 122, &[1, 3, 7, 15, 11, 47, 107, 245]),
    jk(9, 8, &[1, 1, 3, 7, 9, 21, 59, 147, 305]),
    jk(9, 13, &[1, 3, 7, 5, 1, 15, 73, 193, 3]),
    jk(9, 16, &[1, 3, 7, 11, 5, 25, 77, 45, 17]),
    jk(9, 22, &[1, 3, 7, 11, 5, 3, 59, 167, 409]),
    jk(9, 25, &[1, 3, 5, 9, 13, 3, 81, 73, 3]),
    jk(9, 44, &[1, 3, 5, 1, 25, 11, 91, 51, 505]),
    jk(9, 47, &[1, 3, 7, 3, 29, 51, 65, 215, 333]),
    jk(9, 52, &[1, 3, 5, 1, 25, 11, 17, 129, 429]),
    jk(9, 55, &[1, 3, 7, 3, 29, 51, 67, 235, 25]),
    jk(9, 59, &[1, 3, 5, 9, 1, 1, 7, 5, 221]),
    jk(9, 62, &[1, 3, 7, 3, 29, 27, 57, 127, 415]),
];

const fn compute_sobol_direction_slow(dim: usize, bit: usize) -> u64 {
    assert!(dim < MAX_SOBOL_DIM);
    if bit >= SOBOL_BITS {
        return 0;
    }
    if dim == 0 {
        return 1_u64 << (SOBOL_BITS - 1 - bit);
    }
    let poly = JOE_KUO_D2_TO_D64[dim - 1];
    let s = poly.degree as usize;
    let mut ring = [0_u64; 9];
    let mut k = 0;
    while k < s {
        let m_k = ((poly.m_half[k] as u64) << 1) | 1;
        ring[k] = m_k << (SOBOL_BITS - 1 - k);
        if k == bit {
            return ring[k];
        }
        k += 1;
    }
    while k <= bit {
        let oldest = ring[k % s];
        let mut val = oldest ^ (oldest >> s);
        let mut j = 1;
        while j < s {
            if ((poly.coeffs >> (s - 1 - j)) & 1) != 0 {
                val ^= ring[(k + s - j) % s];
            }
            j += 1;
        }
        ring[k % s] = val;
        if k == bit {
            return val;
        }
        k += 1;
    }
    0
}

const fn build_fast_dirs() -> [[u64; FAST_COLS]; MAX_SOBOL_DIM] {
    let mut table = [[0_u64; FAST_COLS]; MAX_SOBOL_DIM];
    let mut d = 0;
    while d < MAX_SOBOL_DIM {
        let mut b = 0;
        while b < FAST_COLS {
            table[d][b] = compute_sobol_direction_slow(d, b);
            b += 1;
        }
        d += 1;
    }
    table
}

/// Static precomputed 53-bit direction numbers for the 4 hottest Gray-code bits (`2,048` bytes).
static SOBOL_FAST_DIRS_53: [[u64; FAST_COLS]; MAX_SOBOL_DIM] = build_fast_dirs();

/// Returns the 53-bit Sobol direction word $v_{d, \text{bit}}$ for `dim < 64` and `bit < 53`.
#[inline]
#[must_use]
pub fn sobol_direction_53(dim: usize, bit: usize) -> u64 {
    debug_assert!(dim < MAX_SOBOL_DIM);
    if bit < FAST_COLS {
        SOBOL_FAST_DIRS_53[dim][bit]
    } else if bit < SOBOL_BITS {
        compute_sobol_direction_slow(dim, bit)
    } else {
        0
    }
}

/// Returns total static `.rodata` bytes occupied by the 64D Sobol tables.
#[inline]
#[must_use]
pub fn sobol_static_bytes() -> usize {
    core::mem::size_of_val(&JOE_KUO_D2_TO_D64) + core::mem::size_of_val(&SOBOL_FAST_DIRS_53)
}
