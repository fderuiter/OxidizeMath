//! Static 64D 53-bit base-2 Niederreiter `GF(2)` generator tables (`< 2.2 KiB` `.rodata`).
//!
//! Implements Bratley, Fox & Niederreiter (1992), ACM TOMS 18(2), 195–213 using the first 64
//! monic irreducible polynomials over $\mathbb{F}_2\[x\]$ (degrees $1 \le e_j \le 8$).
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

/// Maximum dimension supported by [`super::NiederreiterView`] (`64`).
pub const MAX_NIEDERREITER_DIM: usize = 64;
/// Number of IEEE-754 `f64` significand bits generated per Niederreiter coordinate (`53`).
pub const NIEDERREITER_BITS: usize = 53;
const FAST_COLS: usize = 4;
const MAX_V_LEN: usize = NIEDERREITER_BITS + 10;

/// First 64 monic irreducible polynomials over $\mathbb{F}_2\[x\]$ ordered by non-decreasing degree (`128` bytes).
static NIEDERREITER_IRRED_GF2: [u16; MAX_NIEDERREITER_DIM] = [
    2, 3, 7, 11, 13, 19, 25, 31, 37, 41, 47, 55, 59, 61, 67, 73, 87, 91, 97, 103, 109, 115, 117,
    131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213, 229, 239, 241, 247, 253, 283,
    285, 299, 301, 313, 319, 333, 351, 355, 357, 361, 369, 375, 379, 391, 395, 397, 415, 419, 425,
    433, 445, 451,
];

const fn poly_degree(poly: u16) -> usize {
    (15 - poly.leading_zeros()) as usize
}

const fn gf2_poly_mul_u64(mut a: u64, mut b: u64) -> u64 {
    let mut prod = 0_u64;
    while b != 0 {
        if (b & 1) != 0 {
            prod ^= a;
        }
        a <<= 1;
        b >>= 1;
    }
    prod
}

const fn compute_niederreiter_col_slow(dim: usize, col: usize) -> u64 {
    assert!(dim < MAX_NIEDERREITER_DIM);
    if col >= NIEDERREITER_BITS {
        return 0;
    }
    if dim == 0 {
        return 1_u64 << (NIEDERREITER_BITS - 1 - col);
    }
    let irred = NIEDERREITER_IRRED_GF2[dim] as u64;
    let e = poly_degree(NIEDERREITER_IRRED_GF2[dim]);
    let mut px_q = 1_u64;
    let mut row = 0_usize;
    let mut col_word = 0_u64;

    while row < NIEDERREITER_BITS {
        px_q = gf2_poly_mul_u64(px_q, irred);
        let m = row + e;
        let mut v = [0_u8; MAX_V_LEN];
        v[m - 1] = 1;
        let max_needed = col + e;
        let mut r = m;
        while r < max_needed && r < MAX_V_LEN {
            let mut sum = 0_u8;
            let mut k = 0_usize;
            while k < m {
                if ((px_q >> k) & 1) != 0 {
                    sum ^= v[r - m + k];
                }
                k += 1;
            }
            v[r] = sum;
            r += 1;
        }
        let mut u = 0_usize;
        while u < e && (row + u) < NIEDERREITER_BITS {
            if v[col + u] != 0 {
                col_word |= 1_u64 << (NIEDERREITER_BITS - 1 - (row + u));
            }
            u += 1;
        }
        row += e;
    }
    col_word
}

const fn build_fast_gens() -> [[u64; FAST_COLS]; MAX_NIEDERREITER_DIM] {
    let mut table = [[0_u64; FAST_COLS]; MAX_NIEDERREITER_DIM];
    let mut d = 0;
    while d < MAX_NIEDERREITER_DIM {
        let mut c = 0;
        while c < FAST_COLS {
            table[d][c] = compute_niederreiter_col_slow(d, c);
            c += 1;
        }
        d += 1;
    }
    table
}

/// Static precomputed 53-bit generator columns for the 4 hottest Gray-code bits (`2,048` bytes).
static NIEDERREITER_FAST_GENS_53: [[u64; FAST_COLS]; MAX_NIEDERREITER_DIM] = build_fast_gens();

/// Returns the 53-bit base-2 Niederreiter generator column $C^{(\text{dim})}_{\cdot, \text{bit}}$.
#[inline]
#[must_use]
pub fn niederreiter_generator_53(dim: usize, bit: usize) -> u64 {
    debug_assert!(dim < MAX_NIEDERREITER_DIM);
    if bit < FAST_COLS {
        NIEDERREITER_FAST_GENS_53[dim][bit]
    } else if bit < NIEDERREITER_BITS {
        compute_niederreiter_col_slow(dim, bit)
    } else {
        0
    }
}

/// Returns total static `.rodata` bytes occupied by the 64D Niederreiter tables.
#[inline]
#[must_use]
pub fn niederreiter_static_bytes() -> usize {
    core::mem::size_of_val(&NIEDERREITER_IRRED_GF2)
        + core::mem::size_of_val(&NIEDERREITER_FAST_GENS_53)
}
