//! Static 64-prime table and packed Faure prime-radix permutations (`~9.4 KiB` `.rodata`).
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use crate::scramble::splitmix64;

/// Maximum dimension supported by [`super::HaltonView`] (`64`).
pub const MAX_HALTON_DIM: usize = 64;
const TOTAL_PRIME_SUM: usize = 8_893;
const FIRST_9BIT_PRIME_DIM: usize = 54; // p_55 = 257 (0-based dim 54)
const HI_BIT_BYTES: usize = 352; // ceil(2812 / 8)

/// First 64 prime bases ($p_1 = 2 \dots p_{64} = 311$) for 64D Halton sampling (`256` bytes).
pub static FIRST_64_PRIMES: [u32; MAX_HALTON_DIM] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

struct FaureTables {
    offsets: [u16; MAX_HALTON_DIM + 1],
    low_bytes: [u8; TOTAL_PRIME_SUM],
    hi_bits: [u8; HI_BIT_BYTES],
}

const fn compute_faure_base(base: usize) -> [u16; 312] {
    let mut out = [0_u16; 312];
    if base == 2 {
        out[0] = 0;
        out[1] = 1;
        return out;
    }
    if base.is_multiple_of(2) {
        let half = base / 2;
        let sub = compute_faure_base(half);
        let mut i = 0;
        while i < half {
            out[i] = 2 * sub[i];
            out[half + i] = 2 * sub[i] + 1;
            i += 1;
        }
    } else {
        let mid = (base - 1) / 2;
        let prev = compute_faure_base(base - 1);
        let mut i = 0;
        while i < mid {
            let v = prev[i];
            out[i] = if v as usize >= mid { v + 1 } else { v };
            i += 1;
        }
        out[mid] = mid as u16;
        while i < base - 1 {
            let v = prev[i];
            out[i + 1] = if v as usize >= mid { v + 1 } else { v };
            i += 1;
        }
    }
    out
}

const fn build_faure_tables() -> FaureTables {
    let mut offsets = [0_u16; MAX_HALTON_DIM + 1];
    let mut low_bytes = [0_u8; TOTAL_PRIME_SUM];
    let mut hi_bits = [0_u8; HI_BIT_BYTES];
    let mut cursor = 0_usize;
    let mut hi_base_offset = 0_usize;
    let mut dim = 0_usize;
    while dim < MAX_HALTON_DIM {
        offsets[dim] = cursor as u16;
        let p = FIRST_64_PRIMES[dim] as usize;
        let perm = compute_faure_base(p);
        if dim == FIRST_9BIT_PRIME_DIM {
            hi_base_offset = cursor;
        }
        let mut d = 0_usize;
        while d < p {
            let val = perm[d];
            low_bytes[cursor + d] = (val & 0xFF) as u8;
            if dim >= FIRST_9BIT_PRIME_DIM && (val & 0x100) != 0 {
                let bit_idx = (cursor + d) - hi_base_offset;
                hi_bits[bit_idx >> 3] |= 1_u8 << (bit_idx & 7);
            }
            d += 1;
        }
        cursor += p;
        dim += 1;
    }
    assert!(cursor == TOTAL_PRIME_SUM);
    offsets[MAX_HALTON_DIM] = cursor as u16;
    FaureTables {
        offsets,
        low_bytes,
        hi_bits,
    }
}

static FAURE_TABLES: FaureTables = build_faure_tables();

/// Looks up the static Faure prime-radix permutation $\sigma_{p_{\text{dim}}}(\text{digit})$ in $O(1)$.
#[inline]
#[must_use]
pub fn faure_permute(dim: usize, digit: u32) -> u32 {
    debug_assert!(dim < MAX_HALTON_DIM);
    debug_assert!(digit < FIRST_64_PRIMES[dim]);
    let base_off = FAURE_TABLES.offsets[dim] as usize;
    let idx = base_off + (digit as usize);
    let mut val = FAURE_TABLES.low_bytes[idx] as u32;
    if dim >= FIRST_9BIT_PRIME_DIM {
        let hi_start = FAURE_TABLES.offsets[FIRST_9BIT_PRIME_DIM] as usize;
        let bit_idx = idx - hi_start;
        if ((FAURE_TABLES.hi_bits[bit_idx >> 3] >> (bit_idx & 7)) & 1) != 0 {
            val |= 0x100;
        }
    }
    val
}

/// Applies static Faure permutation composed with a stateless bijective Owen digit permutation in $\mathbb{Z}_p$.
#[inline]
#[must_use]
pub fn faure_owen_permute(
    dim: usize,
    prime: u32,
    depth: u32,
    prefix_hash: u64,
    digit: u32,
    scramble_seed: Option<u64>,
) -> u32 {
    let s0 = faure_permute(dim, digit);
    let Some(seed) = scramble_seed else {
        return s0;
    };
    let p = prime as u64;
    let h1 = splitmix64(
        seed ^ ((dim as u64).wrapping_add(1)).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (depth as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ prefix_hash,
    );
    let h2 = splitmix64(h1 ^ 0xA076_1D64_78BD_642F);
    let a1 = if p > 2 { 1 + (h1 % (p - 1)) } else { 1 };
    let b1 = (h1 >> 32) % p;
    let t1 = faure_permute(dim, ((a1 * (s0 as u64) + b1) % p) as u32);
    let a2 = if p > 2 { 1 + (h2 % (p - 1)) } else { 1 };
    let b2 = (h2 >> 32) % p;
    let t2 = faure_permute(dim, ((a2 * (t1 as u64) + b2) % p) as u32);
    let h3 = splitmix64(h2 ^ 0xE703_7ED1_A0B4_28DB);
    let a3 = if p > 2 { 1 + (h3 % (p - 1)) } else { 1 };
    let b3 = (h3 >> 32) % p;
    faure_permute(dim, ((a3 * (t2 as u64) + b3) % p) as u32)
}

/// Returns total static `.rodata` bytes occupied by the 64D prime and Faure tables.
#[inline]
#[must_use]
pub fn faure_static_bytes() -> usize {
    core::mem::size_of_val(&FIRST_64_PRIMES) + core::mem::size_of_val(&FAURE_TABLES)
}
